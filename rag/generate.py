import psycopg2
from pgvector.psycopg2 import register_vector
import os
import numpy as np
from openai import OpenAI

import time
import re

from dotenv import load_dotenv

from rag.embed import get_embedding_model
from rag.prompts import DOCUMENT_QA_USER_PROMPT_TEMPLATE, DOCUMENT_QA_REFINE_USER_PROMPT_TEMPLATE, FINAL_RESPONSE_USER_PROMPT_TEMPLATE
from rag.config import CONFIG

load_dotenv()

client = OpenAI()

def get_sources_and_context(query, embedding_model, num_chunks=3, id_constitucion=1):
    embedding = np.array(embedding_model.embed_query(query))
    with psycopg2.connect(os.environ["DB_CONNECTION_STRING"]) as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
            register_vector(conn)
            cur.execute(
                "SELECT id, text, source FROM document WHERE id_constitucion = %s ORDER BY embedding <=> %s LIMIT %s",
                (id_constitucion, embedding, num_chunks),
            )
            rows = cur.fetchall()
            document_ids = [row[0] for row in rows]
            context = ''.join([f"Source {i+1}:\n{row[1]}\n" for i,row in enumerate(rows)])
            sources = [row[2] for row in rows]
    return document_ids, sources, context

def response_stream(response):
    for chunk in response:
        #print(chunk)
        if hasattr(chunk.choices[0].delta, 'content'):
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

def prepare_response(response, stream):
    if stream:
        return response_stream(response)
    else:
        return response["choices"][-1]["message"]["content"]

def generate_response(
    llm,
    temperature=0.0,
    stream=False,
    system_content="",
    assistant_content="",
    user_content="",
    max_retries=3,
    retry_interval=60,
):
    """Generate response from an LLM."""
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=llm,
                temperature=temperature,
                stream=stream,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": user_content},
                ],
            )
            return prepare_response(response=response, stream=stream)

        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(retry_interval)  # default is per-minute rate limits
            retry_count += 1
    return ""

class QueryAgent:
    def __init__(
        self,
        embedding_model_name="text-embedding-ada-002",
        llm="gpt-3.5-turbo",
        temperature=0.0,
        max_context_length=4096,
        system_content="",
        assistant_content="",
        constitucion_id=1
    ):
        # Embedding model
        self.embedding_model = get_embedding_model(
            embedding_model_name=embedding_model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"device": "cuda", "batch_size": 100},
        )

        # LLM
        self.llm = llm
        self.temperature = temperature
        self.context_length = max_context_length - len(system_content + assistant_content)
        self.system_content = system_content
        self.assistant_content = assistant_content
        self.constitucion_id = constitucion_id

    def __call__(self, query, num_chunks=3, stream=True):
        # Get sources and context
        document_ids, sources, context = get_sources_and_context(
            query=query, embedding_model=self.embedding_model, num_chunks=num_chunks, id_constitucion=self.constitucion_id
        )

        # Generate response
        user_content = DOCUMENT_QA_USER_PROMPT_TEMPLATE.render(context=context,
                                                                query=query
                                                            )
        answer = generate_response(
            llm=self.llm,
            temperature=self.temperature,
            stream=stream,
            system_content=self.system_content,
            assistant_content=self.assistant_content,
            user_content=user_content[: self.context_length],
        )

        if CONFIG["refine"]:

            user_refine_content = DOCUMENT_QA_REFINE_USER_PROMPT_TEMPLATE.render(existing_answer=answer,
                                                                                context=context,
                                                                                query=query
                                                                            )
            
            refined_answer = generate_response(
                llm=self.llm,
                temperature=self.temperature,
                stream=stream,
                system_content=self.system_content,
                assistant_content=self.assistant_content,
                user_content=user_refine_content[: self.context_length],
            )
            # Result
            result = {
                "question": query,
                "sources": sources,
                "document_ids": document_ids,
                "answer": refined_answer,
                "answer": answer,
                "llm": self.llm,
                "user_content": user_content,
                "user_refine_content": user_refine_content
            }

        else:
            result = {
                "question": query,
                "sources": sources,
                "document_ids": document_ids,
                "answer": answer,
                "llm": self.llm,
                "user_content": user_content,
            }

        return result

class ComparisonAgent:
    def __init__(
        self,
        embedding_model_name="text-embedding-ada-002",
        llm="gpt-4o-mini",
        temperature=0.0,
        max_context_length=4096,
        system_content="",
        assistant_content=""
    ):
        # Embedding model
        self.embedding_model = get_embedding_model(
            embedding_model_name=embedding_model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"device": "cuda", "batch_size": 100},
        )

        # LLM
        self.llm = llm
        self.temperature = temperature
        self.context_length = max_context_length - len(system_content + assistant_content)
        self.system_content = system_content
        self.assistant_content = assistant_content

    def __call__(self, query, first_response, second_response, stream=True):

        # Generate response
        user_content = FINAL_RESPONSE_USER_PROMPT_TEMPLATE .render(query=query,
                                                               first_response=re.sub(r'\[\d+\]', '', first_response),
                                                               second_response=re.sub(r'\[\d+\]', '', second_response))
        answer = generate_response(
            llm=self.llm,
            temperature=self.temperature,
            stream=stream,
            system_content=self.system_content,
            assistant_content=self.assistant_content,
            user_content=user_content[: self.context_length],
        )

        # Result
        result = {
            "question": query,
            "answer": answer,
            "llm": self.llm,
            "user_content": user_content,
            "system_content": self.system_content
        }
        return result