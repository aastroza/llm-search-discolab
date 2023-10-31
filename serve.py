from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from modal import asgi_app, Secret, Image, Stub
from dotenv import load_dotenv; load_dotenv()
import re
import time
import asyncio
import json
import yaml

from rag.embed import EmbedChunks
from rag.generate import get_sources_and_context, QueryAgent, ComparisonAgent
from rag.prompts import DOCUMENT_QA_SYSTEM_PROMPT, DOCUMENT_QA_USER_PROMPT_TEMPLATE, FINAL_RESPONSE_SYSTEM_PROMPT_TEMPLATE
from rag.config import MAX_CONTEXT_LENGTHS, CONFIG, CONSTITUCIONES

# Creates the FastAPI web server.
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image = (
    Image.debian_slim()
    .pip_install(
        "openai",
        "langchain",
        "python-dotenv",
        "psycopg2-binary",
        "pgvector",
        "jinja2",
        "tiktoken"
    )
)

stub = Stub("new-rag-discolab")

class Query(BaseModel):
    query: str
    document1_id: int = 1
    document2_id: int = 2

@app.get("/ping")
async def ping():
    return "pong"

@app.post("/stream")
def stream(query: Query) -> StreamingResponse:

    constitucion1_agent = QueryAgent(embedding_model_name=CONFIG["embedding_model"],
                                llm=CONFIG["chat_model"],
                                temperature=CONFIG["temperature"],
                                max_context_length=MAX_CONTEXT_LENGTHS[CONFIG["chat_model"]],
                                system_content=DOCUMENT_QA_SYSTEM_PROMPT,
                                assistant_content="",
                                constitucion_id=query.document1_id)
    
    constitucion2_agent = QueryAgent(embedding_model_name=CONFIG["embedding_model"],
                                llm=CONFIG["chat_model"],
                                temperature=CONFIG["temperature"],
                                max_context_length=MAX_CONTEXT_LENGTHS[CONFIG["chat_model"]],
                                system_content=DOCUMENT_QA_SYSTEM_PROMPT,
                                assistant_content="",
                                constitucion_id=query.document2_id)
    
    result1 = constitucion1_agent(query=query.query,
                                    num_chunks=3,
                                    stream=True)

    result2 = constitucion2_agent(query=query.query,
                                    num_chunks=3,
                                    stream=True)

    return StreamingResponse(
                            produce_streaming_answer(qe_result1 = result1,
                                                    qe_result2 = result2,
                                                    prompt=query.query,
                                                    document1_id=query.document1_id,
                                                    document2_id=query.document2_id),
                            media_type="text/event-stream")

def produce_streaming_answer(qe_result1, qe_result2, prompt, document1_id, document2_id):
    
    yield "\n\n**[DOCUMENT 1]**\n"
    answer = []
    for answer_piece in qe_result1["answer"]:
        answer.append(answer_piece)
        yield f'{{"content" : "{answer_piece}"}}\n'
    response_final_1 = "".join(answer)

    yield "\n\n**[SOURCES DOCUMENT 1]**\n"
    sources_idx_1 = sorted(set(re.findall(r'[\d]', response_final_1)))
    if len(sources_idx_1) > 0:
        for idx in sources_idx_1:
            source = qe_result1["sources"][int(idx)-1]
            yield f'[{idx}] {source}\n'

    yield "\n\n**[DOCUMENT 2]**\n"
    answer = []
    for answer_piece in qe_result2["answer"]:
        answer.append(answer_piece)
        yield f'{{"content" : "{answer_piece}"}}\n'
    response_final_2 = "".join(answer)

    yield "\n\n**[SOURCES DOCUMENT 2]**\n"
    sources_idx_2 = sorted(set(re.findall(r'[\d]', response_final_2)))
    if len(sources_idx_2) > 0:
        for idx in sources_idx_2:
            source = qe_result1["sources"][int(idx)-1]
            yield f'[{idx}] {source}\n'
    
    if len(sources_idx_1) + len(sources_idx_2) > 0:

        final_agent = ComparisonAgent(embedding_model_name=CONFIG["embedding_model"],
                        llm=CONFIG["chat_model"],
                        temperature=CONFIG["temperature"],
                        max_context_length=MAX_CONTEXT_LENGTHS[CONFIG["chat_model"]],
                        system_content=FINAL_RESPONSE_SYSTEM_PROMPT_TEMPLATE.render(document1_title=CONSTITUCIONES[str(document1_id)],
                                                                                    document2_title=CONSTITUCIONES[str(document2_id)]),
                        assistant_content="")

        yield "\n\n**[FINAL RESPONSE]**\n"
        response_final = final_agent(query=prompt,
                                    first_response=response_final_1,
                                    second_response=response_final_2,
                                    stream=True)
        for answer_piece in response_final["answer"]:
            yield f'{{"content" : "{answer_piece}"}}\n'
    
    yield "\n\n**[END]**\n"


@stub.function(
    image=image,
    secret=Secret.from_name("new-discolab"),
    keep_warm=1,
)

@asgi_app()
def api():
    return app