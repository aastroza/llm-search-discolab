from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from modal import asgi_app, Secret, Image, Stub
from dotenv import load_dotenv; load_dotenv()
import re

from rag.generate import QueryAgent, ComparisonAgent
from rag.prompts import DOCUMENT_QA_SYSTEM_PROMPT, FINAL_RESPONSE_SYSTEM_PROMPT_TEMPLATE
from rag.config import MAX_CONTEXT_LENGTHS, CONFIG, CONSTITUCIONES
from rag.magic import get_language

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
        "tiktoken",
        "marvin==1.5.1",
        "pydantic-settings"
    )
)

stub = Stub("new-rag-discolab")

class Query(BaseModel):
    query: str
    document1_id: int = 1
    document2_id: int = 2

class Reply(BaseModel):
    query: str
    document1_id: int = 1
    document2_id: int = 2
    answer1: str
    answer2: str
    sources1: str
    sources2: str
    answer_comparison: str

@app.get("/ping")
async def ping():
    return "pong"

@app.post("/stream")
def stream(query: Query) -> StreamingResponse:

    print(f'Query: {query.query}')

    constitucion1_agent = QueryAgent(embedding_model_name=CONFIG["embedding_model"],
                                llm=CONFIG["chat_model"],
                                temperature=CONFIG["temperature"],
                                max_context_length=MAX_CONTEXT_LENGTHS[CONFIG["chat_model"]],
                                system_content=DOCUMENT_QA_SYSTEM_PROMPT.render(language=get_language(query.query)),
                                assistant_content="",
                                constitucion_id=query.document1_id)
    
    constitucion2_agent = QueryAgent(embedding_model_name=CONFIG["embedding_model"],
                                llm=CONFIG["chat_model"],
                                temperature=CONFIG["temperature"],
                                max_context_length=MAX_CONTEXT_LENGTHS[CONFIG["chat_model"]],
                                system_content=DOCUMENT_QA_SYSTEM_PROMPT.render(language=get_language(query.query)),
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

@app.post("/chat")
def stream(query: Query) -> Reply:

    print(f'Query: {query.query}')

    constitucion1_agent = QueryAgent(embedding_model_name=CONFIG["embedding_model"],
                                llm=CONFIG["chat_model"],
                                temperature=CONFIG["temperature"],
                                max_context_length=MAX_CONTEXT_LENGTHS[CONFIG["chat_model"]],
                                system_content=DOCUMENT_QA_SYSTEM_PROMPT.render(language=get_language(query.query)),
                                assistant_content="",
                                constitucion_id=query.document1_id)
    
    constitucion2_agent = QueryAgent(embedding_model_name=CONFIG["embedding_model"],
                                llm=CONFIG["chat_model"],
                                temperature=CONFIG["temperature"],
                                max_context_length=MAX_CONTEXT_LENGTHS[CONFIG["chat_model"]],
                                system_content=DOCUMENT_QA_SYSTEM_PROMPT.render(language=get_language(query.query)),
                                assistant_content="",
                                constitucion_id=query.document2_id)
    
    result1 = constitucion1_agent(query=query.query,
                                    num_chunks=3,
                                    stream=False)

    print(f'First answer: {result1["answer"]}')

    result2 = constitucion2_agent(query=query.query,
                                    num_chunks=3,
                                    stream=False)

    print(f'Second answer: {result2["answer"]}')

    result3, sources1, sources2 = produce_answer(qe_result1 = result1,
                                                qe_result2 = result2,
                                                prompt=query.query,
                                                document1_id=query.document1_id,
                                                document2_id=query.document2_id)
                                                    
    return Reply(query=query.query,
                document1_id=query.document1_id,
                document2_id=query.document2_id,
                answer1=result1["answer"],
                answer2=result2["answer"],
                sources1=sources1,
                sources2=sources2,
                answer_comparison=result3["answer"])

def produce_streaming_answer(qe_result1, qe_result2, prompt, document1_id, document2_id):
    
    yield "\n\n**[DOCUMENT 1]**\n"
    answer = []
    for answer_piece in qe_result1["answer"]:
        answer.append(answer_piece)
        yield f'{{"content" : "{answer_piece}"}}\n'
    response_final_1 = "".join(answer)

    yield "\n\n**[SOURCES DOCUMENT 1]**\n"
    sources_idx_1 = sorted(set(re.findall(r'\[(\d)\]', response_final_1)))
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
    sources_idx_2 = sorted(set(re.findall(r'\[(\d)\]', response_final_2)))
    if len(sources_idx_2) > 0:
        for idx in sources_idx_2:
            source = qe_result2["sources"][int(idx)-1]
            yield f'[{idx}] {source}\n'
    
    if len(sources_idx_1) + len(sources_idx_2) > 0:

        final_agent = ComparisonAgent(embedding_model_name=CONFIG["embedding_model"],
                        llm=CONFIG["chat_model"],
                        temperature=CONFIG["temperature"],
                        max_context_length=MAX_CONTEXT_LENGTHS[CONFIG["chat_model"]],
                        system_content=FINAL_RESPONSE_SYSTEM_PROMPT_TEMPLATE.render(document1_title=CONSTITUCIONES[str(document1_id)],
                                                                                    document2_title=CONSTITUCIONES[str(document2_id)],
                                                                                    language=get_language(prompt)),
                        assistant_content="")

        yield "\n\n**[FINAL RESPONSE]**\n"
        response_final = final_agent(query=prompt,
                                    first_response=response_final_1,
                                    second_response=response_final_2,
                                    stream=True)
        for answer_piece in response_final["answer"]:
            yield f'{{"content" : "{answer_piece}"}}\n'

    else:

        yield "\n\n**[FINAL RESPONSE]**\n"
        yield f'{{"content" : ""}}\n'

    yield "\n\n**[END]**\n"

def produce_answer(qe_result1, qe_result2, prompt, document1_id, document2_id):
    
    print(qe_result1)
    sources_idx_1 = sorted(set(re.findall(r'\[(\d)\]', qe_result1['answer'])))
    sources_text_1 = ''
    print(sources_idx_1)
    if len(sources_idx_1) > 0:
        for idx in sources_idx_1:
            source = qe_result1["sources"][int(idx)-1]
            sources_text_1 += f'[{idx}] {source}\n'

    sources_idx_2 = sorted(set(re.findall(r'\[(\d)\]', qe_result2['answer'])))
    sources_text_2 = ''
    if len(sources_idx_2) > 0:
        for idx in sources_idx_2:
            source = qe_result2["sources"][int(idx)-1]
            sources_text_2 += f'[{idx}] {source}\n'
    
    
    if len(sources_idx_1) + len(sources_idx_2) > 0:

        final_agent = ComparisonAgent(embedding_model_name=CONFIG["embedding_model"],
                        llm=CONFIG["chat_model"],
                        temperature=CONFIG["temperature"],
                        max_context_length=MAX_CONTEXT_LENGTHS[CONFIG["chat_model"]],
                        system_content=FINAL_RESPONSE_SYSTEM_PROMPT_TEMPLATE.render(document1_title=CONSTITUCIONES[str(document1_id)],
                                                                                    document2_title=CONSTITUCIONES[str(document2_id)],
                                                                                    language=get_language(prompt)),
                        assistant_content="")


        response_final = final_agent(query=prompt,
                                    first_response=qe_result1['answer'],
                                    second_response=qe_result2['answer'],
                                    stream=False)
    
        return response_final, sources_text_1, sources_text_2


@stub.function(
    image=image,
    secret=Secret.from_name("new-discolab"),
    keep_warm=1,
)

@asgi_app()
def api():
    return app