import os
import sys
import psycopg2
from pgvector.psycopg2 import register_vector

from rag.config import EMBEDDING_DIMENSIONS
from rag.utils import execute_bash

def store_pg_results(chunk):
    with psycopg2.connect(os.environ["DB_CONNECTION_STRING"]) as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
            register_vector(conn)
            #for text, source, embedding in zip(batch["text"], batch["source"], batch["embeddings"]):
            cur.execute("INSERT INTO document (id_constitucion, id_capitulo, id_articulo, text, source, embedding) VALUES (%s, %s, %s, %s, %s, %s)", (chunk["id_constitucion"], chunk["id_capitulo"], chunk["id_articulo"], chunk["text"], chunk["source"], chunk["embedding"]))


def set_index(embedded_chunks):
    # Drop current Vector DB and prepare for new one
    execute_bash(f'psql "{os.environ["DB_CONNECTION_STRING"]}" -c "DROP TABLE document;"')
    # execute_bash(
    #     f"sudo -u postgres psql -f ../migrations/vector-{EMBEDDING_DIMENSIONS[embedding_model_name]}.sql"
    # )
    with psycopg2.connect(os.environ["DB_CONNECTION_STRING"]) as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
            register_vector(conn)
            cur.execute('CREATE TABLE document (id serial primary key, id_constitucion int, id_capitulo int, id_articulo int, "text" text not null, source text not null, embedding vector(1536));')

    for i, chunk in enumerate(embedded_chunks):
        #print(f"Storing chunk {i+1}/{len(embedded_chunks)}")
        sys.stdout.write('\r'+f"Storing chunk {i+1}/{len(embedded_chunks)}")
        store_pg_results(chunk)