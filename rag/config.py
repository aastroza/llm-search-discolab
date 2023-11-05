# Mappings
EMBEDDING_DIMENSIONS = {
    "text-embedding-ada-002": 1536
}
MAX_CONTEXT_LENGTHS = {
    "gpt-4": 8192,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384
}

CONFIG = {
    "chat_model": "gpt-3.5-turbo",
    "embedding_model": "text-embedding-ada-002",
    "temperature": 0.0,
    "chunk_size": 1024,
    "chunk_overlap": 0,
    "similarity_topn": 3,
    "streaming": True,
    "refine": False,
    "document1_id_default": 1,
    "document2_id_default": 2
}

CONSTITUCIONES = {
    "1": "Constitución Actual 1980",
    "2": "Anteproyecto Expertos 2023",
    "3": "Propuesta Convención 2022",
    "4": "Propuesta Bachelet 2016",
    "5": "Propuesta Constitucional 2023"
}

MARVIN_MODEL_NAME = 'openai/gpt-3.5-turbo'
MARVIN_MODEL_TEMPERATURE = 0