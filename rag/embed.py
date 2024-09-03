from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

def get_embedding_model(embedding_model_name, model_kwargs, encode_kwargs):
    if embedding_model_name == "text-embedding-ada-002":
        embedding_model = OpenAIEmbeddings(
            model=embedding_model_name,
        )
    return embedding_model

class EmbedChunks:
    def __init__(self, model_name):
        if model_name == "text-embedding-ada-002":
            self.embedding_model = OpenAIEmbeddings(
                model=model_name
            )
    
    def __call__(self, chunk):
        embedding = self.embedding_model.embed_documents([chunk["text"]])
        return {"text": chunk["text"],
                "source": chunk["source"],
                "id_constitucion": chunk["id_constitucion"],
                "id_capitulo": chunk["id_capitulo"],
                "id_articulo": chunk["id_articulo"],
                "embedding": embedding[0]}