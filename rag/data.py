from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_article(record):

    article_content = record['CONTENIDO_ARTICULO'].replace('\\n', '\n')
    source_text = f"{record['NOMBRE_CAPITULO']}: {record['TITULO_CAPITULO']}, {record['NOMBRE_ARTICULO']}"
    article_text = f"{record['NOMBRE_CAPITULO']}: {record['TITULO_CAPITULO']}\n{record['NOMBRE_ARTICULO']}\n\n{article_content}"
    
    id_constitucion = record['ID_CONSTITUCION']
    id_capitulo = record['ID_CAPITULO']
    id_articulo = record['ID_ARTICULO']
    
    return { "id_constitucion": id_constitucion,
            "id_capitulo": id_capitulo,
            "id_articulo": id_articulo,
            "source": source_text,
            "text": article_text,}

def chunk_article(article, chunk_size=300, chunk_overlap=30):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[" ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len)
    chunks = text_splitter.create_documents(
        texts=[article["text"]], 
        metadatas=[{"source": article["source"],
                    "id_constitucion": article["id_constitucion"],
                    "id_capitulo": article["id_capitulo"],
                    "id_articulo": article["id_articulo"]
                    }])
    return [{"text": chunk.page_content,
             "source": chunk.metadata["source"],
             "id_constitucion": chunk.metadata["id_constitucion"],
             "id_capitulo": chunk.metadata["id_capitulo"],
             "id_articulo": chunk.metadata["id_articulo"]} for chunk in chunks]