from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

_embeddings = None
_vectorstore = None

def get_embeddings(embedding_model):
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return _embeddings

def get_vectorstore(chunks, persist_dir, embedding_model):
    global _vectorstore
    embeddings = get_embeddings(embedding_model)

    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )

    if chunks:
        # Clear old documents and index the new ones — no folder deletion needed
        existing_ids = _vectorstore.get()["ids"]
        if existing_ids:
            _vectorstore.delete(ids=existing_ids)
        _vectorstore.add_documents(chunks)

    return _vectorstore