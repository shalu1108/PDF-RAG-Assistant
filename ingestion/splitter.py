from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_documents(documents)
