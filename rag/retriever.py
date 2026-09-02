def retrieve_chunks(vectorstore, query, k):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    try:
        return retriever.invoke(query)
    except:
        return retriever.get_relevant_documents(query)
