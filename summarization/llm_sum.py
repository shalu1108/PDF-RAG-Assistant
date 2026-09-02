def summarize_with_llm(llm, chunks):
    chunk_summaries = []
    for c in chunks:
        prompt = (
            "Summarize the following section of a document. "
            "Be concise, factual, and clear.\n\n" + c
        )
        out = llm.invoke(prompt)
        chunk_summaries.append(out.content)

    combine_prompt = (
        "You are given several section summaries from one document. "
        "Combine them into a single, well-organized overall summary.\n\n"
        "Formatting rules:\n"
        "- Use short bold headings for each major section\n"
        "- Use bullet points for lists of facts, skills, or achievements\n"
        "- Do not repeat information across sections\n"
        "- Keep the tone clear and professional\n\n"
        "Section summaries:\n\n" + "\n\n".join(chunk_summaries)
    )
    final = llm.invoke(combine_prompt)
    return chunk_summaries, final.content