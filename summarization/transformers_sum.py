from transformers import pipeline


def summarize_chunks(chunks, model_name):

    summarizer = pipeline(
        "text2text-generation",
        model=model_name
    )

    summaries = []

    for chunk in chunks:

        # flan-t5 cannot handle very large inputs
        chunk = chunk[:1500]

        prompt = "Summarize this document content: " + chunk

        result = summarizer(
            prompt,
            max_length=150,
            min_length=40,
            do_sample=False
        )

        summaries.append(
            result[0]["generated_text"]
        )

    final_summary = "\n\n".join(summaries)

    return summaries, final_summary