from utils.prompts import QA_PROMPT

def answer_question(llm, docs, question):
    context = "\n\n".join(
        f"[{d.metadata.get('source_filename')}]\n{d.page_content}"
        for d in docs
    )
    prompt = QA_PROMPT.format(context=context, question=question)
    return llm.invoke(prompt).content, prompt
