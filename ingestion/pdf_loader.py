from langchain_core.documents import Document
from langchain_community.document_loaders.pdf import PyPDFLoader
import os
import re
import wordninja


def clean_text(text: str) -> str:
    # Collapse newlines into spaces
    text = text.replace("\n", " ")
    # Insert a space where a lowercase letter is directly followed by an uppercase letter
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    # Insert a space between a letter and a digit stuck together
    text = re.sub(r"(?<=[a-zA-Z])(?=[0-9])", " ", text)
    text = re.sub(r"(?<=[0-9])(?=[a-zA-Z])", " ", text)
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()
    # Fix remaining glued-together lowercase words
    text = fix_long_words(text)
    return text


def fix_long_words(text):
    words = text.split(" ")
    fixed = []
    for w in words:
        if len(w) > 15 and w.isalpha():  # suspiciously long, likely glued
            fixed.append(" ".join(wordninja.split(w)))
        else:
            fixed.append(w)
    return " ".join(fixed)


def load_pdfs(directory: str):
    documents = []
    for file in os.listdir(directory):
        if file.lower().endswith(".pdf"):
            path = os.path.join(directory, file)
            loader = PyPDFLoader(path)
            pages = loader.load()
            full_text = " ".join(clean_text(p.page_content) for p in pages)
            documents.append(
                Document(
                    page_content=full_text,
                    metadata={"source_filename": file}
                )
            )
    return documents