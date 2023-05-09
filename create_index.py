import os

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


INDEX_ROOT = "./custom_index"


def split_sources(sources):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks


def create_index(chunks_):
    texts = []
    meta = []
    for chunk in chunks_:
        for doc in chunk:
            texts.append(doc.page_content)
            meta.append(doc.metadata)
    return FAISS.from_texts(texts, LlamaCppEmbeddings(model_path='./models/ggml-model-q4_0.bin'), metadatas=meta)


if __name__ == "__main__":

    docs_path = "./docs/"
    all_chunks = []
    pdf_files = [
        os.path.join(docs_path, f)
        for f in os.listdir(docs_path)
        if os.path.isfile(os.path.join(docs_path, f)) and f.lower().endswith(".pdf")
    ]

    for file in pdf_files:
        loader = PyPDFLoader(file)
        docs = loader.load()
        chunks = split_sources(docs)
        all_chunks.append(chunks)

    print("indexing...")
    index = create_index(all_chunks)

    print(f"saving to {INDEX_ROOT}...")
    index.save_local(INDEX_ROOT)
