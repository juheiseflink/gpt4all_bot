from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores.faiss import FAISS

from create_index import INDEX_ROOT
xxx

TEMPLATE = """
Please use the following context to answer questions.
Context: {context}
---
Question: {question}
===
"""

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
embeddings = LlamaCppEmbeddings(model_path='./models/ggml-model-q4_0.bin')
llm = GPT4All(model='./models/gpt4all-converted.bin', callback_manager=callback_manager, verbose=True)


def similarity_search(query, index):
    matched_docs_ = index.similarity_search(query, k=4)
    sources_ = []
    for doc in matched_docs_:
        sources_.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    return matched_docs_, sources_


question = input("question: ")
matched_docs, sources = similarity_search(question, FAISS.load_local(INDEX_ROOT, embeddings))
context = "\n".join([doc.page_content for doc in matched_docs])
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"]).partial(context=context)
llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run(question))
