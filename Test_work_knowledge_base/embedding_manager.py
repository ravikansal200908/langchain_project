from langchain_community.document_loaders import TextLoader, UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


import os

persist_directory = "chroma_store"

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def ingest_documents():
    documents = []
    for file in os.listdir("data"):
        loader = UnstructuredFileLoader(os.path.join("data", file))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file  # ✅ very important!
        documents.extend(docs)

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()


def get_vector_db():
    return Chroma(persist_directory=persist_directory, embedding_function=embedding)


def delete_chunks_by_filename(file_name):
    vectordb = get_vector_db()
    vectordb._collection.delete(where={"source": {"$eq": file_name}})
