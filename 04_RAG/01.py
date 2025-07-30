# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings


from langchain.vectorstores import Chroma


HUGGINGFACE_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


# Load your document
loader = TextLoader("data/python.txt")
docs = loader.load()

# print("docs : ", docs)

# Split into chunks
# default is 1000, 200
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60)
chunks = splitter.split_documents(docs)

print("chunks: ", chunks[0])
print("=================")
print("chunks: ", chunks[1])
print("chunks: ", len(chunks))

# Convert to embeddings

# model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=HUGGINGFACE_EMBEDDING_MODEL,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


embedding_model = HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL)

# Store in vector DB
vector_store = Chroma.from_documents(
    chunks,
    embedding_model,
    persist_directory="./db"
    )
vector_store.persist()
