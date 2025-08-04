from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from embedding_manager import get_vector_db
import os
from dotenv import load_dotenv

load_dotenv()


def get_rag_chain():
    retriever = get_vector_db().as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001", google_api_key=os.getenv("GEMINI_API_KEY", "AIzaSyC2wr_qAETa_ZsCNrnyNZ5qyVWyGSbGWNo")
        )
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


def run_rag_with_fallback(query: str) -> tuple[str, str]:
    retriever = get_vector_db().as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    print("\n\n\n")
    print("docs : ", docs)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    rag_answer = None
    if docs:
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        result = chain.invoke(query)

        # result might be a string or a dict with `result` key
        if isinstance(result, dict):
            rag_answer = result.get("result", "").strip()
        else:
            rag_answer = result.strip()

    # Check if answer is meaningful
    if rag_answer and not is_generic_response(rag_answer):
        return rag_answer, "RAG"
    else:
        llm_answer = llm.invoke(query).content
        return llm_answer, "LLM"


def is_generic_response(answer: str) -> bool:
    """
    Returns True if the answer looks like a generic fallback.
    You can extend this as needed.
    """
    generic_phrases = [
        "I’m sorry", 
        "do not contain", 
        "I don't have information",
        "No relevant documents"
    ]
    return any(phrase.lower() in answer.lower() for phrase in generic_phrases)

