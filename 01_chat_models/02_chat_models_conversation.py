from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001", google_api_key=api_key
    )

messages = [
    SystemMessage("You are an expert in social media content strategy"),
    HumanMessage("Give a short tip to create engaging posts on Instagram"),
]

result = llm.invoke(messages)

print(result.content)
