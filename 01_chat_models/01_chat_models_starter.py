from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load variables from .env file into the environment
load_dotenv()

# Access variables using os.getenv
api_key = os.getenv('API_KEY')

print("API Key:", api_key)

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", google_api_key=api_key)


result = llm.invoke("What is the current time in India?")

print(result)

