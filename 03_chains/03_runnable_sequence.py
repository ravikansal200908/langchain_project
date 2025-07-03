from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
import os


load_dotenv()
api_key = os.getenv('API_KEY')

# 1. Create a prompt
prompt = ChatPromptTemplate.from_template("Translate to French: {text}")

# 2. LLM
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001", google_api_key=api_key
    )


# 3. Postprocessing function
def postprocess(message):
    return message.content.upper()


postprocess_runnable = RunnableLambda(postprocess)

# 4. Combine them into a sequence
chain = RunnableSequence(prompt | model | postprocess_runnable)

# Run it
output = chain.invoke({"text": "Good morning"})
print(output)  # Something like: "BONJOUR"
