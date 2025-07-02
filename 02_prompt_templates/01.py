from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import os


load_dotenv()
api_key = os.getenv('API_KEY')

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001", google_api_key=api_key
    )

# template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it to 4 lines max"

# prompt_template = ChatPromptTemplate.from_template(template)

# prompt =  prompt_template.invoke({
#     "tone": "energetic", 
#     "company": "samsung", 
#     "position": "AI Engineer", 
#     "skill": "AI"
# })

# Example 2: Prompt with System and Human Messages (Using Tuples)
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
result = model.invoke(prompt)
print(result)
