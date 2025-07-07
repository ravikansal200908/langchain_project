from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI

import os

# Load environment variables from .env
load_dotenv()
api_key = os.getenv('API_KEY')

# Create a ChatOpenAI model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001", google_api_key=api_key
    )


# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You love facts and you tell facts about {animal}"),
        ("human", "Tell me {count} facts."),
    ]
)

# Create individual runnables (steps in the chain)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(
    first=format_prompt, middle=[invoke_model], last=parse_output
    )

# Run the chain
response = chain.invoke({"animal": "cat", "count": 2})

# Output
print(response)
