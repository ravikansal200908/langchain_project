from langchain.prompts import ChatPromptTemplate

# Define the prompt
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You love facts and you tell facts about {animal}"),
    ("human", "Tell me {count} facts."),
])

# Simulated input
input_data = {"animal": "cat", "count": 2}


# Simple function (same as the lambda)
def format_prompt_step(x):
    return prompt_template.format_prompt(**x)


# Call it
result = format_prompt_step(input_data)
print(result.to_messages())
