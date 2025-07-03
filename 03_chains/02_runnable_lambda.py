from langchain_core.runnables import RunnableLambda


# A custom function
def format_name(data: dict) -> str:
    return f"My name is {data['name']}."


# Wrap it as a RunnableLambda
format_step = RunnableLambda(format_name)

# Call it with input
result = format_step.invoke({"name": "Ravi"})
print(result)  # Output: "My name is Ravi."
