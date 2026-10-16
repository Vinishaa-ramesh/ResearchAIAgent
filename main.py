from dotenv import load_dotenv
from pydantic import BaseModel
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

load_dotenv()  # Load environment variables from .env file

model="openai/gpt-oss-120b"

# LLM Configuration
llm = ChatGroq(
    model=model,
    temperature=0,
    max_tokens=200
)

response = llm.invoke("Why do dreams occur?")  # Invoke the LLM with a prompt
print(response.content)  # Print the response from the LLM