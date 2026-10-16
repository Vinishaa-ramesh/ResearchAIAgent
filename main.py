from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from tools import search_tool, wiki_tool, save_to_txt

load_dotenv()  # Load environment variables from .env file
model="openai/gpt-oss-120b"

# Define output fields for the research response
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# LLM Configuration
llm = ChatGroq(
    model=model,
    temperature=0,
    max_tokens=1000
)

# Define the prompt template for the research assistant
system_prompt = f"""
    You are a research assistant that will help with generating a research paper.
    Answer the query and use necessary tools.
    Wrap the final content in the mentioned format ResearchResponse.
    Save the content to a text file.
"""

tools_list = [search_tool, wiki_tool, save_to_txt]

# Create the agent with the LLM, prompt, and tools
agent = create_agent(
    model=llm,
    tools=tools_list,
    system_prompt=system_prompt,
    response_format=ToolStrategy(ResearchResponse),
)

# query_text = "How to create an AI agent?"
query_text = input("Enter your research query: ")

# Test the agent with a sample query
response = agent.invoke({
    "messages": [
        {"role": "user", "content": query_text},
    ]
})

# response = structured_llm.invoke(query_text)

print(response["structured_response"])