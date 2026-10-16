# Tools that the LLM can use to interact with the world.
# Wikipedia, DuckDuckGo, and Custom Tool (python function)

from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from datetime import datetime
from ddgs import DDGS

@tool
def search_tool(query: str) -> str:
    """Search the web using DuckDuckGo.
    
    Returns search results or a readable error if search fails.
    """
    try:
        with DDGS(timeout=10) as ddgs:
            results = list(
                ddgs.text(query, max_results=5)
            )

        if not results:
            return (
                f"No DuckDuckGo results found for '{query}'. "
                "Try a different query or use Wikipedia."
            )

        formatted_results = []

        for result in results:
            formatted_results.append(
                f"Title: {result.get('title', 'N/A')}\n"
                f"URL: {result.get('href', 'N/A')}\n"
                f"Snippet: {result.get('body', 'N/A')}"
            )

        return "\n\n".join(formatted_results)

    except Exception as e:
        return (
            f"DuckDuckGo search failed for '{query}'. "
            f"Error: {type(e).__name__}: {e}. "
            "Try Wikipedia or another available source."
        )
    
@tool
def save_to_txt(content: str, sources: list[str], filename: str = "research_output.txt") -> str:
    """Save research content to a text file.

    Args:
        content: The research output to save.
        sources: The list of sources used.
        filename: The name of the output file.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_output_{timestamp}.txt"

    formatted_text = f"Research Output - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{content}\n\n{sources}"

    filename = "output/"+filename
    with open(filename, "w") as file:
        file.write(formatted_text)
    
    return f"Content saved to {filename}"

class SafeWikipediaAPIWrapper(WikipediaAPIWrapper):
    def run(self, query: str) -> str:
        try:
            return super().run(query)
        except Exception as e:
            return (
                f"Wikipedia search failed for '{query}'. "
                f"Error: {type(e).__name__}: {e}. "
                "Try DuckDuckGo search instead."
            )

# search_tool = DuckDuckGoSearchRun()

api_wrapper = SafeWikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# custom tool to save research output to a text file