# tools/search.py

import os
from langchain_core.tools import BaseTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun

def get_search_tool() -> BaseTool:
    """
    Returns a unified search tool instance.
    
    Prioritizes using the Tavily AI search engine if the TAVILY_API_KEY is available,
    as it provides higher quality results for AI agents. Falls back to DuckDuckGoSearchRun
    if the key is not found.
    
    Returns:
        BaseTool: An instance of the search tool.
    """
    if os.getenv("TAVILY_API_KEY"):
        print("🔧 Tavily API key found. Using TavilySearchResults.")
        # max_results can be adjusted to get more or fewer source snippets.
        # Tavily's tool is more advanced and returns structured documents.
        # For a simple string output similar to DuckDuckGoSearchRun, we can wrap it.
        # However, it's often better to use the structured output later.
        # For now, let's use its default behavior which is excellent for RAG.
        return TavilySearchResults(max_results=3)
    else:
        print("⚠️ Tavily API key not found. Falling back to DuckDuckGoSearchRun.")
        return DuckDuckGoSearchRun()