# tools/search_tool.py
"""
Unified search tool wrapper supporting Tavily and DuckDuckGo
"""
import os
import time
from typing import List, Dict, Callable

def get_search_tool() -> Callable[[str], List[Dict[str, str]]]:
    """
    Returns a unified search tool instance.
    
    Prioritizes Tavily AI if TAVILY_API_KEY is available (higher quality),
    falls back to DuckDuckGo if not (free, no API key needed).
    
    Returns:
        Callable that takes a query string and returns list of search results.
        Each result is a dict with keys: title, url, content
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if tavily_key:
        print("🔍 Using Tavily search (API key found)")
        return _get_tavily_search(tavily_key)
    else:
        print("🔍 Using DuckDuckGo search (no Tavily API key)")
        return _get_duckduckgo_search()


def _get_tavily_search(api_key: str) -> Callable:
    """Create Tavily search function"""
    try:
        from tavily import TavilyClient
    except ImportError:
        raise ImportError(
            "Tavily search requires 'tavily-python' package. "
            "Install with: pip install tavily-python"
        )
    
    client = TavilyClient(api_key=api_key)
    
    def search(query: str, max_results: int = 3, max_retries: int = 3) -> List[Dict[str, str]]:
        """Search using Tavily API"""
        for attempt in range(max_retries):
            try:
                response = client.search(query, max_results=max_results)
                
                # Tavily returns structured results
                results = []
                for item in response.get('results', []):
                    results.append({
                        'title': item.get('title', 'No title'),
                        'url': item.get('url', ''),
                        'content': item.get('content', '')
                    })
                
                return results
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  Tavily search error (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(1)
                else:
                    print(f"❌ Tavily search failed after {max_retries} attempts: {e}")
                    return [{
                        'title': 'Search Error',
                        'url': '',
                        'content': f'Search failed: {str(e)}'
                    }]
    
    return search


def _get_duckduckgo_search() -> Callable:
    """Create DuckDuckGo search function"""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ImportError(
            "DuckDuckGo search requires 'duckduckgo-search' package. "
            "Install with: pip install duckduckgo-search"
        )
    
    def search(query: str, max_results: int = 3, max_retries: int = 3) -> List[Dict[str, str]]:
        """Search using DuckDuckGo"""
        for attempt in range(max_retries):
            try:
                with DDGS() as ddgs:
                    search_results = list(ddgs.text(query, max_results=max_results))
                
                # DuckDuckGo returns list of dicts with 'title', 'href', 'body'
                results = []
                for item in search_results:
                    results.append({
                        'title': item.get('title', 'No title'),
                        'url': item.get('href', ''),
                        'content': item.get('body', '')
                    })
                
                return results
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  DuckDuckGo search error (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2)  # Longer delay for rate limiting
                else:
                    print(f"❌ DuckDuckGo search failed after {max_retries} attempts: {e}")
                    return [{
                        'title': 'Search Error',
                        'url': '',
                        'content': f'Search failed: {str(e)}'
                    }]
    
    return search


def format_search_results(results: List[Dict[str, str]]) -> str:
    """
    Format search results into a readable string for LLM consumption.
    
    Args:
        results: List of search result dicts with title, url, content
        
    Returns:
        Formatted string with numbered results
    """
    if not results:
        return "No search results found."
    
    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(
            f"[{i}] {result['title']}\n"
            f"URL: {result['url']}\n"
            f"Content: {result['content']}\n"
        )
    
    return "\n".join(formatted)


# CLI testing
if __name__ == "__main__":
    import sys
    
    search = get_search_tool()
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What is artificial intelligence?"
    
    print(f"\n🔎 Testing search with query: '{query}'\n")
    results = search(query)
    
    print(format_search_results(results))
    print(f"\n✅ Found {len(results)} results")

