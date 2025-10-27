# exp/test_4_search_tool.py

import os
import sys

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

from tools.search import get_search_tool

def run_search_test():
    """
    Isolates and tests the search tool to inspect its output format and content.
    """
    print("--- 🧪 Testing the Search Tool (DuckDuckGoSearchRun) ---")

    # 1. Get an instance of the search tool
    search_tool = get_search_tool()
    print(f"Tool Initialized: {type(search_tool)}\n")

    # 2. Define some sample queries to test
    queries_to_test = [
        "What is LangChain Expression Language (LCEL)?",
        "Latest advancements in generative AI in 2025"
    ]

    # 3. Execute each query and inspect the result
    for i, query in enumerate(queries_to_test, 1):
        print(f">>> Executing Query {i}: \"{query}\"")
        print("-----------------------------------------------------------------")

        try:
            # Invoke the tool with a single query string
            result = search_tool.invoke(query)

            # Print metadata about the result
            print(f"✅ Result received successfully.")
            print(f"   - Type of result: {type(result)}")
            print(f"   - Length of result: {len(result)} characters")

            # Print the actual content of the result
            print("\n--- Search Result Content ---")
            print(result)
            print("---------------------------\n")

        except Exception as e:
            print(f"❌ An error occurred while running the search: {e}\n")


if __name__ == "__main__":
    run_search_test()