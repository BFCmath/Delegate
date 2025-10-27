# exp/test_6_react_loop.py

import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# Import the chain builder and the search tool
from approaches.react.chain import get_chain
from tools.search import get_search_tool

def run_react_loop_test():
    """
    Executes the full ReAct graph for a single topic and streams the output,
    showing the agent's state and actions at each step of the loop.
    """
    print("--- 🧪 Testing Full ReAct Loop with Streaming ---")
    load_dotenv()

    # 1. Initialize the model and tool
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
    search_tool = get_search_tool()

    # 2. Get the compiled LangGraph runnable
    react_graph = get_chain(llm, search_tool)

    # 3. Define the topic to test
    topic = "From 2020 to 2050, how many elderly people will there be in Japan? What is their consumption potential across various aspects such as clothing, food, housing, and transportation? Based on population projections, elderly consumer willingness, and potential changes in their consumption habits, please produce a market size analysis report for the elderly demographic."
    
    inputs = {"topic": topic}
    print(f"\n🚀 Starting Agent for Topic: \"{topic}\"")
    print("==================================================")

    final_state = {}
    # 4. Use .stream() to iterate through each step of the graph's execution
    for step in react_graph.stream(inputs):
        # The key of the dictionary is the name of the node that just ran
        node_name = list(step.keys())[0]
        # The value is the updated state
        state = list(step.values())[0]

        print(f"\n\n--- ✅ Executing Node: {node_name.upper()} ---")

        if node_name == "agent":
            print("Scratchpad:")
            # The agent's output is the last message in the scratchpad
            agent_output = state['scratchpad'][-1].content
            print(agent_output)
        
        elif node_name == "tool":
            print("Scratchpad:")
            # The tool's output is the last message (an observation)
            tool_output = state['scratchpad'][-1].content
            print(tool_output)
            
        elif node_name == "generate_final_output":
            # This is the final step before the end
            print("Final article has been generated.")
        
        # The final state is in the special '__end__' key
        if "__end__" in step:
            final_state = step["__end__"]

    print("\n\n==================================================")
    print("--- 🎉 Loop Finished ---")
    
    if final_state.get("article"):
        print("\n--- 📄 Final Generated Article ---")
        print(final_state["article"])
        
        print("\n--- 📊 Final Metadata ---")
        print(f"Search Count: {final_state['metadata']['search_count']}")
    else:
        print("--- ⚠️ No final article was generated. ---")


if __name__ == "__main__":
    run_react_loop_test()