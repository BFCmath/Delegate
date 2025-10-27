# exp/test_5_react_prompt.py

import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# Import the prompt we want to test
from approaches.react.prompts import REACT_PROMPT

def run_react_prompt_test():
    """
    Isolates and tests the ReAct prompt to see the model's raw
    "Thought/Action/Action Input" output for the first step.
    """
    print("--- 🧪 Testing ReAct Prompt (First Step) ---")
    load_dotenv()

    # 1. Initialize the model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

    # 2. Define the inputs for the prompt
    # Change this question to experiment with different topics.
    question_to_test = "What were the key contributing factors to the rise and fall of the dot-com bubble in the late 1990s and early 2000s?"
    
    # For the first turn, the scratchpad is empty.
    initial_scratchpad = ""

    # 3. Create a simple chain for this test
    # Chain: Prompt -> LLM -> String Output
    test_chain = REACT_PROMPT | llm | StrOutputParser()

    print(f"\nInput Question: '{question_to_test}'")
    print("Simulating the first agent turn with an empty scratchpad...")
    
    # 4. Invoke the chain and get the result
    llm_output = test_chain.invoke({
        "question": question_to_test,
        "scratchpad": initial_scratchpad
    })

    # 5. Print the raw output for evaluation
    print("\n--- ✅ LLM Raw Output ---")
    print(llm_output)
    print("--------------------------")
    print("\nCheck if the output above follows the 'Thought/Action/Action Input' format.")


if __name__ == "__main__":
    run_react_prompt_test()