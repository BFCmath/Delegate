# exp/test_1_few_shot_prompting.py

import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

def run_few_shot_test():
    """Tests the effectiveness of few-shot prompting for format control."""
    print("--- 🧪 Testing Method 1: Few-Shot Prompting ---")
    load_dotenv()

    # 1. Initialize the model with the specified configuration
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

    # 2. Define a detailed prompt with explicit rules and an example
    few_shot_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert research assistant. Your goal is to generate exactly 5 diverse search queries to answer the user's topic.

YOU MUST FOLLOW THESE RULES:
1.  Generate EXACTLY 5 search queries. Not 4, not 6. Five.
2.  Each query must be on a new line.
3.  Do not number the queries or use any other formatting like dashes.

Here is an example of the desired output format:
---
Topic: The history of the Roman Empire
Output:
Key turning points in the Roman Republic's fall
Major engineering achievements of the Roman Empire
Daily life for a common citizen in ancient Rome
Roman military tactics and strategies
Lasting impact of Roman law on modern legal systems
---""",
            ),
            ("user", "Research Topic: {topic}"),
        ]
    )

    # 3. Create and invoke the chain
    chain = few_shot_prompt | llm | StrOutputParser()
    topic = "The impact of the printing press on the European Renaissance."
    print(f"Input Topic: '{topic}'\n")
    
    result_string = chain.invoke({"topic": topic})

    # 4. Print the result
    print("--- ✅ Raw String Result ---")
    print(result_string)
    
    print("\n--- ✅ Result as a Python List ---")
    query_list = [line for line in result_string.strip().split('\n') if line]
    print(query_list)
    print(f"Number of queries generated: {len(query_list)}")


if __name__ == "__main__":
    run_few_shot_test()