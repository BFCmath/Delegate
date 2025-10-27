# exp/test_2_parser_with_retries.py

import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# 1. Define the desired data structure with Pydantic
class QueryList(BaseModel):
    queries: list[str] = Field(
        description="A list of exactly 5 diverse search queries related to the user's topic."
    )

def run_parser_test():
    """Tests the PydanticOutputParser with built-in retry logic."""
    print("--- 🧪 Testing Method 2: Output Parser with Retries ---")
    load_dotenv()

    # 2. Initialize the model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

    # 3. Set up the parser
    parser = PydanticOutputParser(pydantic_object=QueryList)

    # 4. Create a prompt that includes the auto-generated format instructions
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert research assistant. Generate a list of search queries based on the user's topic.\n"
                "{format_instructions}",
            ),
            ("user", "Research Topic: {topic}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    # 5. Create the chain and add retry logic
    # The .with_retry() method automatically catches parsing errors and asks the LLM to fix them.
    chain = (prompt | llm | parser).with_retry(
        # You can customize the retry behavior, e.g., number of attempts
        stop_after_attempt=3
    )

    topic = "The impact of the printing press on the European Renaissance."
    print(f"Input Topic: '{topic}'\n")

    # 6. Invoke the chain
    result_object = chain.invoke({"topic": topic})

    # 7. Print the result
    print("--- ✅ Pydantic Object Result ---")
    print(result_object)
    
    print("\n--- ✅ Accessing the List ---")
    print(result_object.queries)
    print(f"Number of queries generated: {len(result_object.queries)}")


if __name__ == "__main__":
    run_parser_test()