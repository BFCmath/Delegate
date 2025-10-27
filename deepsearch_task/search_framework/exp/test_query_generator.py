# exp/test_query_generator.py

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# To test a prompt, we import it directly from its source file.
# This ensures that any tweaks you make to the original prompt file
# are reflected immediately when you run this test script.
from approaches.zeroshot.prompts import QUERY_GENERATOR_PROMPT

def test_prompt():
    """
    A simple, self-contained function to test the query generator prompt.
    """
    # Load environment variables from the root .env file
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found. Make sure it's in a .env file in the root directory.")

    # 1. Define the topic you want to test
    # Change this variable to experiment with different topics.
    topic_to_test = "The history and evolution of Jazz music in New Orleans."

    # 2. Initialize the model
    # Use a specific model for consistent testing.
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

    # 3. Create a simple chain for this test
    # Chain: Prompt -> LLM -> String Output
    test_chain = QUERY_GENERATOR_PROMPT | llm | StrOutputParser()

    # 4. Invoke the chain and get the result
    print("--- 🧪 Testing Query Generator Prompt ---")
    print(f"Input Topic: '{topic_to_test}'\n")
    print("⏳ Invoking model to generate queries...")
    
    generated_queries = test_chain.invoke({"topic": topic_to_test})

    # 5. Print the output for evaluation
    print("\n--- ✅ Generated Queries ---")
    print(generated_queries)
    print("--------------------------")

if __name__ == "__main__":
    test_prompt()