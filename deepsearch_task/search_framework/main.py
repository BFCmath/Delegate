# main.py

import os
import json
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from base_framework import run_pipeline

def main():
    """
    Runs a batch processing pipeline with performance and resource tracking.
    """
    load_dotenv()
    
    # --- Load all available API keys ---
    api_keys = []
    main_key = os.getenv("GOOGLE_API_KEY")
    if main_key:
        api_keys.append(main_key)
    
    # Add additional API keys
    for i in range(1, 6):
        key = os.getenv(f"GOOGLE_API{i}_KEY")
        if key:
            api_keys.append(key)
    
    if not api_keys:
        raise ValueError("No GOOGLE_API_KEY found in environment variables.")
    
    print(f"🔑 Loaded {len(api_keys)} API key(s) for rotation")

    # --- Configuration ---
    selected_approach = "minddeepsearch"
    input_file = "query.jsonl"
    output_dir = "outputs"
    
    # --- File Handling ---
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{selected_approach}.jsonl")
    resource_file = os.path.join(output_dir, f"{selected_approach}_resources.txt")

    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found.")
        return

    # --- Tracking Initialization ---
    processing_times = []
    total_searches = 0
    total_llm_calls = 0

    print(f"🚀 Starting batch processing from '{input_file}'...")
    print(f"Outputs will be saved to '{output_dir}'.")

    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'a', encoding='utf-8') as f_out, \
             open(resource_file, 'w', encoding='utf-8') as f_res:
            
            lines = f_in.readlines()
            
            for i, line in enumerate(lines):
                start_time = time.time()
                
                data = json.loads(line.strip())
                query_id = data["id"]
                research_prompt = data["prompt"]
                # Rotate API keys: use different key for each query
                current_api_key = api_keys[i % len(api_keys)]
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash", 
                    temperature=0.5,
                    google_api_key=current_api_key
                )
                
                print(f"\n[{i+1}/{len(lines)}] Processing ID: {query_id} | Using API Key #{(i % len(api_keys)) + 1}")
                print(f"Prompt: '{research_prompt[:80]}...'")

                result_data = run_pipeline(
                    topic=research_prompt,
                    model=llm,
                    approach_name=selected_approach
                )
                
                end_time = time.time()
                
                # --- Extract and store data ---
                final_report = result_data["article"]
                output_record = {"id": query_id, "prompt": research_prompt, "article": final_report}
                f_out.write(json.dumps(output_record) + '\n')

                metadata = result_data["metadata"]
                duration = end_time - start_time
                llm_calls = metadata["llm_calls"]
                search_count = metadata["search_count"]

                processing_times.append(duration)
                total_searches += search_count
                total_llm_calls += llm_calls
                
                # Write individual stats to the resource file
                f_res.write(f"ID: {query_id}\n")
                f_res.write(f"  - Total Time: {duration:.2f} seconds\n")
                f_res.write(f"  - LLM API Calls: {llm_calls}\n")
                f_res.write(f"  - Searches Conducted: {search_count}\n")
                f_res.write("---\n")
                
                print(f"✅ ID {query_id} finished in {duration:.2f}s. (LLM Calls: {llm_calls}, Searches: {search_count})")

            # --- Final Summary ---
            if processing_times:
                avg_time = sum(processing_times) / len(processing_times)
                summary = (
                    f"\n====================\n"
                    f"  BATCH SUMMARY\n"
                    f"====================\n"
                    f"Total Prompts Processed: {len(processing_times)}\n"
                    f"Average Time per Prompt: {avg_time:.2f} seconds\n"
                    f"Total LLM API Calls: {total_llm_calls}\n"
                    f"Total Searches Conducted: {total_searches}\n"
                )
                f_res.write(summary)
                print(summary)

    except Exception as e:
        print(f"❌ A critical error occurred: {e}")

    print("\n🎉 Batch processing complete.")

if __name__ == "__main__":
    main()