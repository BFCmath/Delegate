# experiments/llm_deepresearch_experiment.py
"""
LLM Deep Research Experiment: Gemini 2.5 Flash with ReAct + Search
"""
import os
import time
import pandas as pd
import asyncio
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.react_agent import ReActAgent
from tools.search_tool import get_search_tool
from tools.api_key_manager import create_key_manager

load_dotenv()


async def run_llm_deepresearch_experiment(
    test_df: pd.DataFrame,
    output_file: str,
    max_iterations: int = 10
):
    """
    Run LLM-only deep research experiment with Gemini + ReAct.
    
    Args:
        test_df: DataFrame with columns: id, prompt
        output_file: Path to save results (JSONL format)
        max_iterations: Max search iterations per query
        
    Returns:
        Summary dict with metrics
    """
    print(f"Running Gemini 2.5 Flash on {len(test_df)} queries (max_iterations={max_iterations})")
    
    # Initialize components
    key_manager = create_key_manager(cooldown_seconds=1)
    search_tool = get_search_tool()
    
    # Prepare output file (overwrite if exists)
    with open(output_file, 'w') as f:
        pass  # Clear file
    
    # Track metrics
    total_time = 0.0
    total_search_count = 0
    total_iterations = 0
    completed_count = 0
    
    for idx, row in test_df.iterrows():
        print(f"\n[{idx+1}/{len(test_df)}] Query ID: {row['id']}")
        print(f"Prompt: {row['prompt'][:80]}...")
        
        try:
            # Get model with API key rotation
            model = key_manager.get_model(
                model_name="gemini-2.5-flash",
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=4096,  # Longer for research
                    temperature=0.7  # Slightly creative for research
                )
            )
            
            # Create agent
            agent = ReActAgent(
                model=model,
                search_tool=search_tool,
                max_iterations=max_iterations,
                is_local_model=False
            )
            
            # Run agent
            t_start = time.time()
            result = await agent.run(row['prompt'])
            t_end = time.time()
            
            latency = t_end - t_start
            total_time += latency
            total_search_count += result.metadata['search_count']
            total_iterations += result.metadata['iterations']
            
            if result.metadata['completed']:
                completed_count += 1
            
            # Prepare output record
            output_record = {
                "id": int(row["id"]),
                "prompt": row["prompt"],
                "article": result.article
            }
            
            # Append to JSONL file immediately
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
            
            print(f"✅ Completed in {latency:.2f}s | Searches: {result.metadata['search_count']} | Iterations: {result.metadata['iterations']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            # Save error record
            error_record = {
                "id": int(row["id"]),
                "prompt": row["prompt"],
                "article": f"ERROR: {str(e)}"
            }
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_record, ensure_ascii=False) + '\n')
    
    # Calculate summary
    n_total = len(test_df)
    summary = {
        'total_queries': n_total,
        'completed': completed_count,
        'completion_rate': completed_count / n_total if n_total else 0,
        'total_time': total_time,
        'avg_time_per_query': total_time / n_total if n_total else 0,
        'total_search_count': total_search_count,
        'avg_search_per_query': total_search_count / n_total if n_total else 0,
        'total_iterations': total_iterations,
        'avg_iterations': total_iterations / n_total if n_total else 0
    }
    
    print(f"\n{'='*60}")
    print(f"LLM Deep Research Results:")
    print(f"  Queries: {n_total}")
    print(f"  Completed: {completed_count} ({summary['completion_rate']:.1%})")
    print(f"  Avg Time: {summary['avg_time_per_query']:.2f}s")
    print(f"  Avg Searches: {summary['avg_search_per_query']:.1f}")
    print(f"  Avg Iterations: {summary['avg_iterations']:.1f}")
    print(f"{'='*60}")
    
    return summary

