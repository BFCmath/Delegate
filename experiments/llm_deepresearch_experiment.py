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
    max_iterations: int = 10,
    debug_file: str = None,
    fail_fast: bool = True
):
    """
    Run LLM-only deep research experiment with Gemini + ReAct.
    
    Args:
        test_df: DataFrame with columns: id, prompt
        output_file: Path to save results (JSONL format)
        max_iterations: Max search iterations per query
        debug_file: Optional path to save debug info (full ReAct pipeline)
        fail_fast: If True, stop immediately on first error. If False, continue with remaining queries.
        
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
    
    # Prepare debug file if requested
    if debug_file:
        with open(debug_file, 'w') as f:
            pass  # Clear file
        print(f"🐛 Debug mode enabled - saving full ReAct traces to {debug_file}")
    
    # Model configuration for key rotation (passed to agent)
    model_config = {
        "model_name": "gemini-2.5-flash",
        "generation_config": genai.types.GenerationConfig(
            max_output_tokens=4096,  # Longer for research
            temperature=0.7  # Slightly creative for research
        )
    }
    
    print(f"🔄 API Key Rotation: Enabled (cycling keys per LLM call)")
    print(f"📊 Available Keys: {len(key_manager.api_keys)}")
    
    # Track metrics
    total_time = 0.0
    total_search_count = 0
    total_iterations = 0
    completed_count = 0
    
    for idx, row in test_df.iterrows():
        print(f"\n[{idx+1}/{len(test_df)}] Query ID: {row['id']}")
        print(f"Prompt: {row['prompt'][:80]}...")
        
        try:
            # Create agent with key_manager (cycles keys per iteration)
            agent = ReActAgent(
                model=key_manager,  # Pass key_manager instead of single model
                search_tool=search_tool,
                max_iterations=max_iterations,
                is_local_model=False,
                model_config=model_config  # Config for getting fresh models
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
            
            # Save debug info if requested
            if debug_file:
                debug_record = {
                    "id": int(row["id"]),
                    "prompt": row["prompt"],
                    "metadata": result.metadata,
                    "react_pipeline": result.scratchpad,  # Full Thought/Action/Observation trace
                    "article": result.article[:200] + "..." if len(result.article) > 200 else result.article
                }
                with open(debug_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(debug_record, ensure_ascii=False, indent=2) + '\n')
            
            print(f"✅ Completed in {latency:.2f}s | Searches: {result.metadata['search_count']} | Iterations: {result.metadata['iterations']}")
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            print(f"\n{'='*80}")
            print(f"❌ EXPERIMENT FAILED ON QUERY {idx+1}/{len(test_df)}")
            print(f"{'='*80}")
            print(f"Query ID: {row['id']}")
            print(f"Error Type: {error_type}")
            print(f"Error Message: {error_msg}")
            print(f"{'='*80}\n")
            
            if fail_fast:
                # Stop immediately - don't write error to output
                print("⚠️  Fail-fast mode: Stopping experiment immediately.")
                print(f"⚠️  {completed_count} queries completed successfully before failure.")
                raise RuntimeError(
                    f"Experiment stopped on query {idx+1}/{len(test_df)} (ID: {row['id']}). "
                    f"Error: {error_msg}"
                ) from e
            else:
                # Continue with next query (legacy behavior)
                print("⚠️  Continuing with next query (fail_fast=False)...")
                
                # Save error record for debugging
                if debug_file:
                    debug_error_record = {
                        "id": int(row["id"]),
                        "prompt": row["prompt"],
                        "error": error_msg,
                        "error_type": error_type,
                        "react_pipeline": []
                    }
                    with open(debug_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(debug_error_record, ensure_ascii=False, indent=2) + '\n')
    
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
    
    # Print key usage statistics
    print(f"\n{'='*60}")
    print("📊 API KEY USAGE STATISTICS")
    print(f"{'='*60}")
    total_calls = sum(key_info['usage_count'] for key_info in key_manager.api_keys)
    for i, key_info in enumerate(key_manager.api_keys, 1):
        key_suffix = key_info['key'][-8:] if len(key_info['key']) >= 8 else key_info['key'][-4:]
        usage = key_info['usage_count']
        percentage = (usage / total_calls * 100) if total_calls > 0 else 0
        bar = "█" * min(int(percentage / 2), 50)  # Cap at 50 chars
        print(f"KEY_{i} (***{key_suffix}): {usage:3d} calls {bar} ({percentage:.1f}%)")
    print(f"\nTotal API calls: {total_calls}")
    expected_per_key = total_calls / len(key_manager.api_keys) if len(key_manager.api_keys) > 0 else 0
    print(f"Expected per key: {expected_per_key:.1f}")
    max_deviation = max(abs(key_info['usage_count'] - expected_per_key) 
                        for key_info in key_manager.api_keys) if key_manager.api_keys else 0
    print(f"Max deviation: {max_deviation:.1f}")
    if max_deviation <= 2:
        print("✅ Keys are evenly distributed!")
    else:
        print("⚠️  Uneven distribution - some keys may have failed")
    print(f"{'='*60}")
    
    return summary

