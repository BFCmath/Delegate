# experiments/offline_llm_deepresearch_experiment.py
"""
Offline LLM Deep Research Experiment: Qwen 2.5 Math 7B with ReAct + Search
"""
import os
import time
import pandas as pd
import asyncio
import json
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.react_agent import ReActAgent, LocalModelWrapper
from tools.search_tool import get_search_tool


def _device_dtype():
    """Determine best device and dtype"""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


async def run_offline_llm_deepresearch_experiment(
    test_df: pd.DataFrame,
    output_file: str,
    max_iterations: int = 10,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    debug_file: str = None,
    fail_fast: bool = True
):
    """
    Run offline LLM deep research experiment with local Qwen base model + ReAct.
    
    Args:
        test_df: DataFrame with columns: id, prompt
        output_file: Path to save results (JSONL format)
        max_iterations: Max search iterations per query
        model_id: HuggingFace model ID
        debug_file: Optional path to save debug info (full ReAct pipeline)
        fail_fast: If True, stop immediately on first error. If False, continue with remaining queries.
        
    Returns:
        Summary dict with metrics
    """
    print(f"Running Offline LLM ({model_id}) on {len(test_df)} queries")
    print(f"Max iterations: {max_iterations}")
    print("⚠️  This will download the model (~14GB) if not cached")
    
    # Load model
    print(f"\n📥 Loading model: {model_id}...")
    device, dtype = _device_dtype()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if device != "cpu" else None,
        torch_dtype=dtype,
        trust_remote_code=True
    )
    
    print(f"✅ Model loaded on {device}")
    
    # Wrap model
    model_wrapper = LocalModelWrapper(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=32768  # Much longer for deep research articles
    )
    
    # Get search tool
    search_tool = get_search_tool()
    
    # Prepare output file
    with open(output_file, 'w') as f:
        pass  # Clear file
    
    # Prepare debug file if requested
    if debug_file:
        with open(debug_file, 'w') as f:
            pass  # Clear file
        print(f"🐛 Debug mode enabled - saving full ReAct traces to {debug_file}")
    
    # Track metrics
    total_time = 0.0
    total_search_count = 0
    total_iterations = 0
    completed_count = 0
    
    for idx, row in test_df.iterrows():
        print(f"\n[{idx+1}/{len(test_df)}] Query ID: {row['id']}")
        print(f"Prompt: {row['prompt'][:80]}...")
        
        try:
            # Create agent
            agent = ReActAgent(
                model=model_wrapper,
                search_tool=search_tool,
                max_iterations=max_iterations,
                is_local_model=True
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
            
            # Append to JSONL file
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
            
            # Save debug info if requested
            if debug_file:
                debug_record = {
                    "id": int(row["id"]),
                    "prompt": row["prompt"],
                    "metadata": result.metadata,
                    "react_pipeline": result.scratchpad,
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
                print("⚠️  Fail-fast mode: Stopping experiment immediately.")
                print(f"⚠️  {completed_count} queries completed successfully before failure.")
                raise RuntimeError(
                    f"Experiment stopped on query {idx+1}/{len(test_df)} (ID: {row['id']}). "
                    f"Error: {error_msg}"
                ) from e
            else:
                print("⚠️  Continuing with next query (fail_fast=False)...")
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
    print(f"Offline LLM Deep Research Results:")
    print(f"  Queries: {n_total}")
    print(f"  Completed: {completed_count} ({summary['completion_rate']:.1%})")
    print(f"  Avg Time: {summary['avg_time_per_query']:.2f}s")
    print(f"  Avg Searches: {summary['avg_search_per_query']:.1f}")
    print(f"  Avg Iterations: {summary['avg_iterations']:.1f}")
    print(f"{'='*60}")
    
    return summary

