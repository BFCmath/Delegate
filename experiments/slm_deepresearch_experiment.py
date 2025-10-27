# experiments/slm_deepresearch_experiment.py
"""
SLM Deep Research Experiment: Qwen 2.5 Math 1.5B with ReAct + Search
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


async def run_slm_deepresearch_experiment(
    test_df: pd.DataFrame,
    output_file: str,
    max_iterations: int = 10,
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
):
    """
    Run SLM deep research experiment with local Qwen 1.5B base model + ReAct.
    
    Args:
        test_df: DataFrame with columns: id, prompt
        output_file: Path to save results (JSONL format)
        max_iterations: Max search iterations per query
        model_id: HuggingFace model ID
        
    Returns:
        Summary dict with metrics
    """
    print(f"Running SLM ({model_id}) on {len(test_df)} queries")
    print(f"Max iterations: {max_iterations}")
    
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
        max_new_tokens=4096  # Longer for research
    )
    
    # Get search tool
    search_tool = get_search_tool()
    
    # Prepare output file
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
            
            print(f"✅ Completed in {latency:.2f}s | Searches: {result.metadata['search_count']} | Iterations: {result.metadata['iterations']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
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
    print(f"SLM Deep Research Results:")
    print(f"  Queries: {n_total}")
    print(f"  Completed: {completed_count} ({summary['completion_rate']:.1%})")
    print(f"  Avg Time: {summary['avg_time_per_query']:.2f}s")
    print(f"  Avg Searches: {summary['avg_search_per_query']:.1f}")
    print(f"  Avg Iterations: {summary['avg_iterations']:.1f}")
    print(f"{'='*60}")
    
    return summary

