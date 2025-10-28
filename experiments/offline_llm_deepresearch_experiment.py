# experiments/offline_llm_deepresearch_experiment.py
"""
Offline LLM Deep Research Experiment: Qwen3 4B GGUF with ReAct + Search
"""
import os
import time
import pandas as pd
import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.react_agent import ReActAgent
from tools.search_tool import get_search_tool

# Import llama-cpp-python for GGUF models
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    print("❌ llama-cpp-python not installed. Install with: pip install llama-cpp-python")
    LLAMA_CPP_AVAILABLE = False


class GGUFModelWrapper:
    """
    Wrapper for GGUF models using llama-cpp-python.
    """

    def __init__(self, model_path: str, max_new_tokens: int = 16384, n_ctx: int = 32768):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is required for GGUF models. Install with: pip install llama-cpp-python")

        print(f"📥 Loading GGUF model: {model_path}")
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,  # Context window
            n_threads=-1,  # Use all available threads
            verbose=False
        )
        self.max_new_tokens = max_new_tokens
        print(f"✅ GGUF model loaded (context: {n_ctx}, max output: {max_new_tokens})")

    def generate(self, prompt: str) -> str:
        """Generate response from GGUF model"""
        try:
            response = self.model(
                prompt,
                max_tokens=self.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                stop=["<|im_end|>", "<|endoftext|>", "</s>"],  # Common stop tokens
                echo=False
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            print(f"❌ GGUF generation error: {e}")
            return f"Error: GGUF generation failed - {str(e)}"


async def run_offline_llm_deepresearch_experiment(
    test_df: pd.DataFrame,
    output_file: str,
    max_iterations: int = 10,
    model_id: str = "unsloth/Qwen3-4B-Instruct-2507-GGUF",
    debug_file: str = None,
    fail_fast: bool = True
):
    """
    Run offline LLM deep research experiment with GGUF Qwen model + ReAct.

    Args:
        test_df: DataFrame with columns: id, prompt
        output_file: Path to save results (JSONL format)
        max_iterations: Max search iterations per query
        model_id: GGUF model path or HuggingFace model ID
        debug_file: Optional path to save debug info (full ReAct pipeline)
        fail_fast: If True, stop immediately on first error. If False, continue with remaining queries.

    Returns:
        Summary dict with metrics
    """
    print(f"Running Offline LLM ({model_id}) on {len(test_df)} queries")
    print(f"Max iterations: {max_iterations}")

    if not LLAMA_CPP_AVAILABLE:
        raise ImportError("llama-cpp-python is required. Install with: pip install llama-cpp-python")

    # Handle GGUF model path
    if model_id.endswith('.gguf'):
        # Direct path to GGUF file
        model_path = model_id
    else:
        # Assume it's a HuggingFace model ID - download and find GGUF
        from huggingface_hub import hf_hub_download
        import os

        print(f"📥 Downloading GGUF model: {model_id}...")

        # Try to find the GGUF file
        try:
            # First try to find any .gguf file in the repo
            model_path = hf_hub_download(
                repo_id=model_id,
                filename="*.gguf",
                cache_dir=os.path.expanduser("~/.cache/huggingface")
            )
        except Exception:
            # If no specific GGUF found, try the default name
            try:
                model_path = hf_hub_download(
                    repo_id=model_id,
                    filename="model.gguf",
                    cache_dir=os.path.expanduser("~/.cache/huggingface")
                )
            except Exception as e:
                print(f"❌ Could not find GGUF file for {model_id}")
                print("Make sure the model repo contains a .gguf file")
                raise e

    print(f"📁 Using GGUF file: {model_path}")

    # Load GGUF model
    model_wrapper = GGUFModelWrapper(
        model_path=model_path,
        max_new_tokens=32768,  # Much longer for deep research articles
        n_ctx=32768  # Context window
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

