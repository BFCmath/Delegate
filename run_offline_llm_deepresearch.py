#!/usr/bin/env python3
"""
Run Offline LLM Deep Research Experiment (Qwen 7B + ReAct + Search)
"""
import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.deepresearch_loader import load_deepresearch_queries_as_df


async def main():
    parser = argparse.ArgumentParser(description='Run offline LLM deep research experiment')
    parser.add_argument('--queries', type=int, default=10, help='Number of queries (default: 10)')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--max-iterations', type=int, default=10, help='Max search iterations per query')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-4B-Instruct-2507',
                        help='HuggingFace model ID')
    parser.add_argument('--input-file', type=str, help='Custom query JSONL file (optional)')
    parser.add_argument('--output', type=str, help='Output directory (default: auto-generated)')
    parser.add_argument('--language', type=str, default='en', choices=['en', 'zh'], help='Query language')
    parser.add_argument('--debug', action='store_true', help='Save full ReAct pipeline (thoughts/actions/observations) for debugging')
    parser.add_argument('--no-fail-fast', action='store_true', help='Continue with remaining queries on error (default: stop immediately)')
    args = parser.parse_args()
    
    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_short = args.model.split('/')[-1].replace('-', '_').lower()
        output_dir = Path(f"results_offline_llm_deepresearch_{model_short}_{args.queries}queries_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load queries
    print(f"\n📂 Loading {args.queries} queries...")
    test_df = load_deepresearch_queries_as_df(
        query_file=args.input_file,
        language=args.language,
        n_samples=args.queries,
        random_seed=args.seed
    )
    
    # Save query list
    query_file = output_dir / "queries.csv"
    test_df.to_csv(query_file, index=False)
    print(f"✅ Saved query list to {query_file}")
    
    # Import and run experiment
    from experiments.offline_llm_deepresearch_experiment import run_offline_llm_deepresearch_experiment
    
    print("\n" + "="*80)
    print(f"EXPERIMENT: OFFLINE LLM DEEP RESEARCH ({args.model} + vLLM + ReAct + Search)")
    print("="*80)
    print("⚠️  Note: This will download the model (~8GB) if not cached")
    print("🚀 Using vLLM for optimized inference")
    print("⚠️  GPU recommended for acceptable performance")
    
    results_file = output_dir / "results.jsonl"
    debug_file = output_dir / "debug_pipeline.jsonl" if args.debug else None
    fail_fast = not args.no_fail_fast  # Default True, unless --no-fail-fast is set
    
    summary = await run_offline_llm_deepresearch_experiment(
        test_df.copy(),
        str(results_file),
        max_iterations=args.max_iterations,
        model_id=args.model,
        debug_file=str(debug_file) if debug_file else None,
        fail_fast=fail_fast
    )
    
    # Save summary
    import json
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\n📊 Results saved to: {output_dir}")
    print(f"   - results.jsonl (articles in benchmark format)")
    print(f"   - summary.json (metrics)")
    print(f"   - queries.csv (query list)")
    if args.debug:
        print(f"   - debug_pipeline.jsonl (full ReAct traces for debugging)")
    
    print(f"\n📈 Summary:")
    print(f"   Queries: {summary['total_queries']}")
    print(f"   Completed: {summary['completed']} ({summary['completion_rate']:.1%})")
    print(f"   Avg Time: {summary['avg_time_per_query']:.2f}s")
    print(f"   Avg Searches: {summary['avg_search_per_query']:.1f}")
    print(f"   Avg Iterations: {summary['avg_iterations']:.1f}")
    
    print(f"\n💡 To run RACE benchmark evaluation:")
    print(f"   1. Copy results.jsonl to: deepsearch_task/deep_research_bench/data/test_data/raw_data/<model_name>.jsonl")
    print(f"   2. Run: python deepsearch_task/deep_research_bench/deepresearch_bench_race.py")


if __name__ == "__main__":
    asyncio.run(main())

