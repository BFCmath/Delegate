# tools/deepresearch_loader.py
"""
Load and prepare Deep Research queries for evaluation
"""
import json
import pandas as pd
from pathlib import Path

def load_deepresearch_queries_as_df(
    query_file: str = None,
    language: str = 'en',
    n_samples: int = None,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Load Deep Research queries from JSONL file and return as pandas DataFrame.
    
    Args:
        query_file: Path to JSONL query file. If None, uses default benchmark file.
        language: Filter by language ('en' or 'zh'). Use None for all languages.
        n_samples: Number of samples to load (None = all matching queries)
        random_seed: Seed for random sampling
        
    Returns:
        DataFrame with columns: id, prompt, topic, language
    """
    # Default to benchmark query file
    if query_file is None:
        query_file = "deepsearch_task/deep_research_bench/data/prompt_data/query.jsonl"
    
    query_path = Path(query_file)
    
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_file}")
    
    # Load JSONL file
    queries = []
    with open(query_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    
    # Convert to DataFrame
    df = pd.DataFrame(queries)
    
    # Ensure required columns exist
    required_cols = ['id', 'prompt']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Query file missing required columns: {missing_cols}")
    
    # Add topic if not present (use first few words of prompt)
    if 'topic' not in df.columns:
        df['topic'] = df['prompt'].str[:50] + '...'
    
    # Add language if not present (default to 'en')
    if 'language' not in df.columns:
        df['language'] = 'en'
    
    # Filter by language if specified
    if language:
        df = df[df['language'] == language].reset_index(drop=True)
    
    # Sample if requested
    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
    
    print(f"Loaded {len(df)} queries (language={language or 'all'})")
    
    return df


def prepare_sample_queries(
    output_file: str = 'research_queries.jsonl',
    n_samples: int = 10,
    language: str = 'en'
):
    """
    Extract sample queries from benchmark and save to a file for quick testing.
    
    Args:
        output_file: Output JSONL file path
        n_samples: Number of samples to extract
        language: Language filter
    """
    df = load_deepresearch_queries_as_df(language=language, n_samples=n_samples)
    
    # Save to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            record = {
                'id': int(row['id']),
                'topic': row.get('topic', ''),
                'language': row['language'],
                'prompt': row['prompt']
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(df)} sample queries to {output_file}")
    return output_file


# CLI usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare Deep Research queries')
    parser.add_argument('--output', default='research_queries.jsonl', help='Output JSONL path')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--language', default='en', choices=['en', 'zh'], help='Language filter')
    args = parser.parse_args()
    
    prepare_sample_queries(
        output_file=args.output,
        n_samples=args.samples,
        language=args.language
    )

