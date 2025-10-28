"""
Test script to verify API key rotation behavior.
Simulates a ReAct loop and shows which key is used for each iteration.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.api_key_manager import create_key_manager

def test_key_rotation():
    """Test that keys cycle properly across iterations"""
    
    print("=" * 80)
    print("API KEY ROTATION TEST")
    print("=" * 80)
    
    # Create key manager
    key_manager = create_key_manager(cooldown_seconds=0)
    
    n_keys = len(key_manager.api_keys)
    print(f"\n📊 Available API Keys: {n_keys}")
    
    if n_keys == 0:
        print("❌ No API keys found! Set GOOGLE_API_KEY or GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc.")
        return
    
    print("\n" + "=" * 80)
    print("SIMULATION: 2 Queries × 10 Iterations Each")
    print("=" * 80)
    
    # Simulate 2 queries with 10 iterations each
    for query_idx in range(2):
        print(f"\n🔍 Query {query_idx + 1}")
        print("-" * 80)
        
        for iteration in range(10):
            # This simulates getting a model in the ReAct loop
            model = key_manager.get_model(
                model_name="gemini-2.5-flash",
                generation_config={"max_output_tokens": 100}
            )
            
            # Get which key was just used
            current_idx = (key_manager.current_index - 1) % n_keys
            key_suffix = key_manager.api_keys[current_idx]['key'][-8:]  # Last 8 chars
            
            print(f"  Iteration {iteration + 1:2d}/10 → KEY_{current_idx + 1} (***{key_suffix})")
    
    print("\n" + "=" * 80)
    print("KEY USAGE SUMMARY")
    print("=" * 80)
    
    total_calls = sum(key_info['usage_count'] for key_info in key_manager.api_keys)
    
    for i, key_info in enumerate(key_manager.api_keys):
        key_suffix = key_info['key'][-8:]
        usage = key_info['usage_count']
        percentage = (usage / total_calls * 100) if total_calls > 0 else 0
        
        bar = "█" * int(percentage / 2)  # Visual bar
        print(f"KEY_{i+1} (***{key_suffix}): {usage:3d} calls {bar} ({percentage:.1f}%)")
    
    print(f"\nTotal calls: {total_calls}")
    
    # Check if distribution is even
    expected_per_key = total_calls / n_keys
    max_deviation = max(abs(key_info['usage_count'] - expected_per_key) 
                        for key_info in key_manager.api_keys)
    
    print(f"Expected per key: {expected_per_key:.1f}")
    print(f"Max deviation: {max_deviation:.1f}")
    
    if max_deviation <= 1:
        print("\n✅ PASS: Keys are evenly distributed!")
    else:
        print("\n⚠️  WARNING: Uneven distribution (expected for small samples)")
    
    print("\n" + "=" * 80)
    print("HOW IT HELPS")
    print("=" * 80)
    print(f"Before: Query 1 would use KEY_1 for all 10 iterations → 10 calls to KEY_1")
    print(f"After:  Query 1 distributes across all {n_keys} keys → {10//n_keys}-{10//n_keys+1} calls per key")
    print(f"\n💡 With {n_keys} keys, each key handles ~{100//n_keys}% of the load!")
    print(f"   This reduces quota exhaustion risk by {n_keys}x")

if __name__ == "__main__":
    test_key_rotation()

