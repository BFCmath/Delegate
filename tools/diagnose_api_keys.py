#!/usr/bin/env python3
"""
API Key Diagnostic Tool - Test each key individually and identify quota issues
"""
import sys
import os
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

def test_single_key(key: str, key_num: int, total_keys: int):
    """Test a single API key with multiple requests"""
    print(f"\n{'='*80}")
    print(f"Testing API Key #{key_num}/{total_keys}")
    print(f"{'='*80}")
    
    key_suffix = key[-8:] if len(key) >= 8 else key[-4:]
    print(f"Key suffix: ***{key_suffix}")
    
    # Configure with this key
    genai.configure(api_key=key)
    
    # Try 3 quick requests
    success_count = 0
    project_numbers = set()
    
    for i in range(1, 4):
        try:
            print(f"\n  Request {i}/3...", end=" ")
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            response = model.generate_content(
                f"Say 'Test {i}' and nothing else.",
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=10,
                    temperature=0
                )
            )
            print(f"✅ Success: {response.text.strip()}")
            success_count += 1
            
            # Small delay between requests
            time.sleep(0.5)
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Failed")
            print(f"     Error: {error_msg[:200]}")
            
            # Extract project number from error if present
            if "project_number:" in error_msg:
                import re
                match = re.search(r'project_number:(\d+)', error_msg)
                if match:
                    project_num = match.group(1)
                    project_numbers.add(project_num)
                    print(f"     Project: {project_num}")
    
    print(f"\n  Result: {success_count}/3 requests succeeded")
    if project_numbers:
        print(f"  Project number(s) seen: {', '.join(project_numbers)}")
    
    return {
        'key_num': key_num,
        'key_suffix': key_suffix,
        'success_count': success_count,
        'project_numbers': project_numbers
    }


def main():
    print("="*80)
    print("🔍 API KEY DIAGNOSTIC TOOL")
    print("="*80)
    print()
    print("This tool tests each API key individually to:")
    print("  1. Verify each key works")
    print("  2. Identify which project each key belongs to")
    print("  3. Detect quota issues")
    print()
    
    # Load API keys
    from tools.api_key_manager import load_api_keys_from_env
    keys = load_api_keys_from_env()
    
    if not keys:
        print("❌ No API keys found!")
        print()
        print("Set environment variables:")
        print("  GOOGLE_API_KEY_1=your_first_key")
        print("  GOOGLE_API_KEY_2=your_second_key")
        print("  ...")
        return 1
    
    print(f"Found {len(keys)} API key(s)")
    print()
    input("Press Enter to start testing... ")
    
    # Test each key
    results = []
    for i, key in enumerate(keys, 1):
        result = test_single_key(key, i, len(keys))
        results.append(result)
        
        # Wait between keys to avoid rate limits
        if i < len(keys):
            print(f"\n⏳ Waiting 3 seconds before testing next key...")
            time.sleep(3)
    
    # Summary
    print("\n" + "="*80)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*80)
    
    all_projects = set()
    for result in results:
        all_projects.update(result['project_numbers'])
    
    print(f"\nTotal keys tested: {len(results)}")
    print(f"Unique project numbers seen: {len(all_projects)}")
    
    if len(all_projects) > 0:
        print(f"\nProjects: {', '.join(all_projects)}")
    
    print("\nPer-key results:")
    for result in results:
        status = "✅" if result['success_count'] == 3 else "⚠️" if result['success_count'] > 0 else "❌"
        print(f"  {status} Key #{result['key_num']} (***{result['key_suffix']}): {result['success_count']}/3 succeeded")
        if result['project_numbers']:
            for proj in result['project_numbers']:
                print(f"      → Project: {proj}")
    
    print()
    
    # Analysis
    if len(all_projects) == 1:
        print("⚠️  WARNING: All keys appear to be from the SAME project!")
        print("   This means they share quota limits.")
        print()
        print("💡 SOLUTION: Create keys from different Google Cloud projects")
        print("   Each project gets its own quota.")
    elif len(all_projects) > 1:
        print("✅ Good! Keys are from DIFFERENT projects")
        print(f"   You have {len(all_projects)} separate quota pools.")
        print()
        if any(r['success_count'] < 3 for r in results):
            print("⚠️  However, some keys are failing...")
            print("   Possible causes:")
            print("   1. Regional rate limits (affects all projects in same region)")
            print("   2. IP-based rate limiting")
            print("   3. Requests too fast (try increasing cooldown)")
            print()
            print("💡 SOLUTIONS:")
            print("   - Increase cooldown_seconds in create_key_manager()")
            print("   - Reduce --max-iterations")
            print("   - Add delays between queries")
    else:
        print("✅ All keys working! No errors detected.")
        print()
        print("If you're still seeing quota issues in experiments:")
        print("   1. You might be making requests too quickly")
        print("   2. Try increasing cooldown_seconds")
        print("   3. Reduce concurrent/parallel requests")
    
    print()
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

