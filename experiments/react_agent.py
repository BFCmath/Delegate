# experiments/react_agent.py
"""
Custom ReAct Agent for Deep Research Tasks
Supports both API-based models (Gemini) and local models (Qwen)
"""
import re
import time
import asyncio
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass

@dataclass
class ReActResult:
    """Result from ReAct agent execution"""
    article: str
    metadata: Dict
    scratchpad: List[str]  # Full conversation history


class ReActAgent:
    """
    ReAct (Reasoning + Acting) agent for deep research.
    
    Implements iterative loop: Thought → Action → Observation
    """
    
    def __init__(
        self,
        model,  # Can be Gemini model, local model wrapper, or key_manager
        search_tool: Callable[[str], List[Dict]],
        max_iterations: int = 10,
        is_local_model: bool = False,
        model_config: Dict = None  # For API models with key rotation
    ):
        """
        Initialize ReAct agent.
        
        Args:
            model: LLM model (Gemini GenerativeModel, local model wrapper, or APIKeyManager)
            search_tool: Function that takes query string and returns search results
            max_iterations: Maximum number of search iterations
            is_local_model: True if using local model (transformers), False for API
            model_config: Config for API models (model_name, generation_config) for key rotation
        """
        self.model = model
        self.search_tool = search_tool
        self.max_iterations = max_iterations
        self.is_local_model = is_local_model
        self.model_config = model_config or {}
        
        # Check if model is actually a key manager (for key rotation)
        self.use_key_rotation = hasattr(model, 'get_model')
        
        # Import prompts
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from prompts import get_react_prompts
        
        self.prompts = get_react_prompts(max_iterations=max_iterations)
    
    def parse_action(self, response: str) -> Dict[str, str]:
        """
        Parse action from LLM response.
        
        Looks for patterns:
        - Search[query]
        - Finish[report]
        
        Args:
            response: LLM response text
            
        Returns:
            Dict with 'type' and 'content' keys
        """
        # Look for Finish action first (can contain multiple brackets)
        finish_match = re.search(r'Finish\[(.*)\]', response, re.DOTALL)
        if finish_match:
            content = finish_match.group(1).strip()
            return {"type": "Finish", "content": content}
        
        # Look for Search action
        search_match = re.search(r'Search\[(.*?)\]', response)
        if search_match:
            query = search_match.group(1).strip()
            return {"type": "Search", "query": query}
        
        # No clear action found - treat as continuation/thought
        return {"type": "Continue", "content": response}
    
    def format_observation(self, results: List[Dict[str, str]]) -> str:
        """
        Format search results for LLM consumption.
        
        Args:
            results: List of dicts with 'title', 'url', 'content'
            
        Returns:
            Formatted string
        """
        if not results:
            return "Observation: No search results found."
        
        formatted = ["Observation: Search results:"]
        for i, result in enumerate(results, 1):
            formatted.append(
                f"\n[{i}] {result['title']}\n"
                f"URL: {result['url']}\n"
                f"{result['content'][:500]}..."  # Truncate long content
            )
        
        return "\n".join(formatted)
    
    async def run(self, question: str) -> ReActResult:
        """
        Run ReAct loop for given research question.
        
        Args:
            question: Research question to investigate
            
        Returns:
            ReActResult with article and metadata
        """
        print(f"🤖 Starting ReAct agent for: {question[:80]}...")
        
        scratchpad = []
        search_count = 0
        iteration = 0
        
        # Initialize with system prompt and question
        system_prompt = self.prompts['system']
        user_prompt = self.prompts['user'].format(question=question)
        
        # Start conversation
        scratchpad.append(f"System: {system_prompt}")
        scratchpad.append(f"User: {user_prompt}")
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"  Iteration {iteration}/{self.max_iterations}...")
            
            # Build full context
            full_context = "\n\n".join(scratchpad)
            
            # Get LLM response with retry logic
            max_retries = 3
            retry_count = 0
            response = None
            
            while retry_count < max_retries:
                try:
                    if self.is_local_model:
                        # Local model (synchronous)
                        response = self.model.generate(full_context)
                    else:
                        # API model (async)
                        if self.use_key_rotation:
                            # Get fresh model with next API key for this iteration
                            current_model = self.model.get_model(**self.model_config)
                            response = await asyncio.to_thread(
                                current_model.generate_content,
                                full_context
                            )
                        else:
                            # Single model (backward compatibility)
                            response = await asyncio.to_thread(
                                self.model.generate_content,
                                full_context
                            )
                        response = response.text
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_msg = str(e)
                    retry_count += 1
                    
                    # Check if it's a quota/rate limit error
                    if "429" in error_msg or "Quota exceeded" in error_msg or "RATE_LIMIT_EXCEEDED" in error_msg:
                        if retry_count < max_retries:
                            wait_time = 2 ** retry_count  # Exponential backoff: 2s, 4s, 8s
                            print(f"⚠️  Quota limit hit, waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                            await asyncio.sleep(wait_time)
                        else:
                            print(f"❌ Quota exceeded after {max_retries} retries: {error_msg}")
                            response = f"Error: Quota exceeded after retries"
                    else:
                        print(f"❌ Error getting LLM response: {error_msg}")
                        response = f"Error: {error_msg}"
                        break  # Non-quota error, don't retry
            
            scratchpad.append(f"Assistant: {response}")
            
            # Parse action
            action = self.parse_action(response)
            
            if action["type"] == "Finish":
                print(f"✅ ReAct completed after {iteration} iterations")
                return ReActResult(
                    article=action["content"],
                    metadata={
                        "search_count": search_count,
                        "iterations": iteration,
                        "completed": True
                    },
                    scratchpad=scratchpad
                )
            
            elif action["type"] == "Search":
                search_count += 1
                query = action["query"]
                print(f"    🔍 Search #{search_count}: {query}")
                
                # Execute search
                try:
                    results = self.search_tool(query)
                    observation = self.format_observation(results)
                except Exception as e:
                    print(f"    ⚠️  Search error: {e}")
                    observation = f"Observation: Search failed - {str(e)}"
                
                scratchpad.append(observation)
            
            else:
                # Continue - just a thought, keep going
                pass
        
        # Max iterations reached without Finish
        print(f"⚠️  Max iterations reached without Finish action")
        
        # Try to extract final report from scratchpad
        final_content = "\n\n".join(scratchpad)
        
        return ReActResult(
            article=final_content,
            metadata={
                "search_count": search_count,
                "iterations": iteration,
                "completed": False,
                "reason": "max_iterations_reached"
            },
            scratchpad=scratchpad
        )


class LocalModelWrapper:
    """
    Wrapper for local transformers models to provide consistent interface.
    """
    
    def __init__(self, model, tokenizer, max_new_tokens: int = 2048):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
    
    def generate(self, prompt: str) -> str:
        """Generate response from local model"""
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response

