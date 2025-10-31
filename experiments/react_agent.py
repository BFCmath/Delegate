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
        Parse action from LLM response using XML-like tags.

        Looks for patterns:
        - <search>query</search> - for search actions
        - <search></search> - for completion (empty search tag)
        - Alternative completion phrases as fallback

        Handles flexible whitespace in tags and provides robust parsing.

        Args:
            response: LLM response text

        Returns:
            Dict with 'type' and 'content' keys
        """
        # Clean response and look for search tag with flexible whitespace
        response = response.strip()

        # More robust regex to handle whitespace around tags
        search_match = re.search(r'<search\s*>\s*(.*?)\s*</search\s*>', response, re.DOTALL | re.IGNORECASE)

        if search_match:
            query = search_match.group(1).strip()

            # Empty search tag means research complete
            if not query:
                return {"type": "ResearchComplete"}

            # Non-empty search tag means search action
            return {"type": "Search", "query": query}

        # Check for alternative completion patterns
        response_lower = response.lower()
        if ("research complete" in response_lower or
            "i have gathered sufficient information" in response_lower or
            "sufficient data collected" in response_lower or
            "research is complete" in response_lower):
            return {"type": "ResearchComplete"}

        # No clear action found - treat as continuation/thought
        return {"type": "Continue", "content": response}

    def extract_thinking(self, response: str) -> str:
        """
        Extract thinking content from LLM response for debugging.

        Args:
            response: LLM response text

        Returns:
            Thinking content or empty string if not found
        """
        thinking_match = re.search(r'<thinking\s*>(.*?)</thinking\s*>', response, re.DOTALL | re.IGNORECASE)
        if thinking_match:
            return thinking_match.group(1).strip()
        return ""
    
    def format_observation(self, results: List[Dict[str, str]]) -> str:
        """
        Format search results for LLM consumption using <output> tags.

        Args:
            results: List of dicts with 'title', 'url', 'content'

        Returns:
            Formatted string wrapped in <output> tags
        """
        if not results:
            return "<output>No search results found.</output>"

        formatted = ["<output>Search results:"]
        for i, result in enumerate(results, 1):
            formatted.append(
                f"\n[{i}] {result['title']}\n"
                f"URL: {result['url']}\n"
                f"{result['content'][:500]}..."  # Truncate long content
            )

        formatted.append("</output>")
        return "\n".join(formatted)
    
    async def run(self, question: str) -> ReActResult:
        """
        Run two-phase ReAct pipeline:
        Phase 1: Research (gather information via searches)
        Phase 2: Report Generation (synthesize into article)
        
        Args:
            question: Research question to investigate
            
        Returns:
            ReActResult with article and metadata
        """
        print(f"🤖 Starting ReAct agent for: {question[:80]}...")
        
        # ========== PHASE 1: RESEARCH ==========
        print("📚 Phase 1: Research (gathering information)")
        
        scratchpad = []
        search_history = []  # Track all searches for report generation
        search_count = 0
        iteration = 0
        research_complete = False
        
        # Initialize with system prompt and question
        system_prompt = self.prompts['system']
        user_prompt = self.prompts['user'].format(question=question)
        
        # Start conversation
        scratchpad.append(f"System: {system_prompt}")
        scratchpad.append(f"User: {user_prompt}")
        
        while iteration < self.max_iterations and not research_complete:
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
                            api_response = await asyncio.to_thread(
                                current_model.generate_content,
                                full_context
                            )
                        else:
                            # Single model (backward compatibility)
                            api_response = await asyncio.to_thread(
                                self.model.generate_content,
                                full_context
                            )
                        
                        # Check if response is valid before accessing text
                        if not api_response.candidates:
                            raise ValueError("No candidates in response - model returned empty result")
                        
                        candidate = api_response.candidates[0]
                        finish_reason = candidate.finish_reason
                        
                        # Check finish reason (0=UNSPECIFIED, 1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER)
                        if finish_reason == 3:  # SAFETY
                            raise ValueError(f"Content blocked by safety filter (SAFETY). This query may violate content policies.")
                        elif finish_reason == 4:  # RECITATION
                            raise ValueError(f"Content blocked (RECITATION). Response contained copyrighted material.")
                        elif finish_reason == 5:  # OTHER
                            raise ValueError(f"Content generation failed (OTHER). Unknown blocking reason.")
                        elif finish_reason not in [0, 1, 2]:  # Valid reasons are UNSPECIFIED, STOP, MAX_TOKENS
                            raise ValueError(f"Invalid finish_reason: {finish_reason}")
                        
                        # Check if text is available
                        if not candidate.content or not candidate.content.parts:
                            # Empty response with STOP - treat as empty thought, continue
                            if finish_reason == 1:  # STOP with no content
                                print(f"    ⚠️  Empty response (STOP with no content) - continuing...")
                                response = ""  # Empty response, will be treated as Continue
                            else:
                                raise ValueError(f"No text parts in response. Finish reason: {finish_reason}")
                        else:
                            response = api_response.text
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_msg = str(e)
                    retry_count += 1
                    
                    # Check if it's a content filter/safety error (don't retry)
                    if "safety filter" in error_msg.lower() or "content blocked" in error_msg.lower() or "SAFETY" in error_msg:
                        print(f"❌ Content blocked by safety filter: {error_msg}")
                        raise ValueError(f"Research phase failed - Content blocked: {error_msg}")
                    
                    # Check if it's a quota/rate limit error
                    if "429" in error_msg or "Quota exceeded" in error_msg or "RATE_LIMIT_EXCEEDED" in error_msg:
                        if retry_count < max_retries:
                            wait_time = 2 ** retry_count  # Exponential backoff: 2s, 4s, 8s
                            print(f"⚠️  Quota limit hit, waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                            await asyncio.sleep(wait_time)
                        else:
                            print(f"❌ Quota exceeded after {max_retries} retries: {error_msg}")
                            raise RuntimeError(f"Research phase failed - Quota exceeded after {max_retries} retries")
                    else:
                        print(f"❌ Error getting LLM response: {error_msg}")
                        raise RuntimeError(f"Research phase failed: {error_msg}")
            
            scratchpad.append(f"Assistant: {response}")

            # Parse action
            action = self.parse_action(response)

            if action["type"] == "ResearchComplete":
                print(f"✅ Research complete after {iteration} iterations")
                research_complete = True
                break

            # Additional check: if response indicates completion even without exact signal
            response_lower = response.lower()
            if ("have sufficient information" in response_lower or
                "have enough data" in response_lower or
                "research is complete" in response_lower or
                "gathered enough information" in response_lower):
                print(f"✅ Research complete detected (alternative phrasing) after {iteration} iterations")
                research_complete = True
                break
            
            elif action["type"] == "Search":
                search_count += 1
                query = action["query"]
                print(f"    🔍 Search #{search_count}: {query}")
                
                # Execute search
                try:
                    results = self.search_tool(query)
                    observation = self.format_observation(results)
                    
                    # Store search for report generation
                    search_history.append({
                        "query": query,
                        "results": observation
                    })
                    
                except Exception as e:
                    print(f"    ⚠️  Search error: {e}")
                    observation = f"<output>Search failed - {str(e)}</output>"
                
                scratchpad.append(observation)
            
            else:
                # Continue - just a thought, keep going
                pass
        
        # ========== PHASE 2: REPORT GENERATION ==========
        print("📝 Phase 2: Report Generation (synthesizing findings)")
        
        if search_count == 0:
            # No searches performed, return empty
            print("⚠️  No searches performed during research phase")
            return ReActResult(
                article="No research was conducted.",
                metadata={
                    "search_count": 0,
                    "iterations": iteration,
                    "completed": False,
                    "reason": "no_searches"
                },
                scratchpad=scratchpad
            )
        
        # Format search findings for report generator
        from prompts import format_search_results_for_report
        search_summary = format_search_results_for_report(search_history)
        
        # Create report generation prompt
        report_prompt = self.prompts['report_generator'].format(
            question=question,
            research_summary=search_summary
        )
        
        # Get report from model
        print("  Generating final report...")
        max_retries = 3
        retry_count = 0
        article = None
        
        while retry_count < max_retries and article is None:
            try:
                if self.is_local_model:
                    article = self.model.generate(report_prompt)
                else:
                    if self.use_key_rotation:
                        current_model = self.model.get_model(**self.model_config)
                        api_response = await asyncio.to_thread(
                            current_model.generate_content,
                            report_prompt
                        )
                    else:
                        api_response = await asyncio.to_thread(
                            self.model.generate_content,
                            report_prompt
                        )
                    
                    # Check if response is valid before accessing text
                    if not api_response.candidates:
                        raise ValueError("No candidates in response - model returned empty result")
                    
                    candidate = api_response.candidates[0]
                    finish_reason = candidate.finish_reason
                    
                    # Check finish reason
                    if finish_reason == 3:  # SAFETY
                        raise ValueError(f"Report generation blocked by safety filter. The synthesized content may violate policies.")
                    elif finish_reason == 4:  # RECITATION
                        raise ValueError(f"Report generation blocked (RECITATION). Content contained copyrighted material.")
                    elif finish_reason == 5:  # OTHER
                        raise ValueError(f"Report generation failed (OTHER). Unknown blocking reason.")
                    elif finish_reason not in [0, 1, 2]:
                        raise ValueError(f"Invalid finish_reason: {finish_reason}")
                    
                    # Check if text is available
                    if not candidate.content or not candidate.content.parts:
                        # Empty response - this shouldn't happen for report generation
                        raise ValueError(f"No text content in report generation response. Finish reason: {finish_reason}")
                    
                    article = api_response.text
                break
                
            except Exception as e:
                error_msg = str(e)
                retry_count += 1
                
                # Check if it's a content filter/safety error (don't retry, fail immediately)
                if "safety filter" in error_msg.lower() or "content blocked" in error_msg.lower() or "SAFETY" in error_msg:
                    print(f"❌ Report generation blocked by safety filter: {error_msg}")
                    raise ValueError(f"Report generation failed - Content blocked: {error_msg}")
                
                # Check if it's a quota/rate limit error
                if "429" in error_msg or "Quota exceeded" in error_msg:
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count
                        print(f"⚠️  Quota limit, waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"❌ Report generation failed after {max_retries} retries due to quota limits")
                        raise RuntimeError(f"Report generation failed - Quota exceeded after {max_retries} retries")
                else:
                    print(f"❌ Report generation error: {error_msg}")
                    raise RuntimeError(f"Report generation failed: {error_msg}")
        
        if article is None:
            raise RuntimeError("Report generation failed - No article generated after all retries")
        
        print(f"✅ Report generated ({len(article)} chars)")
        
        # Add report generation to scratchpad for debugging
        scratchpad.append(f"\n=== PHASE 2: REPORT GENERATION ===")
        scratchpad.append(f"Report Prompt: {report_prompt[:500]}...")
        scratchpad.append(f"Generated Article: {article[:500]}...")
        
        return ReActResult(
            article=article,
            metadata={
                "search_count": search_count,
                "iterations": iteration,
                "completed": research_complete or iteration >= self.max_iterations,
                "phases": {
                    "research": {
                        "iterations": iteration,
                        "searches": search_count,
                        "completed": research_complete
                    },
                    "report": {
                        "generated": True,
                        "length": len(article)
                    }
                }
            },
            scratchpad=scratchpad
        )


class QuantizationConfig:
    """
    Configuration class for different quantization methods.
    """

    SUPPORTED_METHODS = ["awq", "gptq", "bnb_4bit", "bnb_8bit", "none"]

    def __init__(self, method: str = "none", **kwargs):
        """
        Initialize quantization configuration.

        Args:
            method: Quantization method ("awq", "gptq", "bnb_4bit", "bnb_8bit", "none")
            **kwargs: Additional method-specific parameters
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported quantization method: {method}. "
                           f"Supported: {self.SUPPORTED_METHODS}")

        self.method = method
        self.kwargs = kwargs

        # Set default parameters based on method
        if method == "awq":
            self.kwargs.setdefault("quantization", "awq")
        elif method == "gptq":
            self.kwargs.setdefault("quantization", "gptq")
        elif method == "bnb_4bit":
            self.kwargs.setdefault("load_in_4bit", True)
            self.kwargs.setdefault("bnb_4bit_compute_dtype", "float16")
            self.kwargs.setdefault("bnb_4bit_use_double_quant", True)
        elif method == "bnb_8bit":
            self.kwargs.setdefault("load_in_8bit", True)

    def get_vllm_params(self):
        """Get parameters for vLLM initialization"""
        if self.method in ["awq", "gptq"]:
            return {"quantization": self.method, **self.kwargs}
        return {}

    def get_transformers_params(self):
        """Get parameters for transformers initialization"""
        if self.method.startswith("bnb"):
            return self.kwargs
        return {}

    def __str__(self):
        return f"QuantizationConfig(method={self.method}, kwargs={self.kwargs})"


class LocalModelWrapper:
    """
    Wrapper for local models with quantization support.
    Supports vLLM (AWQ/GPTQ) and transformers (BitsAndBytes) backends.
    """

    def __init__(self, model_name: str, max_new_tokens: int = 4096, quantization: QuantizationConfig = None):
        # Set default quantization config
        if quantization is None:
            quantization = QuantizationConfig("none")

        self.quantization = quantization
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        print(f"🚀 Initializing model: {model_name}")
        print(f"📊 Quantization: {quantization}")

        # Try vLLM first (supports AWQ/GPTQ quantization)
        if quantization.method in ["awq", "gptq", "none"]:
            success = self._init_vllm()
            if success:
                self.backend = "vllm"
                return

        # Fall back to transformers (supports BitsAndBytes quantization)
        if quantization.method.startswith("bnb") or not self.vllm_available:
            success = self._init_transformers()
            if success:
                self.backend = "transformers"
                return

        # If both fail, disable model
        print("❌ Failed to initialize model with any backend")
        self.available = False

    def _init_vllm(self):
        """Initialize with vLLM backend"""
        try:
            from vllm import LLM, SamplingParams
            self.vllm_available = True
        except ImportError:
            print("⚠️ vLLM not available, trying transformers fallback")
            self.vllm_available = False
            return False

        try:
            # Get vLLM parameters including quantization
            vllm_params = self.quantization.get_vllm_params()
            vllm_params.update({
                "model": self.model_name,
                "max_model_len": 8192,  # Context window
                "dtype": "half" if self.quantization.method == "none" else "auto"
            })

            print(f"🔧 vLLM params: {vllm_params}")
            self.model = LLM(**vllm_params)

            self.tokenizer = self.model.get_tokenizer()

            # Sampling parameters
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=self.max_new_tokens,
                stop=["<|im_end|>", "<|endoftext|>", "</s>"]
            )

            memory_info = self._get_memory_info()
            print(f"✅ vLLM model loaded successfully (context: 8192 tokens, {memory_info})")
            self.available = True
            return True

        except Exception as e:
            print(f"❌ Failed to load vLLM model: {e}")
            self.vllm_available = False
            return False

    def _init_transformers(self):
        """Initialize with transformers backend (BitsAndBytes fallback)"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            self.transformers_available = True
        except ImportError:
            print("❌ Transformers not available")
            self.transformers_available = False
            return False

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🔧 Loading transformers model on {device}")

            # Get quantization parameters
            quant_params = self.quantization.get_transformers_params()

            if quant_params:
                # Configure BitsAndBytes
                bnb_config = BitsAndBytesConfig(**quant_params)
                print(f"🔧 BitsAndBytes config: {quant_params}")

                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # Load without quantization
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True
                )

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            self.model = model
            self.tokenizer = tokenizer
            self.device = device

            memory_info = self._get_memory_info()
            print(f"✅ Transformers model loaded successfully ({memory_info})")
            self.available = True
            return True

        except Exception as e:
            print(f"❌ Failed to load transformers model: {e}")
            self.transformers_available = False
            return False

    def _get_memory_info(self):
        """Get memory usage information"""
        try:
            import torch
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                return ".1f"
            return "CPU mode"
        except:
            return "unknown"

    def generate(self, prompt: str) -> str:
        """Generate response using the appropriate backend"""
        if not hasattr(self, 'available') or not self.available:
            return "Error: Model not available"

        if self.backend == "vllm":
            return self._generate_vllm(prompt)
        elif self.backend == "transformers":
            return self._generate_transformers(prompt)
        else:
            return "Error: Unknown backend"

    def _generate_vllm(self, prompt: str) -> str:
        """Generate using vLLM backend"""
        try:
            outputs = self.model.generate([prompt], self.sampling_params)
            generated_text = outputs[0].outputs[0].text
            return generated_text.strip()
        except Exception as e:
            print(f"❌ vLLM generation error: {e}")
            return f"Error: vLLM generation failed - {str(e)}"

    def _generate_transformers(self, prompt: str) -> str:
        """Generate using transformers backend"""
        try:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stop_strings=["<|im_end|>", "<|endoftext|>", "</s>"],
                )

            # Decode and clean response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the response
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            return generated_text

        except Exception as e:
            print(f"❌ Transformers generation error: {e}")
            return f"Error: Transformers generation failed - {str(e)}"

