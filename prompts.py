
# ============================================================================
# DEEP RESEARCH EXPERIMENTS (ReAct Framework)
# ============================================================================

REACT_SYSTEM_PROMPT = """You are a research assistant using the ReAct (Reasoning + Acting) framework to conduct deep research.

WORKFLOW:
1. Thought: Analyze what information you need next
2. Action: Choose ONE action:
   - Search[specific query]: Search the web for information
   - Finish[complete markdown report]: End with your final research report
3. Observation: Review the search results I provide
4. Repeat steps 1-3 until you have sufficient information

ACTIONS FORMAT:
- Search[query]: e.g., "Search[latest developments in quantum computing 2024]"
- Finish[report]: e.g., "Finish[## Research Report on Quantum Computing\n\n...]"

RULES:
- Always show your Thought before taking an Action
- Make focused, specific search queries (not too broad)
- After {max_iterations} searches, you MUST use Finish action
- In your final report:
  * Use markdown formatting
  * Structure with clear sections
  * Cite sources with URLs when possible
  * Be comprehensive but concise
  * Directly address the research question

EXAMPLE:
Thought: I need to understand the current state of quantum computing.
Action: Search[quantum computing breakthroughs 2024]

[After receiving results...]

Thought: I have enough information to write the report now.
Action: Finish[## Quantum Computing Research Report\n\n### Overview\n...]
"""

REACT_USER_PROMPT = """Research Question: {question}

Begin your research using the ReAct framework. Show your Thought and Action clearly."""

REPORT_GENERATOR_PROMPT_DEEPRESEARCH = """You are a professional research report writer.

Your task: Synthesize the provided research findings into a comprehensive, well-structured markdown report.

Requirements:
- Use clear markdown formatting with headers, lists, and emphasis
- Structure logically with introduction, main sections, and conclusion
- Cite sources with URLs in your text
- Be thorough but concise
- Directly answer the original research question

Original Research Question:
{question}

Collected Research Findings:
{research_summary}

Generate your final research report now:"""

def get_react_prompts(max_iterations: int = 10):
    """
    Get ReAct framework prompts for deep research.
    
    Args:
        max_iterations: Maximum number of search iterations allowed
        
    Returns:
        Dictionary with 'system', 'user', 'report_generator' prompts
    """
    return {
        'system': REACT_SYSTEM_PROMPT.format(max_iterations=max_iterations),
        'user': REACT_USER_PROMPT,
        'report_generator': REPORT_GENERATOR_PROMPT_DEEPRESEARCH
    }


# ============================================================================
# PROMPT VERSIONING & TRACKING
# ============================================================================

PROMPT_VERSION = "3.0.0"
LAST_UPDATED = "2024-10-27"

PROMPT_CHANGELOG = """
Version 3.0.0 (2024-10-27):
- MAJOR: Added Deep Research experiment prompts with ReAct framework
- Added REACT_SYSTEM_PROMPT for research assistant workflow
- Added REACT_USER_PROMPT for research questions
- Added REPORT_GENERATOR_PROMPT_DEEPRESEARCH for final synthesis
- Added get_react_prompts() helper function
- Support for web search + iterative reasoning loop
- Designed for comprehensive research article generation

Version 2.0.0 (2024-10-12):
- MAJOR: Comprehensive router prompt rewrite to fix "No Response Generated" issue
- Added detailed workflow with 5 clear steps
- Added 3 concrete examples showing word problem → expression conversion
- Emphasized: "ALWAYS respond after receiving tool result"
- Emphasized: "Formulate MATHEMATICAL EXPRESSIONS, not word problems"
- Improved tool description to be clearer about expected input format
- Changed SLM response format to: "The calculation result is: X"
- Added explicit instruction: "Never leave the response empty"
- Added alternative ROUTER_INSTRUCTIONS_EXPERIMENT_MULTISTEP for complex problems
- Added get_router_instructions() helper function

Version 1.0.0 (2024-10-11):
- Initial centralized prompt system
- Extracted prompts from experiments/llm_experiment.py
- Extracted prompts from experiments/slm_experiment.py
- Extracted prompts from experiments/router_agent.py
- Extracted prompts from router_agent_demo.py
- Added helper functions for easy access
- Added template for custom domains
"""
