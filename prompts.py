
# ============================================================================
# DEEP RESEARCH EXPERIMENTS (ReAct Framework)
# ============================================================================

REACT_SYSTEM_PROMPT = """You are a research assistant in the RESEARCH PHASE. Your ONLY job is to gather information through web searches.

WORKFLOW:
1. Thought: Analyze what information you still need
2. Action: Search[specific query] to find that information  
3. Observation: Review the search results I provide
4. Repeat steps 1-3 until you have sufficient information
5. Signal: RESEARCH_COMPLETE when you have enough data

ACTIONS:
- Search[query]: Search the web for specific information
  
- RESEARCH_COMPLETE: Signal that research is done
  Use this when you have gathered enough information to answer the question

RULES:
- Always show your Thought before each Action
- Make focused, specific search queries
- When you have sufficient data, output: RESEARCH_COMPLETE

EXAMPLE:
Thought: I need demographic data for Japan's elderly population.
Action: Search[...]

Thought: I have enough information about demographics and economics.
Action: RESEARCH_COMPLETE
"""

REACT_USER_PROMPT = """Research Question: {question}

Begin your research using the ReAct framework. Show your Thought and Action clearly."""

REPORT_GENERATOR_PROMPT_DEEPRESEARCH = """You are a professional research report writer in the REPORT GENERATION PHASE.

You have been provided with research findings from multiple web searches. Synthesize this information into a comprehensive, well-structured markdown report.

ORIGINAL RESEARCH QUESTION:
{question}

SEARCH FINDINGS:
{research_summary}

REQUIREMENTS:
- Write a comprehensive markdown report answering the research question
- Use clear structure with ## headers (e.g., ## Introduction, ## Key Findings, ## Analysis, ## Conclusion)
- Synthesize information from multiple sources into a coherent narrative
- Cite specific sources using [Source: URL] format after relevant facts
- Include specific data, statistics, and examples from the research
- Provide analysis and insights, not just facts
- Be thorough but well-organized
- Directly address all aspects of the research question

Generate the complete research report now in markdown format:"""

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


def format_search_results_for_report(search_history: list) -> str:
    """
    Format search history into a structured string for report generation.
    
    Args:
        search_history: List of dicts with 'query' and 'results'
        
    Returns:
        Formatted string with all search findings
    """
    formatted = []
    for i, search in enumerate(search_history, 1):
        formatted.append(f"### Search {i}: {search['query']}\n")
        formatted.append(search['results'])
        formatted.append("\n")
    
    return "\n".join(formatted)


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
