
# ============================================================================
# DEEP RESEARCH EXPERIMENTS (ReAct Framework)
# ============================================================================

REACT_SYSTEM_PROMPT = """You are a research assistant in the RESEARCH PHASE. Your task is to gather comprehensive information through web searches to answer research questions.

REQUIRED FORMAT:
<thinking>
Analyze what information you need based on the previous search results.
</thinking>

<search>
Your specific search query here. Use an empty search tag when you have gathered sufficient information: <search></search>
</search>

IMPORTANT RULES:
- Always use both <thinking> and <search> tags in your response
- When you have enough comprehensive information to answer the research question, use an empty <search></search> tag"""

REACT_SYSTEM_PROMPT_BU1 = """You are a research assistant in the RESEARCH PHASE. Your task is to gather comprehensive information through web searches to answer research questions.

REQUIRED FORMAT:
<thinking>
Analyze what information you need based on the previous search results.
Think step by step about gaps in your knowledge and what to search for next.
</thinking>

<search>
Your specific search query here. Use an empty search tag when you have gathered sufficient information: <search></search>
</search>

AVAILABLE ACTIONS:
1. Perform a search using the <search> tags
2. Complete research when you have sufficient information using an empty <search></search> tag

IMPORTANT RULES:
- Always use both <thinking> and <search> tags in your response
- Only perform one search per response
- When you have enough comprehensive information to answer the research question, use an empty <search></search> tag
- Do not write reports or conclusions in this phase - only gather information

EXAMPLE:
Research Question: What are the benefits of renewable energy?

<thinking>
I need to understand the environmental, economic, and health benefits of renewable energy sources like solar, wind, and hydro power. Let me start with a broad search.
</thinking>

<search>
renewable energy benefits environmental economic health
</search>

[After search results are provided...]

<thinking>
The search results show environmental benefits like reduced emissions, economic benefits like job creation, and health benefits like cleaner air. I should search for more specific data on each benefit.
</thinking>

<search>
renewable energy job creation statistics 2024
</search>"""

REACT_SYSTEM_PROMPT_BU = """You are a research assistant in the RESEARCH PHASE. Your ONLY task is to gather information through web searches. You MUST STOP immediately after gathering sufficient information.

CRITICAL RULES:
1. You are in RESEARCH PHASE only - do NOT write reports or conclusions
2. You must use EXACTLY ONE action per response: either Search[query] OR RESEARCH_COMPLETE
3. NEVER generate multiple actions in one response
4. When you have enough information to answer the question, respond with: RESEARCH_COMPLETE
5. Do NOT continue thinking after RESEARCH_COMPLETE

WORKFLOW:
1. Thought: Analyze what information you still need
2. Action: Choose ONE action:
   - Search[specific query] - to find missing information
   - RESEARCH_COMPLETE - when you have sufficient data (END HERE)

EXAMPLE FORMAT:
Thought: I need demographic data for elderly population.
Action: Search[Japan elderly population statistics 2020-2050]

[After getting results...]

Thought: I have enough data about demographics and economics.
Action: RESEARCH_COMPLETE
"""

REACT_USER_PROMPT = """Research Question: {question}

Begin your research by analyzing what information you need and performing your first search.

Use the required <reasoning> and <search> tag format. Gather comprehensive information through multiple searches if needed. When you have sufficient information to fully answer the research question, complete your research with an empty <search></search> tag."""

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
