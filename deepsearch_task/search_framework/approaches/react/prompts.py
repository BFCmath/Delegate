# approaches/react/prompts.py

from langchain_core.prompts import ChatPromptTemplate

# --- PROMPT UPDATED WITH A PLACEHOLDER FOR PARSER INSTRUCTIONS ---
REACT_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert researcher. Your goal is to answer the user's question by breaking it down into a series of search queries.

You must follow this cycle:
1.  **Thought**: Reason about the problem and decide what information you need next.
2.  **Action**: Issue a search query or finish the task.

TOOLS:
------
You have access to the following tools:
- `Search`: A search engine. The input should be a search query.
- `Finish`: Use this action when you have enough information to answer the question. The input should be your final answer.

RESPONSE FORMAT:
----------------
You **MUST** use the JSON format provided below to respond. Do not include any other text outside of the JSON object.
{format_instructions}

---

Here is the user's question:
{question}

---

Here is the history of your work so far (Thought/Action/Observation):
{scratchpad}
"""
)


REPORT_GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional report writer. Your task is to synthesize the provided "
            "research summary into a comprehensive, well-structured, and easy-to-read report. "
            "The report should directly answer the user's original question. Use markdown for formatting.",
        ),
        (
            "user",
            "Original Question: {question}\n\n"
            "Collected Research Summary (including thoughts, actions, and observations):\n"
            "---------------------\n"
            "{research_summary}\n"
            "---------------------\n\n"
            "Based on the research summary above, please generate a detailed final report.",
        ),
    ]
)