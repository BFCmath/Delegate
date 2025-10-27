# approaches/zeroshot/prompts.py

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 1. Define the desired data structure for the query generator
class QueryList(BaseModel):
    """A Pydantic model that represents a list of search queries."""
    queries: list[str] = Field(
        description="A list of 3-5 diverse search queries related to the user's topic."
    )

# 2. Create the prompt for the query generator.
#    It now includes a placeholder for format_instructions, which will be
#    provided automatically by the PydanticOutputParser.
QUERY_GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert research assistant. Your goal is to generate a list of "
            "highly relevant and diverse search queries based on the user's research topic.\n"
            "{format_instructions}", # This is where the parser injects its instructions
        ),
        (
            "user",
            "Research Topic: {topic}",
        ),
    ]
)

# 3. The report generation prompt remains unchanged.
REPORT_GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional report writer. Your task is to synthesize the provided "
            "information into a comprehensive, well-structured, and easy-to-read report. "
            "The report should directly address the user's research topic. "
            "Use markdown for formatting.",
        ),
        (
            "user",
            "Research Topic: {topic}\n\n"
            "Collected Information:\n"
            "---------------------\n"
            "{search_results}\n"
            "---------------------\n\n"
            "Based on the information above, please generate a detailed report.",
        ),
    ]
)