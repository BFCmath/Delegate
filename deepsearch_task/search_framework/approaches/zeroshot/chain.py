# approaches/zeroshot/chain.py

from operator import itemgetter
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.tools import BaseTool

from .prompts import QueryList, QUERY_GENERATOR_PROMPT, REPORT_GENERATOR_PROMPT

def _format_search_results(results: list[list[dict]]) -> str:
    # ... (this function remains the same as before)
    formatted_string = []
    result_counter = 1
    for query_results in results:
        for result in query_results:
            formatted_string.append(
                f"Result {result_counter}:\n"
                f"Title: {result.get('title', 'N/A')}\n"
                f"URL: {result.get('url', 'N/A')}\n"
                f"Content: {result.get('content', 'No content available.')}"
            )
            result_counter += 1
    return "\n\n---\n\n".join(formatted_string)

# --- START OF CHANGE ---
def _structure_final_output(input_dict: dict) -> dict:
    """Structures the final output to include the article and metadata."""
    return {
        "article": input_dict.get("report"),
        "search_count": len(input_dict.get("queries_object").queries)
    }
# --- END OF CHANGE ---

def get_chain(model: BaseChatModel, search_tool: BaseTool) -> Runnable:
    """
    Constructs and returns the LCEL chain for the zeroshot research approach.
    The chain now outputs a dictionary with the final article and search count.
    """
    pydantic_parser = PydanticOutputParser(pydantic_object=QueryList)

    generate_queries_chain = (
        RunnablePassthrough.assign(format_instructions=lambda _: pydantic_parser.get_format_instructions())
        | QUERY_GENERATOR_PROMPT
        | model
        | pydantic_parser
    ).with_retry(stop_after_attempt=3)

    generate_report_chain = REPORT_GENERATOR_PROMPT | model | StrOutputParser()

    zeroshot_chain = (
        {
            "queries_object": generate_queries_chain,
            "topic": itemgetter("topic"),
        }
        | RunnablePassthrough.assign(
            search_results=(
                itemgetter("queries_object")
                | RunnableLambda(lambda x: x.queries)
                | search_tool.map()
            )
        )
        | RunnablePassthrough.assign(
            search_results=itemgetter("search_results") | RunnableLambda(_format_search_results)
        )
        # --- START OF CHANGE ---
        # Instead of ending with the report generator, we assign its output
        # to a new key 'report' and then structure the final output.
        | RunnablePassthrough.assign(report=generate_report_chain)
        | RunnableLambda(_structure_final_output)
        # --- END OF CHANGE ---
    )

    return zeroshot_chain