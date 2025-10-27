# base_framework.py

import importlib
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler

# 1. New callback handler to count LLM calls
class LLMCallCounterCallback(BaseCallbackHandler):
    """Callback handler for counting LLM calls."""
    def __init__(self):
        super().__init__()
        self.llm_calls = 0

    def on_llm_start(self, *args, **kwargs) -> None:
        """Increment the counter on LLM start."""
        self.llm_calls += 1

def run_pipeline(topic: str, model: BaseChatModel, approach_name: str) -> dict:
    """
    The main interface to run a deep research pipeline.

    Args:
        topic (str): The research topic provided by the user.
        model (BaseChatModel): An initialized LangChain chat model.
        approach_name (str): The name of the approach folder.

    Returns:
        A dictionary containing the final article and performance metadata.
    """
    print(f"🚀 Starting pipeline for topic: '{topic}' using '{approach_name}' approach...")

    try:
        approach_module = importlib.import_module(f"approaches.{approach_name}.chain")
    except ModuleNotFoundError:
        raise ValueError(
            f"Approach '{approach_name}' not found or does not contain a 'chain.py' file."
        )

    get_chain_func = getattr(approach_module, "get_chain")

    from tools.search import get_search_tool
    search_tool = get_search_tool()

    research_chain = get_chain_func(model, search_tool)
    
    # 2. Instantiate and use the new call counter callback
    call_counter_callback = LLMCallCounterCallback()
    
    chain_output = research_chain.invoke(
        {"topic": topic},
        config={"callbacks": [call_counter_callback]}
    )
    
    print("✅ Pipeline finished successfully.")

    # 3. Assemble the final result dictionary with the new metric
    final_data = chain_output.get("final_output", {})
    metadata = final_data.get("metadata", {})
    
    result = {
        "article": final_data.get("article", "Error: No article generated."),
        "metadata": {
            "llm_calls": call_counter_callback.llm_calls,
            "search_count": metadata.get("search_count", 0)
        }
    }
    
    return result