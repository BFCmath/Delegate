# approaches/react/chain.py

from typing import List, TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers.string import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.tools import BaseTool

from .prompts import REACT_PROMPT, REPORT_GENERATOR_PROMPT

class ReActAction(BaseModel):
    """A Pydantic model that represents a valid agent action."""
    action: str = Field(description="The tool to use, must be one of ['Search', 'Finish']")
    action_input: str = Field(description="The input for the selected action.")

class ReActState(TypedDict):
    """Represents the state of our ReAct agent."""
    question: str
    scratchpad: Annotated[list, add_messages]
    iterations: int
    # This will now hold the final output dictionary
    final_output: Optional[dict]
    parsed_action: Optional[ReActAction]

def get_chain(model: BaseChatModel, search_tool: BaseTool) -> Runnable:
    """Constructs the LangGraph runnable for the ReAct research approach."""
    MAX_ITERATIONS = 10

    def _format_messages(messages: list) -> str:
        """Joins the content of messages into a single string."""
        return "\n".join([msg.content for msg in messages])

    def run_agent(state: ReActState):
        """Runs the agent LLM to decide the next action."""
        parser = PydanticOutputParser(pydantic_object=ReActAction)
        agent_chain = (
            RunnablePassthrough.assign(
                format_instructions=lambda _: parser.get_format_instructions()
            )
            | REACT_PROMPT
            | model
            | parser
        ).with_retry()
        formatted_scratchpad = _format_messages(state["scratchpad"])
        parsed_action_obj = agent_chain.invoke({
            "question": state["question"], "scratchpad": formatted_scratchpad
        })
        log_entry = (
            f"Thought: I will perform the action '{parsed_action_obj.action}' "
            f"with input '{parsed_action_obj.action_input}'.\n"
            f"Action: {parsed_action_obj.action}\n"
            f"Action Input: {parsed_action_obj.action_input}"
        )
        return {
            "scratchpad": [AIMessage(content=log_entry)],
            "iterations": state["iterations"] + 1,
            "parsed_action": parsed_action_obj,
        }

    def run_tool(state: ReActState):
        """Executes the tool chosen by the agent."""
        action = state["parsed_action"]
        if action.action == "Search":
            result = search_tool.invoke(action.action_input)
            return {"scratchpad": [HumanMessage(content=f"Observation: {result}")]}
        return {"scratchpad": [HumanMessage(content="Observation: Invalid action specified.")]}

    def should_continue(state: ReActState):
        """Determines whether to continue the loop or end."""
        action = state["parsed_action"]
        if action.action == "Finish" or state["iterations"] >= MAX_ITERATIONS:
            return "end"
        return "continue"
    
    # --- START OF FIX ---
    # Merge report generation and final structuring into a single node.
    def generate_final_output_node(state: ReActState):
        """Synthesizes the final report and structures the output."""
        print("--- Synthesizing final report ---")
        
        # 1. Generate the report
        research_summary = _format_messages(state["scratchpad"])
        report_generator_chain = REPORT_GENERATOR_PROMPT | model | StrOutputParser()
        article = report_generator_chain.invoke({
            "question": state["question"],
            "research_summary": research_summary,
        })
        
        # 2. Calculate the metadata
        search_count = sum(1 for msg in state["scratchpad"] if isinstance(msg, AIMessage) and "Action: Search" in msg.content)
        
        # 3. Return the final, structured dictionary
        final_output_dict = {
            "article": article,
            "metadata": {"search_count": search_count},
        }
        return {"final_output": final_output_dict}
    # --- END OF FIX ---

    # Build and compile the graph
    graph = StateGraph(ReActState)
    graph.add_node("agent", run_agent)
    graph.add_node("tool", run_tool)
    # The new node replaces the old report generator
    graph.add_node("generate_final_output", generate_final_output_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"continue": "tool", "end": "generate_final_output"}
    )
    graph.add_edge("tool", "agent")
    graph.add_edge("generate_final_output", END)
    
    runnable_graph = graph.compile()
    
    # --- START OF FIX ---
    # The final chain is now simpler. It invokes the graph and then extracts
    # the 'final_output' dictionary from the graph's final state.
    final_chain = RunnableLambda(
        lambda x: {"question": x["topic"], "scratchpad": [], "iterations": 0}
    ) | runnable_graph
    # --- END OF FIX ---

    return final_chain