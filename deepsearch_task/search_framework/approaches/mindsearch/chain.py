# approaches/mindsearch/chain.py

import json
import re
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable, RunnableLambda

from .prompts import (
    SEARCHER_SYSTEM_PROMPT_EN,
    FEWSHOT_EXAMPLE_EN,
    SEARCHER_INPUT_TEMPLATE_EN,
    SEARCHER_CONTEXT_TEMPLATE_EN,
    GRAPH_PROMPT_EN,
    GRAPH_FEWSHOT_EXAMPLE_EN,
    FINAL_RESPONSE_EN
)


# ============ WebSearchGraph Implementation ============

class WebSearchGraph:
    def __init__(self, model: BaseChatModel, search_tool: BaseTool):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.adjacency_list: Dict[str, List[dict]] = defaultdict(list)
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.futures = []
        self.model = model
        self.search_tool = search_tool
        
    def add_root_node(self, node_content: str, node_name: str = "root"):
        """Add the initial question as root node"""
        self.nodes[node_name] = dict(content=node_content, type="root")
        self.adjacency_list[node_name] = []
        
    def add_node(self, node_name: str, node_content: str):
        """Add a search sub-question node and trigger search"""
        self.nodes[node_name] = dict(content=node_content, type="searcher")
        self.adjacency_list[node_name] = []
        
        # Get parent nodes' responses for context
        parent_nodes = []
        for start_node, adj in self.adjacency_list.items():
            for neighbor in adj:
                if (node_name == neighbor['name'] and 
                    start_node in self.nodes and 
                    "response" in self.nodes[start_node]):
                    parent_nodes.append(self.nodes[start_node])
                    
        parent_response = [
            dict(question=node["content"], answer=node["response"]['content'])
            for node in parent_nodes
        ]
        
        # Submit search task to thread pool
        future = self.executor.submit(
            self._search_node,
            node_name,
            node_content,
            self.nodes["root"]["content"],
            parent_response
        )
        self.futures.append(future)
        
    def _search_node(self, node_name: str, question: str, topic: str, history: List[dict]):
        """Execute search for a node using SearcherAgent"""
        searcher = SearcherAgent(self.model, self.search_tool)
        response = searcher.search(question, topic, history)
        
        # Store the response in the node
        self.nodes[node_name]["response"] = {
            'content': response['content'],
            'citations': response.get('citations', {})
        }
        self.nodes[node_name]["memory"] = response.get('memory', {})
        
        return node_name
        
    def add_response_node(self, node_name: str = "response"):
        """Add response node indicating search completion"""
        self.nodes[node_name] = dict(type="end")
        
    def add_edge(self, start_node: str, end_node: str):
        """Add edge between nodes"""
        self.adjacency_list[start_node].append(
            dict(id=str(uuid.uuid4()), name=end_node, state=2)
        )
        
    def node(self, node_name: str) -> dict:
        """Get node information"""
        return self.nodes.get(node_name, {}).copy()
        
    def wait_for_all_searches(self):
        """Wait for all search tasks to complete"""
        if not self.futures:
            return
            
        for future in as_completed(self.futures):
            try:
                future.result(timeout=120)
            except Exception as e:
                print(f"Search failed: {e}")
        self.futures.clear()
        
    def reset(self):
        """Reset the graph"""
        self.nodes = {}
        self.adjacency_list = defaultdict(list)
        self.futures = []


# ============ Searcher Agent Implementation ============

class SearcherAgent:
    def __init__(self, model: BaseChatModel, search_tool: BaseTool):
        """Initialize searcher with LLM and search tool"""
        self.llm = model
        self.search_tool = search_tool
        
    def search(self, question: str, topic: str, history: List[dict]) -> dict:
        """Execute search for a specific question"""
        
        # Build context from history
        context_messages = []
        if history:
            for item in history:
                context_messages.append(
                    SEARCHER_CONTEXT_TEMPLATE_EN.format(
                        question=item['question'],
                        answer=item['answer']
                    )
                )
        
        # Build the input message
        input_message = SEARCHER_INPUT_TEMPLATE_EN.format(
            topic=topic,
            question=question
        )
        
        if context_messages:
            input_message = "\n".join(context_messages) + "\n" + input_message
            
        # Create tool info for the prompt
        tool_info = """
- FastWebBrowser.search: Search the web for information
  Parameters: {"query": ["search query 1", "search query 2"]}
"""
        
        system_prompt = SEARCHER_SYSTEM_PROMPT_EN.format(tool_info=tool_info)
        system_prompt += "\n" + FEWSHOT_EXAMPLE_EN
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=input_message)
        ]
        
        # ReAct loop for searching
        max_iterations = 2
        search_results = {}
        final_response = ""
        
        for i in range(max_iterations):
            response = self.llm.invoke(messages)
            content = response.content
            # print("Sleep for 10 seconds searcher")
            # import time
            # time.sleep(10)
            # Check if there's an action to execute
            if "<|action_start|>" in content and "<|action_end|>" in content:
                # Extract action
                action_match = re.search(
                    r'<\|action_start\|><\|plugin\|>(.*?)<\|action_end\|>',
                    content,
                    re.DOTALL
                )
                
                if action_match:
                    action_json = action_match.group(1)
                    try:
                        action_data = json.loads(action_json)
                        
                        if action_data['name'] == 'FastWebBrowser.search':
                            queries = action_data['parameters']['query']
                            if isinstance(queries, str):
                                queries = [queries]
                                
                            # Execute searches
                            all_results = []
                            for idx, query in enumerate(queries):
                                results = self.search_tool.invoke(query)
                                for r_idx, result in enumerate(results):
                                    citation_idx = len(search_results) + 1
                                    search_results[citation_idx] = result.get('url', '')
                                    all_results.append(
                                        f"[[{citation_idx}]] {result.get('content', '')}"
                                    )
                            
                            # Add search results to conversation
                            search_response = "\n".join(all_results)
                            messages.append(AIMessage(content=content))
                            messages.append(
                                HumanMessage(
                                    content=f"Search Results:\n{search_response}"
                                )
                            )
                            
                    except json.JSONDecodeError:
                        break
            else:
                # No more actions, this is the final response
                final_response = content
                break
        
        # Ensure we have a valid response
        if not final_response or final_response.strip() == "":
            final_response = f"Unable to generate a comprehensive answer for: {question}. Search may have failed or returned insufficient results."
                
        return {
            'content': final_response,
            'citations': search_results,
            'memory': {'messages': messages}
        }


# ============ MindSearch Planner Agent ============

class MindSearchAgent:
    def __init__(self, model: BaseChatModel, search_tool: BaseTool):
        """Initialize MindSearch planner agent"""
        self.llm = model
        self.search_tool = search_tool
        self.max_turn = 10
        
    def forward(self, question: str) -> dict:
        """Main execution loop for MindSearch"""
        
        # Initialize graph
        graph = WebSearchGraph(self.llm, self.search_tool)
        local_dict = {'graph': graph}
        global_dict = {'WebSearchGraph': WebSearchGraph}
        
        # Initial message
        system_prompt = GRAPH_PROMPT_EN + "\n" + GRAPH_FEWSHOT_EXAMPLE_EN
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User Question: {question}")
        ]
        
        for turn in range(self.max_turn):
            # Phase 1: Data Aggregation if we have search results
            if turn > 0:
                reference_text = self._generate_references_from_graph(graph.nodes)
                if reference_text:
                    messages.append(
                        HumanMessage(
                            content=f"Current search results:\n{reference_text}"
                        )
                    )
            
            # Phase 2: Planner invocation
            response = self.llm.invoke(messages)
            planner_output = response.content
            # Handle list responses
            if isinstance(planner_output, list):
                planner_output = planner_output[0] if planner_output else ""
            
            messages.append(AIMessage(content=planner_output))
            
            # Extract and execute code
            code = self._extract_code(planner_output)
            
            if not code:
                break
                
            # Phase 3: Execute code
            try:
                exec(code, global_dict, local_dict)
                
                # Wait for all search tasks to complete
                graph.wait_for_all_searches()
                
                # Check if we should terminate (response node added)
                if "add_response_node" in code:
                    break
                    
            except Exception as e:
                print(f"Error executing code: {e}")
                break
        
        # Final summarization
        reference_text = self._generate_references_from_graph(graph.nodes)
        
        # Check if we have any search results
        if not reference_text or reference_text.strip() == "":
            return {
                "article": "No search results were generated. The question may have been answered immediately or an error occurred during search execution.",
                "search_count": 0
            }
        
        final_messages = [
            SystemMessage(content=FINAL_RESPONSE_EN),
            HumanMessage(content=reference_text)
        ]
        
        final_response = self.llm.invoke(final_messages)
        
        # Count searches
        search_count = sum(
            1 for name, data in graph.nodes.items()
            if data.get('type') == 'searcher' and 'response' in data
        )
        
        return {
            "article": final_response.content,
            "search_count": search_count
        }
        
    def _extract_code(self, text) -> str:
        """Extract Python code from LLM output"""
        # Handle list responses
        if isinstance(text, list):
            text = text[0] if text else ""
        
        # Ensure it's a string
        if not isinstance(text, str):
            text = str(text)
        
        # Remove import statements and graph creation
        text = re.sub(r"from ([\w.]+) import WebSearchGraph", "", text)
        text = re.sub(r"graph\s*=\s*WebSearchGraph\(\)", "", text)
        
        # Try to extract code between action tags first (more specific)
        action_match = re.search(
            r'<\|action_start\|><\|interpreter\|>\s*```python\s*\n(.+?)```\s*<\|action_end\|>',
            text,
            re.DOTALL
        )
        if action_match:
            return action_match.group(1).strip()
        
        # Try without action_end tag
        action_match2 = re.search(
            r'<\|action_start\|><\|interpreter\|>\s*```python\s*\n(.+?)```',
            text,
            re.DOTALL
        )
        if action_match2:
            return action_match2.group(1).strip()
            
        # Try to extract code between triple backticks
        triple_match = re.search(r"```python\s*\n(.+?)```", text, re.DOTALL)
        if triple_match:
            return triple_match.group(1).strip()
            
        return ""
        
    def _generate_references_from_graph(self, nodes: Dict[str, dict]) -> str:
        """Generate formatted references from graph nodes"""
        references = []
        citation_counter = 0
        
        for name, data in nodes.items():
            if name in ["root", "response"]:
                continue
                
            if "response" not in data:
                continue
                
            # Update citation indices
            content = data["response"]["content"]
            citations = data["response"].get("citations", {})
            
            # Renumber citations
            updated_content = content
            for old_idx in sorted(citations.keys(), key=int, reverse=True):
                citation_counter += 1
                updated_content = updated_content.replace(
                    f"[[{old_idx}]]",
                    "" # f"[[{citation_counter}]]"
                )
                
            references.append(f"## {data['content']}\n\n{updated_content}")
            
        return "\n\n".join(references)


# ============ Framework Interface ============

def get_chain(model: BaseChatModel, search_tool: BaseTool) -> Runnable:
    """
    Constructs and returns the MindSearch chain for the framework.
    
    Args:
        model: The LangChain chat model to use
        search_tool: The search tool to use
        
    Returns:
        A Runnable that accepts {"topic": str} and returns 
        {"final_output": {"article": str, "metadata": {"search_count": int}}}
    """
    
    def run_mindsearch(input_dict: dict) -> dict:
        """Run MindSearch and structure the output"""
        agent = MindSearchAgent(model, search_tool)
        result = agent.forward(input_dict["topic"])
        
        # Structure output to match framework expectations
        return {
            "final_output": {
                "article": result["article"],
                "metadata": {
                    "search_count": result["search_count"]
                }
            }
        }
    
    return RunnableLambda(run_mindsearch) 