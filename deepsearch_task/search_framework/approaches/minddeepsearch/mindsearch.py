"""
MindSearch implementation using LangChain, Gemini-2.5-flash, and Tavily API
"""

import json
import re
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import BaseMessage


# ============ Prompts (exact from original) ============

SEARCHER_SYSTEM_PROMPT_EN = """## Character Introduction
You are an intelligent assistant that can call web search tools. Please collect information and reply to the question based on the current problem. You can use the following tools:
{tool_info}
## Reply Format

When calling the tool, please follow the format below:
```
Your thought process...<|action_start|><|plugin|>{{"name": "tool_name", "parameters": {{"param1": "value1"}}}}<|action_end|>
```

## Requirements

- Each key point in the response should be marked with the source of the search results to ensure the credibility of the information. The citation format is `[[int]]`. If there are multiple citations, use multiple [[]] to provide the index, such as `[[id_1]][[id_2]]`.
- Based on the search results of the "current problem", write a detailed and complete reply to answer the "current problem".
"""

FEWSHOT_EXAMPLE_EN = """
## Example

### search
When I want to search for "What season is Honor of Kings now", I will operate in the following format:
Now it is 2024, so I should search for the keyword of the Honor of Kings<|action_start|><|plugin|>{{"name": "FastWebBrowser.search", "parameters": {{"query": ["Honor of Kings Season", "season for Honor of Kings in 2024"]}}}}<|action_end|>

### select
In order to find the strongest shooters in Honor of Kings in season s36, I needed to look for web pages that mentioned shooters in Honor of Kings in season s36. After an initial browse of the web pages, I found that web page 0 mentions information about Honor of Kings in s36 season, but there is no specific mention of information about the shooter. Webpage 3 mentions that "the strongest shooter in s36 has appeared?", which may contain information about the strongest shooter. Webpage 13 mentions "Four T0 heroes rise, archer's glory", which may contain information about the strongest archer. Therefore, I chose webpages 3 and 13 for further reading.<|action_start|><|plugin|>{{"name": "FastWebBrowser.select", "parameters": {{"index": [3, 13]}}}}<|action_end|>
"""

SEARCHER_INPUT_TEMPLATE_EN = """## Final Problem
{topic}
## Current Problem
{question}
"""

SEARCHER_CONTEXT_TEMPLATE_EN = """## Historical Problem
{question}
Answer: {answer}
"""

GRAPH_PROMPT_EN = """## Character Profile
You are a programmer capable of Python programming in a Jupyter environment. You can utilize the provided API to construct a Web Search Graph, ultimately generating and executing code.

## API Description

Below is the API documentation for the WebSearchGraph class, including detailed attribute descriptions:

### Class: WebSearchGraph

This class manages nodes and edges of a web search graph and conducts searches via a web proxy.

#### Initialization Method

Initializes an instance of WebSearchGraph.

**Attributes:**

- nodes (Dict[str, Dict[str, str]]): A dictionary storing all nodes in the graph. Each node is indexed by its name and contains content, type, and other related information.
- adjacency_list (Dict[str, List[str]]): An adjacency list storing the connections between all nodes in the graph. Each node is indexed by its name and contains a list of adjacent node names.

#### Method: add_root_node

Adds the initial question as the root node.
**Parameters:**

- node_content (str): The user's question.
- node_name (str, optional): The node name, default is 'root'.

#### Method: add_node

Adds a sub-question node and returns search results.
**Parameters:**

- node_name (str): The node name.
- node_content (str): The sub-question content.

**Returns:**

- str: Returns the search results.

#### Method: add_response_node

Adds a response node when the current information satisfies the question's requirements.

**Parameters:**

- node_name (str, optional): The node name, default is 'response'.

#### Method: add_edge

Adds an edge.

**Parameters:**

- start_node (str): The starting node name.
- end_node (str): The ending node name.

#### Method: reset

Resets nodes and edges.

#### Method: node

Get node information.

python
def node(self, node_name: str) -> str

**Parameters:**

- node_name (str): The node name.

**Returns:**

- str: Returns a dictionary containing the node's information, including content, type, thought process (if any), and list of predecessor nodes.

## Task Description
By breaking down a question into sub-questions that can be answered through searches (unrelated questions can be searched concurrently), each search query should be a single question focusing on a specific person, event, object, specific time point, location, or knowledge point. It should not be a compound question (e.g., a time period). Step by step, build the search graph to finally answer the question.

## Considerations

1. Each search node's content must be a single question; do not include multiple questions (e.g., do not ask multiple knowledge points or compare and filter multiple things simultaneously, like asking for differences between A, B, and C, or price ranges -> query each separately).
2. Do not fabricate search results; wait for the code to return results.
3. Do not repeat the same question; continue asking based on existing questions.
4. When adding a response node, add it separately; do not add a response node and other nodes simultaneously.
5. In a single output, do not include multiple code blocks; only one code block per output.
6. Each code block should be placed within a code block marker, and after generating the code, add an <|action_end|> tag as shown below:
    <|action_start|><|interpreter|>
    ```python
    # Your code block (Note that the 'Get new added node information' logic must be added at the end of the code block, such as 'graph.node('...')')
    ```<|action_end|>
7. The final response should add a response node with node_name 'response', and no other nodes should be added.
"""

GRAPH_FEWSHOT_EXAMPLE_EN = """
## Response Format
**IMPORTANT**: The 'graph' variable is already initialized. Do NOT create a new instance!

<|action_start|><|interpreter|>```python
# NOTE: graph is already available, do NOT use: graph = WebSearchGraph()
graph.add_root_node(node_content="Which large model API is the cheapest?", node_name="root") # Add the original question as the root node
graph.add_node(
        node_name="Large Model API Providers", # The node name should be meaningful
        node_content="Who are the main large model API providers currently?")
graph.add_node(
        node_name="sub_name_2", # The node name should be meaningful
        node_content="content of sub_name_2")
...
graph.add_edge(start_node="root", end_node="sub_name_1")
...
# Get node info
graph.node("Large Model API Providers"), graph.node("sub_name_2"), ...
```<|action_end|>
"""

FINAL_RESPONSE_EN = """Based on the provided Q&A pairs, write a detailed and comprehensive final response.
- The response content should be logically clear and well-structured to ensure reader understanding.
- Each key point in the response should be marked with the source of the search results (consistent with the indices in the Q&A pairs) to ensure information credibility. The index is in the form of `[[int]]`, and if there are multiple indices, use multiple `[[]]`, such as `[[id_1]][[id_2]]`.
- The response should be comprehensive and complete, without vague expressions like "based on the above content". The final response should not include the Q&A pairs provided to you.
- The language style should be professional and rigorous, avoiding colloquial expressions.
- Maintain consistent grammar and vocabulary usage to ensure overall document consistency and coherence."""


# ============ Data Classes ============

@dataclass
class AgentMessage:
    sender: str
    content: str
    formatted: dict = field(default_factory=dict)
    stream_state: Optional[str] = None
    
    def model_dump(self):
        return {
            'sender': self.sender,
            'content': self.content,
            'formatted': self.formatted,
            'stream_state': self.stream_state
        }


# ============ WebSearchGraph Implementation ============

class WebSearchGraph:
    def __init__(self, api_key: str = None):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.adjacency_list: Dict[str, List[dict]] = defaultdict(list)
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.futures = []
        self.api_key = api_key
        
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
        searcher = SearcherAgent(api_key=self.api_key)
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
            print("No search tasks to wait for")
            return
            
        print(f"Waiting for {len(self.futures)} search task(s) to complete...")
        for future in as_completed(self.futures):
            try:
                node_name = future.result(timeout=120)  # Increased timeout to 120s
                print(f"[OK] Search completed for node: {node_name}")
            except Exception as e:
                print(f"[ERROR] Search failed: {e}")
                import traceback
                traceback.print_exc()
        self.futures.clear()
        print("All search tasks completed")
        
    def reset(self):
        """Reset the graph"""
        self.nodes = {}
        self.adjacency_list = defaultdict(list)
        self.futures = []


# ============ Searcher Agent Implementation ============

class SearcherAgent:
    def __init__(self, api_key: str = None):
        """Initialize searcher with Gemini and Tavily"""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=api_key,
            temperature=0.1
        )
        self.search_tool = TavilySearchResults(max_results=5)
        
    def search(self, question: str, topic: str, history: List[dict]) -> dict:
        """Execute search for a specific question"""
        
        print(f"\n  [SEARCHER] Starting search for: {question[:80]}...")
        print(f"  [SEARCHER] Topic: {topic[:80]}...")
        print(f"  [SEARCHER] History items: {len(history)}")
        
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
        max_iterations = 5
        search_results = {}
        final_response = ""
        
        for i in range(max_iterations):
            print(f"  [SEARCHER] ReAct iteration {i + 1}/{max_iterations}")
            response = self.llm.invoke(messages)
            content = response.content
            print(f"  [SEARCHER] LLM response length: {len(content)} chars")
            
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
                            
                            print(f"  [SEARCHER] Executing {len(queries)} web search(es)")
                            for q in queries:
                                print(f"    - {q}")
                                
                            # Execute searches
                            all_results = []
                            for idx, query in enumerate(queries):
                                results = self.search_tool.invoke(query)
                                print(f"  [SEARCHER] Query '{query[:50]}...' returned {len(results)} results")
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
                        print(f"Failed to parse action JSON: {action_json}")
                        break
            else:
                # No more actions, this is the final response
                final_response = content
                break
        
        # Ensure we have a valid response
        if not final_response or final_response.strip() == "":
            final_response = f"Unable to generate a comprehensive answer for: {question}. Search may have failed or returned insufficient results."
        
        print(f"  [SEARCHER] [OK] Search complete")
        print(f"  [SEARCHER] Final response: {len(final_response)} chars")
        print(f"  [SEARCHER] Citations collected: {len(search_results)}")
                
        return {
            'content': final_response,
            'citations': search_results,
            'memory': {'messages': messages}
        }


# ============ MindSearch Planner Agent ============

class MindSearchAgent:
    def __init__(self, api_key: str = None):
        """Initialize MindSearch planner agent"""
        self.api_key = api_key
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=api_key,
            temperature=0.1
        )
        self.max_turn = 10
        
    def forward(self, question: str) -> str:
        """Main execution loop for MindSearch"""
        
        print(f"\n{'='*70}")
        print(f"STARTING MINDSEARCH")
        print(f"Question: {question}")
        print(f"{'='*70}\n")
        
        # Initialize graph with API key
        graph = WebSearchGraph(api_key=self.api_key)
        local_dict = {'graph': graph}
        global_dict = {'WebSearchGraph': WebSearchGraph}
        
        # Initial message
        system_prompt = GRAPH_PROMPT_EN + "\n" + GRAPH_FEWSHOT_EXAMPLE_EN
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User Question: {question}")
        ]
        
        print(f"\n[TURN 0] Initial Messages to Planner:")
        print(f"  System Prompt Length: {len(system_prompt)} chars")
        print(f"  User Question: {question}")
        print(f"-" * 70)
        
        for turn in range(self.max_turn):
            print(f"\n{'#'*70}")
            print(f"[TURN {turn + 1}] Starting iteration")
            print(f"{'#'*70}")
            
            # Phase 1: Data Aggregation if we have search results
            if turn > 0:
                print(f"\n[TURN {turn + 1}] Phase 1: Aggregating search results...")
                reference_text = self._generate_references_from_graph(graph.nodes)
                if reference_text:
                    print(f"  Generated reference text: {len(reference_text)} chars")
                    print(f"  Preview: {reference_text[:200]}...")
                    messages.append(
                        HumanMessage(
                            content=f"Current search results:\n{reference_text}"
                        )
                    )
                else:
                    print(f"  No reference text generated (no completed searches)")
            
            # Phase 2: Planner invocation
            print(f"\n[TURN {turn + 1}] Phase 2: Invoking Planner LLM...")
            print(f"  Total messages in context: {len(messages)}")
            
            response = self.llm.invoke(messages)
            planner_output = response.content
            
            # Handle list responses
            if isinstance(planner_output, list):
                print(f"\n[TURN {turn + 1}] Planner returned a list with {len(planner_output)} items")
                planner_output_str = planner_output[0] if planner_output else ""
            else:
                planner_output_str = planner_output
            
            print(f"\n[TURN {turn + 1}] Planner Output:")
            print(f"  Type: {type(planner_output)}")
            print(f"  Length: {len(planner_output_str)} chars")
            print(f"  Content:\n{planner_output_str[:1000]}...")
            print(f"-" * 70)
            
            # Use the string version for further processing
            planner_output = planner_output_str
            
            messages.append(AIMessage(content=planner_output))
            
            # Extract and execute code
            print(f"\n[TURN {turn + 1}] Phase 3: Extracting code...")
            code = self._extract_code(planner_output)
            
            if not code:
                print(f"\n{'!'*70}")
                print(f"ERROR: No code found in planner output")
                print(f"Tried multiple extraction patterns but none matched")
                print(f"{'!'*70}\n")
                break
            
            print(f"  Extracted code ({len(code)} chars):")
            print(f"  {'-' * 66}")
            for line in code.split('\n'):
                print(f"  {line}")
            print(f"  {'-' * 66}")
                
            # Phase 3: Execute code
            try:
                print(f"\n[TURN {turn + 1}] Executing code...")
                exec(code, global_dict, local_dict)
                print(f"  [OK] Code executed successfully")
                
                # Wait for all search tasks to complete
                graph.wait_for_all_searches()
                
                # Check if we should terminate (response node added)
                if "add_response_node" in code:
                    print(f"\n[TURN {turn + 1}] Response node added, terminating search phase")
                    break
                    
            except Exception as e:
                print(f"Error executing code at turn {turn}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Debug: Show graph state
        print(f"\n{'='*70}")
        print(f"SEARCH PHASE COMPLETE")
        print(f"{'='*70}")
        print(f"Graph contains {len(graph.nodes)} nodes:")
        for node_name, node_data in graph.nodes.items():
            has_response = "response" in node_data
            node_type = node_data.get('type', 'unknown')
            print(f"  - {node_name:<30} (type: {node_type:<10}, has_response: {has_response})")
        print(f"{'='*70}\n")
                
        # Final summarization
        print(f"[FINAL] Generating reference text from graph...")
        reference_text = self._generate_references_from_graph(graph.nodes)
        
        # Check if we have any search results
        if not reference_text or reference_text.strip() == "":
            print(f"[FINAL] [WARNING] No reference text generated")
            return "No search results were generated. The question may have been answered immediately or an error occurred during search execution."
        
        print(f"[FINAL] Reference text generated: {len(reference_text)} chars")
        print(f"[FINAL] Invoking final summarization LLM...")
        
        final_messages = [
            SystemMessage(content=FINAL_RESPONSE_EN),
            HumanMessage(content=reference_text)
        ]
        
        final_response = self.llm.invoke(final_messages)
        
        print(f"[FINAL] [OK] Final response generated: {len(final_response.content)} chars")
        print(f"\n{'='*70}")
        print(f"MINDSEARCH COMPLETE")
        print(f"{'='*70}\n")
        
        return final_response.content
        
    def _extract_code(self, text) -> str:
        """Extract Python code from LLM output"""
        # Handle list responses (Gemini sometimes returns lists)
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
                    f"[[{citation_counter}]]"
                )
                
            references.append(f"## {data['content']}\n\n{updated_content}")
            
        return "\n\n".join(references)


# ============ Example Usage ============

def main():
    import os
    import dotenv
    dotenv.load_dotenv()
    # Set your API keys
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
    
    # Initialize MindSearch
    agent = MindSearchAgent(api_key=os.environ["GOOGLE_API_KEY"])
    
    # Ask a question
    question = "From 2020 to 2050, how many elderly people will there be in Japan? What is their consumption potential across various aspects such as clothing, food, housing, and transportation? Based on population projections, elderly consumer willingness, and potential changes in their consumption habits, please produce a market size analysis report for the elderly demographic"
    
    # Get the answer
    answer = agent.forward(question)
    print(answer)


if __name__ == "__main__":
    main()