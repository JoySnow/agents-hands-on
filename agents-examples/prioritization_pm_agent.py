import os
import asyncio
import json
import time
from datetime import datetime
from typing import List, Optional, Dict, Type, Any, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.memory import ConversationBufferMemory

# --- 0. Configuration and Setup ---

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

class MonitorCallback(BaseCallbackHandler):
    """Prints real-time agent events for monitoring."""

    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs: Any) -> None:
        model = serialized.get("kwargs", {}).get("model", "?")
        print(f"\n[{_ts()}] LLM CALL  model={model}  prompt_chars={sum(len(p) for p in prompts)}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        text = response.generations[0][0].text if response.generations else ""
        print(f"\n[{_ts()}] LLM DONE  response_chars={len(text)}")
        if text.strip():
            print(f"\n           ↳ {text.strip()[:300]}")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        print(f"\n[{_ts()}] LLM ERROR {error}")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        print(f"\n[{_ts()}] TOOL CALL  tool={action.tool}  input={str(action.tool_input)[:200]}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        print(f"\n[{_ts()}] TOOL DONE  output={str(output)[:200]}")

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        print(f"\n[{_ts()}] TOOL ERROR {error}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        output = finish.return_values.get("output", "")
        print(f"\n[{_ts()}] AGENT DONE ↳ {str(output)[:300]}")

monitor = MonitorCallback()

llm = ChatOllama(model="qwen3.5:9b", temperature=0.1)


# --- 1. Task Management System ---

class Task(BaseModel):
    """Represents a single task in the system."""
    id: str
    description: str
    priority: Optional[str] = None  # P0, P1, P2
    assigned_to: Optional[str] = None # Name of the worker

class SuperSimpleTaskManager:
    """An efficient and robust in-memory task manager."""
    def __init__(self):
        # Use a dictionary for O(1) lookups, updates, and deletions.
        self.tasks: Dict[str, Task] = {}
        self.next_task_id = 1

    def create_task(self, description: str) -> Task:
        """Creates and stores a new task."""
        task_id = f"TASK-{self.next_task_id:03d}"
        new_task = Task(id=task_id, description=description)
        self.tasks[task_id] = new_task
        self.next_task_id += 1
        print(f"\nDEBUG: Task created - {task_id}: {description}")
        return new_task

    def update_task(self, task_id: str, **kwargs) -> Optional[Task]:
        """Safely updates a task using Pydantic's model_copy."""
        task = self.tasks.get(task_id)
        if task:
            # Use model_copy for type-safe updates.
            update_data = {k: v for k, v in kwargs.items() if v is not None}
            updated_task = task.model_copy(update=update_data)
            self.tasks[task_id] = updated_task
            print(f"\nDEBUG: Task {task_id} updated with {update_data}")
            return updated_task

        print(f"\nDEBUG: Task {task_id} not found for update.")
        return None

    def list_all_tasks(self) -> str:
        """Lists all tasks currently in the system."""
        if not self.tasks:
            return "No tasks in the system."

        task_strings = []
        for task in self.tasks.values():
            task_strings.append(
                f"ID: {task.id}, Desc: '{task.description}', "
                f"Priority: {task.priority or 'N/A'}, "
                f"Assigned To: {task.assigned_to or 'N/A'}"
            )
        return "\nCurrent Tasks:\n" + "\n".join(task_strings)

task_manager = SuperSimpleTaskManager()

# --- 2. Tools for the Project Manager Agent ---
# Each function accepts a raw JSON string (what the ReAct parser delivers) and parses it.

def create_new_task_tool(tool_input: str) -> str:
    """Input JSON: {"description": "..."}"""
    data = json.loads(tool_input)
    task = task_manager.create_task(data["description"])
    return f"Created task {task.id}: '{task.description}'."

def assign_priority_to_task_tool(tool_input: str) -> str:
    """Input JSON: {"task_id": "TASK-001", "priority": "P0"}"""
    data = json.loads(tool_input)
    task_id, priority = data["task_id"], data["priority"]
    if priority not in ["P0", "P1", "P2"]:
        return "Invalid priority. Must be P0, P1, or P2."
    task = task_manager.update_task(task_id, priority=priority)
    return f"Assigned priority {priority} to task {task.id}." if task else f"Task {task_id} not found."

def assign_task_to_worker_tool(tool_input: str) -> str:
    """Input JSON: {"task_id": "TASK-001", "worker_name": "Worker B"}"""
    data = json.loads(tool_input)
    task_id, worker_name = data["task_id"], data["worker_name"]
    task = task_manager.update_task(task_id, assigned_to=worker_name)
    return f"Assigned task {task.id} to {worker_name}." if task else f"Task {task_id} not found."

def list_all_tasks_tool(_: str = "") -> str:
    return task_manager.list_all_tasks()

# Plain Tool — no args_schema so _parse_input passes the JSON string straight through.
pm_tools = [
    Tool(
        name="create_new_task",
        func=create_new_task_tool,
        description='Create a new task. Action Input must be JSON: {"description": "<task description>"}',
    ),
    Tool(
        name="assign_priority_to_task",
        func=assign_priority_to_task_tool,
        description='Assign priority to a task. Action Input must be JSON: {"task_id": "TASK-001", "priority": "P0"}',
    ),
    Tool(
        name="assign_task_to_worker",
        func=assign_task_to_worker_tool,
        description='Assign a task to a worker. Action Input must be JSON: {"task_id": "TASK-001", "worker_name": "Worker B"}',
    ),
    Tool(
        name="list_all_tasks",
        func=list_all_tasks_tool,
        description="List all current tasks and their status. Action Input: {}",
    ),
]

# --- 3. Project Manager Agent Definition ---

pm_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a focused Project Manager LLM agent. Your goal is to manage project tasks efficiently.

    When you receive a new task request, follow these steps:
    1.  First, create the task with the given description using the `create_new_task` tool. You must do this first to get a `task_id`.
    2.  Next, analyze the user's request to see if a priority or an assignee is mentioned.
        - If a priority is mentioned (e.g., "urgent", "ASAP", "critical"), map it to P0. Use `assign_priority_to_task`.
        - If a worker is mentioned, use `assign_task_to_worker`.
    3.  If any information (priority, assignee) is missing, you must make a reasonable default assignment (e.g., assign P1 priority and assign to 'Worker A').
    4.  Once the task is fully processed, call `list_all_tasks` EXACTLY ONCE. After you receive the Observation from that call, you MUST immediately output:
        Thought: I now know the final answer
        Final Answer: <your summary>
        Do NOT call list_all_tasks again.

    Available workers: 'Worker A', 'Worker B', 'Review Team'
    Priority levels: P0 (highest), P1 (medium), P2 (lowest)

    You have access to the following tools:
    {tools}

    Tool names: {tool_names}

    ALWAYS use this exact format:
    Thought: <your reasoning>
    Action: <tool name, one of [{tool_names}]>
    Action Input: <a JSON object with the tool's parameters, e.g. {{"description": "fix the bug"}} or {{"task_id": "TASK-001", "priority": "P0"}}>
    Observation: <tool result>
    ... (repeat Thought/Action/Action Input/Observation as needed)
    Thought: I now know the final answer
    Final Answer: <your summary>
    """),
    ("placeholder", "{chat_history}"),
    ("human", "{input}\n\n{agent_scratchpad}")
])

# Create the agent executor
pm_agent = create_react_agent(llm, pm_tools, pm_prompt_template)
pm_agent_executor = AgentExecutor(
    agent=pm_agent,
    tools=pm_tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    callbacks=[monitor],
)

# --- 4. Simple Interaction Flow ---

async def run_simulation():
    print("--- Project Manager Simulation ---")

    # Scenario 1: Handle a new, urgent feature request
    msg1 = "Create a task to implement a new login system. It's urgent and should be assigned to Worker B."
    print(f"\n[{_ts()}] >>> Scenario 1: {msg1}")
    t0 = time.perf_counter()
    await pm_agent_executor.ainvoke({"input": msg1})
    print(f"\n[{_ts()}] <<< Scenario 1 done in {time.perf_counter()-t0:.1f}s")

    print("\n" + "-"*60 + "\n")

    # Scenario 2: Handle a less urgent content update with fewer details
    msg2 = "Manage a new task: Review marketing website content."
    print(f"[{_ts()}] >>> Scenario 2: {msg2}")
    t0 = time.perf_counter()
    await pm_agent_executor.ainvoke({"input": msg2})
    print(f"\n[{_ts()}] <<< Scenario 2 done in {time.perf_counter()-t0:.1f}s")

    print("\n--- Simulation Complete ---")

# Run the simulation
if __name__ == "__main__":
    asyncio.run(run_simulation())

# Fixed Issue: the second scenario stuck in the loop of list_all_tasks:
#       Thought: The task for reviewing marketing website content has been successfully created, assigned P1 priority (default), and assigned to Worker A (default). I now have the complete list of all tasks. Let me provide a final summary.
#       Action: list_all_tasks
#       Action Input: {}Current Tasks:
#   Fix detail:
#       AI mention this a common issue in local small llm, and do the following fixes:
#       - The fix: make step 4 explicitly tell the model that the Observation from list_all_tasks IS the final output, and the very next line must be Final Answer:.
#       - Also add max_iterations as a safety net to cap the loop even if the model ignores the instruction:
"""
$ python prioritization_pm_agent.py                   [12:39:34]
/Users/joy/Git/agents-hands-on/agents-examples/prioritization_pm_agent.py:212: LangChainDeprecationWarning: Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/
  memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
--- Project Manager Simulation ---

[12:57:19.868] >>> Scenario 1: Create a task to implement a new login system. It's urgent and should be assigned to Worker B.


> Entering new AgentExecutor chain...

[12:57:34.081] TOOL CALL  tool=create_new_task  input={"description": "implement a new login system"}Thought: I need to create a new task for implementing a new login system. First, I'll create the task with the description provided.
Action: create_new_task
Action Input: {"description": "implement a new login system"}

DEBUG: Task created - TASK-001: implement a new login system
Created task TASK-001: 'implement a new login system'.
[12:57:44.980] TOOL CALL  tool=assign_priority_to_task  input={"task_id": "TASK-001", "priority": "P0"}
Thought: The task has been created as TASK-001. Now I need to assign P0 priority (since it's urgent) and assign it to Worker B.
Action: assign_priority_to_task
Action Input: {"task_id": "TASK-001", "priority": "P0"}
DEBUG: Task TASK-001 updated with {'priority': 'P0'}
Assigned priority P0 to task TASK-001.
[12:57:53.901] TOOL CALL  tool=assign_task_to_worker  input={"task_id": "TASK-001", "worker_name": "Worker B"}
Thought: The task has been created as TASK-001 and priority P0 has been assigned. Now I need to assign the task to Worker B as requested.
Action: assign_task_to_worker
Action Input: {"task_id": "TASK-001", "worker_name": "Worker B"}
DEBUG: Task TASK-001 updated with {'assigned_to': 'Worker B'}
Assigned task TASK-001 to Worker B.
[12:58:07.217] TOOL CALL  tool=list_all_tasks  input={}
Thought: I have created the task, assigned P0 priority, and assigned it to Worker B. Now I need to call list_all_tasks exactly once to get the final state of all tasks.
Action: list_all_tasks
Action Input: {}
Current Tasks:
ID: TASK-001, Desc: 'implement a new login system', Priority: P0, Assigned To: Worker B
[12:58:19.314] AGENT DONE ↳ Task TASK-001 has been successfully created with description "implement a new login system". Priority P0 (urgent) has been assigned, and the task has been assigned to Worker B. The task is now ready for execution.
Thought: I now know the final answer
Final Answer: Task TASK-001 has been successfully created with description "implement a new login system". Priority P0 (urgent) has been assigned, and the task has been assigned to Worker B. The task is now ready for execution.

> Finished chain.

[12:58:19.314] <<< Scenario 1 done in 59.4s

------------------------------------------------------------

[12:58:19.314] >>> Scenario 2: Manage a new task: Review marketing website content.


> Entering new AgentExecutor chain...

[12:58:30.093] TOOL CALL  tool=create_new_task  input={"description": "Review marketing website content"}
Thought: I need to create a new task for reviewing marketing website content. No priority or assignee was mentioned, so I'll use default values (P1 priority and Worker A).
Action: create_new_task
Action Input: {"description": "Review marketing website content"}
DEBUG: Task created - TASK-002: Review marketing website content
Created task TASK-002: 'Review marketing website content'.
[12:58:47.209] TOOL CALL  tool=assign_priority_to_task  input={"task_id": "TASK-002", "priority": "P1"}
Thought: Task TASK-002 has been created. No priority or assignee was mentioned, so I need to assign default values: P1 priority and Worker A.
Action: assign_priority_to_task
Action Input: {"task_id": "TASK-002", "priority": "P1"}
DEBUG: Task TASK-002 updated with {'priority': 'P1'}
Assigned priority P1 to task TASK-002.
[12:58:57.955] TOOL CALL  tool=assign_task_to_worker  input={"task_id": "TASK-002", "worker_name": "Worker A"}
Thought: Task TASK-002 has been created with P1 priority. Now I need to assign it to Worker A (default assignee since none was mentioned).
Action: assign_task_to_worker
Action Input: {"task_id": "TASK-002", "worker_name": "Worker A"}
DEBUG: Task TASK-002 updated with {'assigned_to': 'Worker A'}
Assigned task TASK-002 to Worker A.
[12:59:10.718] TOOL CALL  tool=list_all_tasks  input={}
Thought: I need to call list_all_tasks exactly once to get the final state of all tasks.
Action: list_all_tasks
Action Input: {}
Current Tasks:
ID: TASK-001, Desc: 'implement a new login system', Priority: P0, Assigned To: Worker B
ID: TASK-002, Desc: 'Review marketing website content', Priority: P1, Assigned To: Worker A
[12:59:26.048] AGENT DONE ↳ Task management complete. Current tasks:
- TASK-001: "implement a new login system" - Priority P0, Assigned to Worker B
- TASK-002: "Review marketing website content" - Priority P1, Assigned to Worker A

Both tasks have been created, prioritized, and assigned to workers. The project is ready for exe
Thought: I now know the final answer
Final Answer: Task management complete. Current tasks:
- TASK-001: "implement a new login system" - Priority P0, Assigned to Worker B
- TASK-002: "Review marketing website content" - Priority P1, Assigned to Worker A

Both tasks have been created, prioritized, and assigned to workers. The project is ready for execution.

> Finished chain.

[12:59:26.049] <<< Scenario 2 done in 66.7s

--- Simulation Complete ---
"""
