"""Graph Assembly - LangGraph workflow builder.

Assembles nodes into a complete agent workflow using LangGraph.
Provides factory functions to create configured agent graphs.

Usage:
    from agent.graph import build_agent_graph
    from agent.config import AgentConfig
    from agent.models import get_model
    from agent.environments import get_environment
    from agent.memory import get_memory_client, MemoryConfig
    
    config = AgentConfig.from_yaml("config.yaml")
    model = get_model("api", model_name="gemini/gemini-2.0-flash")
    env = get_environment({"type": "local"})
    
    # With memory
    memory = get_memory_client(MemoryConfig())
    graph = build_agent_graph(config, model, env, memory_client=memory)
    result = graph.invoke(initial_state)
"""

from typing import Any, TYPE_CHECKING
import logging

from langgraph.graph import StateGraph, END

from agent.state import AgentState, create_initial_state
from agent.config import AgentConfig, render_template
from agent.nodes import (
    ThinkNode,
    ActNode,
    ObserveNode,
    DecideNode,
    should_continue,
    CONTINUE,
    FINISH,
)

if TYPE_CHECKING:
    from agent.models.base import BaseModelProvider
    from agent.environments.base import Environment
    from agent.memory.base import BaseMemoryClient

logger = logging.getLogger(__name__)


def build_agent_graph(
    config: AgentConfig,
    model: "BaseModelProvider",
    environment: "Environment",
    memory_client: "BaseMemoryClient | None" = None,
) -> StateGraph:
    """Build the agent workflow graph.
    
    Creates a LangGraph StateGraph with the following flow:
    think -> act -> observe -> decide -> (continue/finish)
    
    Args:
        config: Agent configuration
        model: LLM model provider
        environment: Execution environment
        memory_client: Optional memory client for recall/retain
        
    Returns:
        Compiled LangGraph StateGraph
    """
    # Create nodes (with optional memory integration)
    think = ThinkNode(config=config, model=model, memory_client=memory_client)
    act = ActNode(config=config, environment=environment)
    observe = ObserveNode(config=config, memory_client=memory_client)
    decide = DecideNode(config=config)
    
    # Build graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("think", think)
    graph.add_node("act", act)
    graph.add_node("observe", observe)
    graph.add_node("decide", decide)
    
    # Add edges
    graph.add_edge("think", "act")
    graph.add_edge("act", "observe")
    graph.add_edge("observe", "decide")
    
    # Conditional edge from decide
    graph.add_conditional_edges(
        "decide",
        should_continue,
        {
            CONTINUE: "think",
            FINISH: END,
        }
    )
    
    # Set entry point
    graph.set_entry_point("think")
    
    logger.info(f"Agent graph built (memory={'enabled' if memory_client else 'disabled'})")
    
    return graph.compile()


def run_agent(
    task: str,
    config: AgentConfig,
    model: "BaseModelProvider",
    environment: "Environment",
    memory_client: "BaseMemoryClient | None" = None,
    instance_id: str | None = None,
) -> AgentState:
    """Run agent on a task and return final state.
    
    Convenience function that:
    1. Builds the graph (with optional memory)
    2. Creates initial state with system/task prompts
    3. Runs the agent to completion
    4. Returns final state
    
    Args:
        task: The task/problem description
        config: Agent configuration
        model: LLM model provider
        environment: Execution environment
        memory_client: Optional memory client for experience
        instance_id: Optional instance ID for tracking
        
    Returns:
        Final AgentState after completion
    """
    # Build graph
    graph = build_agent_graph(config, model, environment, memory_client)

    # DockerConfig info: image, cwd, timeout..
    env_vars = environment.get_template_vars()
    # Model info: model_name, temperature...
    model_vars = model.get_template_vars()
    
    # Render prompt template with env/model infor
    system_prompt = render_template(
        config.system_template,
        **env_vars,
        **model_vars,
    )
    
    task_prompt = render_template(
        config.instance_template,
        task=task,
        **env_vars,
    )
    
    # Create initial state
    initial_state = create_initial_state(
        system_message=system_prompt,
        task_message=task_prompt,
    )
    
    # Add instance_id to working memory for tracking
    if instance_id:
        initial_state["working_memory"]["instance_id"] = instance_id
    
    logger.info(f"Running agent on task: {task[:100]}...")
    
    # Run graph
    # Each agent step = 4 nodes (think→act→observe→decide)
    # Set recursion_limit accordingly with buffer
    recursion_limit = config.step_limit * 4 + 10
    final_state = graph.invoke(initial_state, {"recursion_limit": recursion_limit})
    
    logger.info(
        f"Agent finished: status={final_state.get('exit_status')}, "
        f"steps={final_state.get('step_count')}, "
        f"cost=${final_state.get('total_cost', 0):.4f}"
    )
    
    return final_state


async def run_agent_async(
    task: str,
    config: AgentConfig,
    model: "BaseModelProvider",
    environment: "Environment",
    memory_client: "BaseMemoryClient | None" = None,
    instance_id: str | None = None,
) -> AgentState:
    """Run agent on a task asynchronously.
    
    Same as run_agent but async-compatible.
    Use this when you need to await memory operations properly.
    
    Args:
        task: The task/problem description
        config: Agent configuration
        model: LLM model provider
        environment: Execution environment
        memory_client: Optional memory client for experience
        instance_id: Optional instance ID for tracking
        
    Returns:
        Final AgentState after completion
    """
    # Initialize memory client if provided
    if memory_client and not memory_client.is_initialized:
        await memory_client.initialize()
    
    # Run the sync agent (graph handles async internally)
    return run_agent(
        task=task,
        config=config,
        model=model,
        environment=environment,
        memory_client=memory_client,
        instance_id=instance_id,
    )


__all__ = [
    "build_agent_graph",
    "run_agent",
    "run_agent_async",
]
