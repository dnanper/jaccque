"""Graph Assembly - LangGraph workflow builder.

Assembles nodes into a complete agent workflow using LangGraph.
Provides factory functions to create configured agent graphs.

Usage:
    from agent.graph import build_agent_graph
    from agent.config import AgentConfig
    from agent.models import get_model
    from agent.environments import get_environment
    
    config = AgentConfig.from_yaml("config.yaml")
    model = get_model("api", model_name="gemini/gemini-2.0-flash")
    env = get_environment({"type": "local"})
    
    graph = build_agent_graph(config, model, env)
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

logger = logging.getLogger(__name__)


def build_agent_graph(
    config: AgentConfig,
    model: "BaseModelProvider",
    environment: "Environment",
) -> StateGraph:
    """Build the agent workflow graph.
    
    Creates a LangGraph StateGraph with the following flow:
    think -> act -> observe -> decide -> (continue/finish)
    
    Args:
        config: Agent configuration
        model: LLM model provider
        environment: Execution environment
        
    Returns:
        Compiled LangGraph StateGraph
    """
    # Create nodes
    think = ThinkNode(config=config, model=model)
    act = ActNode(config=config, environment=environment)
    observe = ObserveNode(config=config)
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
    
    logger.info("Agent graph built successfully")
    
    return graph.compile()


def run_agent(
    task: str,
    config: AgentConfig,
    model: "BaseModelProvider",
    environment: "Environment",
) -> AgentState:
    """Run agent on a task and return final state.
    
    Convenience function that:
    1. Builds the graph
    2. Creates initial state with system/task prompts
    3. Runs the agent to completion
    4. Returns final state
    
    Args:
        task: The task/problem description
        config: Agent configuration
        model: LLM model provider
        environment: Execution environment
        
    Returns:
        Final AgentState after completion
    """
    # Build graph
    graph = build_agent_graph(config, model, environment)

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
    
    logger.info(f"Running agent on task: {task[:100]}...")
    
    # Run graph
    final_state = graph.invoke(initial_state)
    
    logger.info(
        f"Agent finished: status={final_state.get('exit_status')}, "
        f"steps={final_state.get('step_count')}, "
        f"cost=${final_state.get('total_cost', 0):.4f}"
    )
    
    return final_state


__all__ = [
    "build_agent_graph",
    "run_agent",
]
