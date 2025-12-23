"""Agent package - Modular SWE Agent.

A modular agent framework with:
- Configurable model providers (API, vLLM, HuggingFace)
- Multiple execution environments (local, Docker)
- Extensible node-based workflow
- Pluggable memory systems

Quick Start:
    from agent import AgentConfig, run_agent
    from agent.models import get_model
    from agent.environments import get_environment
    
    config = AgentConfig.default()
    model = get_model("api", model_name="gemini/gemini-2.0-flash")
    env = get_environment({"type": "local"})
    
    result = run_agent("Fix the bug in main.py", config, model, env)
    print(result["exit_status"])
"""

from agent.state import AgentState, create_initial_state, add_message, Message
from agent.config import AgentConfig, render_template
from agent.graph import build_agent_graph, run_agent

__all__ = [
    # State
    "AgentState",
    "Message",
    "create_initial_state",
    "add_message",
    
    # Config
    "AgentConfig",
    "render_template",
    
    # Graph
    "build_agent_graph",
    "run_agent",
]
