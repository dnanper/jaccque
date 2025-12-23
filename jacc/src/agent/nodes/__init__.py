"""Nodes package - Modular workflow components.

Each node is independently testable and performs a single responsibility:
- ThinkNode: LLM reasoning and action generation
- ActNode: Command execution in environment  
- ObserveNode: Output formatting and observation
- DecideNode: Flow control and routing

Usage:
    from agent.nodes import ThinkNode, ActNode, ObserveNode, DecideNode
    
    think = ThinkNode(config=config, model=model)
    act = ActNode(config=config, environment=env)
    observe = ObserveNode(config=config)
    decide = DecideNode(config=config)
"""

from .base import (
    BaseNode,
    ConfigurableNode,
    ModelNode,
    EnvironmentNode,
)
from .think import ThinkNode
from .act import ActNode
from .observe import ObserveNode
from .decide import DecideNode, should_continue, CONTINUE, FINISH

__all__ = [
    # Base classes
    "BaseNode",
    "ConfigurableNode",
    "ModelNode",
    "EnvironmentNode",
    
    # Concrete nodes
    "ThinkNode",
    "ActNode",
    "ObserveNode",
    "DecideNode",
    
    # Routing utilities
    "should_continue",
    "CONTINUE",
    "FINISH",
]
