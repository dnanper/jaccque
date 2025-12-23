"""Base Node - Abstract interface for all nodes.

A node uses 2 types of data:
- Static (Config/Knowledge): data that is constant during the node's lifecycle. We get that
data before the node is called.
- Dynamic (State/Model/Environment): data that the node can access and modify during its 
execution.

Design principles:
- Each node is a callable that transforms state
- Nodes are stateless - dependencies injected via constructor
- Independent and testable in isolation
- Returns partial state updates (merged by LangGraph)
"""
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
import logging

# Blind for user
if TYPE_CHECKING:
    from agent.state import AgentState
    from agent.config import AgentConfig

class BaseNode(ABC):
    """Abstract base class for all graph nodes.
    
    Attributes:
        name: Unique identifier for this node
        logger: Logger instance for this node
    """

    name: str = "base"
    def __init__(self):
        self.logger = logging.getLogger(f"node.{self.name}")
    
    @abstractmethod
    def __call__(self, state: "AgentState") -> dict[str, Any]:
        """Process state and return updates.
        
        Args:
            state: Current Agent State

        Returns:
            Dict of updates to merge into state
        """
        pass
    
    def validate_input(self, state: "AgentState") -> bool:
        """Optional validation before processing.
        
        Override to add custom validation logic.
        
        Args:
            state: Current agent state
            
        Returns:
            True if state is valid for this node
        """
        return True
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

# Configurable means that node can access to all the config that we set before
class ConfigurableNode(BaseNode):
    """Node that requires configuration (only prompt template and response parser)."""
    
    def __init__(self, config: "AgentConfig"):
        super().__init__()
        self.config = config

# Node that use Config and Model
class ModelNode(ConfigurableNode):
    """Node that uses an LLM model.
    
    Extends ConfigurableNode with model injection.
    """
    
    def __init__(self, config: "AgentConfig", model: Any):
        super().__init__(config)
        self.model = model

# Node that use Config and Environment
class EnvironmentNode(ConfigurableNode):
    """Node that uses an execution environment.
    
    Extends ConfigurableNode with environment injection.
    """
    
    def __init__(self, config: "AgentConfig", environment: Any):
        super().__init__(config)
        self.environment = environment
