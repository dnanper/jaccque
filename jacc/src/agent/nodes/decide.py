"""Decide Node - Flow control and routing.

This node:
1. Checks completion conditions
2. Checks resource limits (steps, cost)
3. Determines next action in workflow
4. Sets exit status when finished

Returns routing decision for LangGraph conditional edges.
"""

from typing import Any, Literal, TYPE_CHECKING

from .base import ConfigurableNode

if TYPE_CHECKING:
    from agent.state import AgentState
    from agent.config import AgentConfig


# Routing constants
CONTINUE = "continue"
FINISH = "finish"

RoutingDecision = Literal["continue", "finish"]


class DecideNode(ConfigurableNode):
    """Decision/routing node - controls workflow flow.
    
    Checks various conditions and decides whether to:
    - Continue the loop (back to think)
    - Finish execution (exit)
    """
    
    name = "decide"
    
    def __call__(self, state: "AgentState") -> dict[str, Any]:
        """Evaluate conditions and set exit status if needed.
        
        Args:
            state: Current agent state
            
        Returns:
            State updates (exit_status, exit_message if finishing)
        """
        # Already finished?
        if state.get("exit_status"):
            return {}
        
        action_type = state.get("action_type")
        
        # Check for task completion
        if action_type == "submit":
            self.logger.info("Task submitted - finishing")
            return {
                "exit_status": "success",
                "exit_message": "Task completed successfully",
            }
        
        # Check step limit
        if state["step_count"] >= self.config.step_limit:
            self.logger.warning(f"Step limit reached: {self.config.step_limit}")
            return {
                "exit_status": "step_limit",
                "exit_message": f"Reached step limit of {self.config.step_limit}",
            }
        
        # Check cost limit
        if state["total_cost"] >= self.config.cost_limit:
            self.logger.warning(f"Cost limit reached: ${self.config.cost_limit:.2f}")
            return {
                "exit_status": "cost_limit",
                "exit_message": f"Reached cost limit of ${self.config.cost_limit:.2f}",
            }
        
        # Continue execution
        self.logger.debug(
            f"Continuing: step={state['step_count']}, cost=${state['total_cost']:.4f}"
        )
        return {}
    
    def get_routing(self, state: "AgentState") -> RoutingDecision:
        """Get routing decision for LangGraph conditional edge.
        
        This is called by LangGraph to determine the next node.
        
        Args:
            state: Current agent state
            
        Returns:
            "continue" or "finish"
        """
        if state.get("exit_status"):
            return FINISH
        return CONTINUE


def should_continue(state: dict) -> RoutingDecision:
    """Standalone routing function for LangGraph.
    
    Can be used directly as conditional edge function.
    
    Args:
        state: Current agent state dict
        
    Returns:
        "continue" or "finish"
    """
    if state.get("exit_status"):
        return FINISH
    return CONTINUE

