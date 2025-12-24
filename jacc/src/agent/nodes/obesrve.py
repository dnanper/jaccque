"""Observe Node - Process execution output.

This node:
1. Takes the execution result from state
2. Formats it as an observation message
3. Adds observation to conversation history
4. Handles special cases (timeout, format errors)
"""

import time
from typing import Any, TYPE_CHECKING

from .base import ConfigurableNode
from agent.config import render_template

if TYPE_CHECKING:
    from agent.state import AgentState
    from agent.config import AgentConfig


class ObserveNode(ConfigurableNode):
    """Observation processing node - formats execution output.
    
    Converts execution results into formatted observation
    messages that the LLM can understand.
    """
    
    name = "observe"
    
    def __call__(self, state: "AgentState") -> dict[str, Any]:
        """Format execution output as observation message.
        
        Args:
            state: Current agent state with last_output
            
        Returns:
            State updates with observation message
        """
        action_type = state.get("action_type")
        last_output = state.get("last_output")
        
        # Handle different action types
        if action_type == "submit":
            # Task completed, no observation needed
            return {}
        
        if action_type == "invalid":
            # Format error message
            observation = self._format_error(state)
        elif last_output is None:
            # No output to observe
            return {}
        elif last_output.get("timed_out"):
            # Timeout message
            observation = self._format_timeout(state)
        else:
            # Normal execution output
            observation = self._format_observation(state)
        self.logger.debug(f"Observation: {observation[:200]}...")

        # return only output of the current step, Langraph'll add into messages
        # system
        return {
            "messages": [{
                "role": "user",
                "content": observation,
                "timestamp": time.time(),
            }],
        }
    
    def _format_observation(self, state: "AgentState") -> str:
        """Format normal execution output."""
        output = state.get("last_output", {})
        return render_template(
            self.config.action_observation_template,
            output=output,
            action=state.get("current_action", ""),
        )
    
    def _format_error(self, state: "AgentState") -> str:
        """Format parsing error message."""
        return render_template(
            self.config.format_error_template,
            actions=[],  # No valid actions found
            response=state.get("working_memory", {}).get("last_response", ""),
        )
    
    def _format_timeout(self, state: "AgentState") -> str:
        """Format timeout message."""
        output = state.get("last_output", {})
        return render_template(
            self.config.timeout_template,
            action=state.get("current_action", ""),
            output=output.get("output", ""),
        )
