"""Think Node - LLM reasoning and action generation.

This node:
1. Builds the prompt from conversation history
2. Queries the LLM model
3. Parses the response to extract action
4. Updates state with action and response
"""

import re
import time
from typing import Any, TYPE_CHECKING

from .base import ModelNode

if TYPE_CHECKING:
    from agent.state import AgentState
    from agent.config import AgentConfig


class ThinkNode(ModelNode):
    """LLM reasoning node - generates actions from context.
    
    Queries the model with conversation history and parses
    the response to extract executable actions.
    """
    
    name = "think"
    
    def __call__(self, state: "AgentState") -> dict[str, Any]:
        """Query LLM and parse action from response.
        
        Args:
            state: Current agent state with messages
            
        Returns:
            State updates including assistant message and parsed action
        """
        self.logger.debug(f"Step {state['step_count']}: Querying model")
        
        # Build messages for model
        messages = self._prepare_messages(state)
        
        # Query model
        response = self.model.query(messages)
        content = response.get("content", "")
        cost = response.get("extra", {}).get("cost", 0.0)
        
        self.logger.debug(f"Model response: {content[:200]}...")
        
        # Parse action from response
        action, action_type = self._parse_action(content)
        
        # Build state updates
        updates: dict[str, Any] = {
            "messages": [{
                "role": "assistant",
                "content": content,
                "timestamp": time.time(),
            }],
            "current_action": action,
            "action_type": action_type,
            "total_cost": state["total_cost"] + cost,
        }
        
        # Store parsing info in working memory
        updates["working_memory"] = {
            **state.get("working_memory", {}),
            "last_response": content,
            "last_cost": cost,
        }
        
        return updates
    
    def _prepare_messages(self, state: "AgentState") -> list[dict[str, str]]:
        """Prepare messages for model query.
        
        Converts state messages to model format.
        """
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in state["messages"]
        ]
    
    def _parse_action(self, content: str) -> tuple[str | None, str | None]:
        """Parse action from LLM response.
        
        Args:
            content: LLM response text
            
        Returns:
            Tuple of (action_string, action_type)
        """
        # Check for task completion
        if re.search(self.config.submit_pattern, content, re.IGNORECASE):
            return content, "submit"
        
        # Try to extract bash command
        bash_match = re.search(
            self.config.action_regex,
            content,
            re.DOTALL | re.IGNORECASE
        )
        
        if bash_match:
            action = bash_match.group(1).strip()
            return action, "bash"
        
        # No valid action found
        self.logger.warning("No valid action found in response")
        return None, "invalid"
    
    def validate_input(self, state: "AgentState") -> bool:
        """Validate that we have messages to work with."""
        return len(state.get("messages", [])) > 0
