"""Think Node - LLM reasoning and action generation.

This node:
1. Recalls relevant experience from memory (if available)
2. Builds the prompt from conversation history + memory context
3. Queries the LLM model
4. Parses the response to extract action
5. Updates state with action and response
"""

import asyncio
import re
import time
from typing import Any, TYPE_CHECKING

from .base import ModelNode

if TYPE_CHECKING:
    from agent.state import AgentState
    from agent.config import AgentConfig
    from agent.memory.base import BaseMemoryClient


class ThinkNode(ModelNode):
    """LLM reasoning node - generates actions from context.
    
    Queries the model with conversation history and parses
    the response to extract executable actions.
    
    If memory_client is provided, recalls relevant experience
    before each LLM query to enhance reasoning.
    """
    
    name = "think"
    
    def __init__(
        self,
        config: "AgentConfig",
        model: Any,
        memory_client: "BaseMemoryClient | None" = None,
    ):
        """Initialize ThinkNode with optional memory client.
        
        Args:
            config: Agent configuration
            model: LLM model provider
            memory_client: Optional memory client for recall
        """
        super().__init__(config, model)
        self.memory = memory_client
    
    def __call__(self, state: "AgentState") -> dict[str, Any]:
        """Query LLM and parse action from response.
        
        If memory client is available, recalls relevant experience
        before querying and injects it into the context.
        
        Args:
            state: Current agent state with messages
            
        Returns:
            State updates including assistant message and parsed action
        """
        self.logger.debug(f"Step {state['step_count']}: Querying model")
        
        # Recall relevant experience from memory (if available)
        memory_context = ""
        recall_result = None
        if self.memory and self.memory.is_initialized:
            recall_result = self._recall_experience(state)
            if recall_result and recall_result.results:
                memory_context = self._format_memory_context(recall_result.results)
                self.logger.debug(f"Recalled {len(recall_result.results)} facts from memory")
        
        # Build messages for model (with memory context if available)
        messages = self._prepare_messages(state, memory_context)
        
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
            "last_recall_count": len(recall_result.results) if recall_result else 0,
        }
        
        return updates
    
    def _recall_experience(self, state: "AgentState"):
        """Recall relevant experience from memory.
        
        Runs async recall in sync context using persistent event loop.
        
        Args:
            state: Current agent state
            
        Returns:
            RecallResult or None if recall fails
        """
        try:
            from agent.memory.utils import build_recall_query
            from agent.memory.async_utils import run_async
            
            # Build query from current state
            query = build_recall_query(state)
            
            # Run async recall using persistent event loop
            return run_async(self.memory.recall(query), timeout=15)
                
        except Exception as e:
            self.logger.warning(f"Memory recall failed: {e}")
            return None
    
    def _format_memory_context(self, facts: list) -> str:
        """Format recalled facts as context for LLM prompt.
        
        Args:
            facts: List of MemoryFact objects
            
        Returns:
            Formatted context string
        """
        from agent.memory.utils import format_facts_for_prompt
        return format_facts_for_prompt(facts, max_chars=2000)
    
    def _prepare_messages(
        self,
        state: "AgentState",
        memory_context: str = "",
    ) -> list[dict[str, str]]:
        """Prepare messages for model query.
        
        Converts state messages to model format.
        If memory_context is provided, injects it after the system message.
        
        Args:
            state: Current agent state
            memory_context: Optional context from memory recall
            
        Returns:
            List of message dicts for LLM
        """
        messages = []
        
        for msg in state["messages"]:
            role = msg["role"]
            content = msg["content"]
            
            # Inject memory context after system message
            if role == "system" and memory_context:
                content = content + f"""

<relevant_experience>
The following is relevant experience from previous tasks that may help:

{memory_context}
</relevant_experience>
"""
            
            messages.append({"role": role, "content": content})
        
        return messages
    
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
