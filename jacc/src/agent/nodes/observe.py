"""Observe Node - Process execution output.

This node:
1. Takes the execution result from state
2. Formats it as an observation message
3. Adds observation to conversation history
4. Handles special cases (timeout, format errors)
5. Optionally retains learnings to memory
"""

import asyncio
import time
from typing import Any, TYPE_CHECKING

from .base import ConfigurableNode
from agent.config import render_template

if TYPE_CHECKING:
    from agent.state import AgentState
    from agent.config import AgentConfig
    from agent.memory.base import BaseMemoryClient


class ObserveNode(ConfigurableNode):
    """Observation processing node - formats execution output.
    
    Converts execution results into formatted observation
    messages that the LLM can understand.
    
    If memory_client is provided, selectively retains
    learnings from significant observations.
    """
    
    name = "observe"
    
    def __init__(
        self,
        config: "AgentConfig",
        memory_client: "BaseMemoryClient | None" = None,
    ):
        """Initialize ObserveNode with optional memory client.
        
        Args:
            config: Agent configuration
            memory_client: Optional memory client for retaining learnings
        """
        super().__init__(config)
        self.memory = memory_client
    
    def __call__(self, state: "AgentState") -> dict[str, Any]:
        """Format execution output as observation message.
        
        Also retains significant learnings to memory if available.
        
        Args:
            state: Current agent state with last_output
            
        Returns:
            State updates with observation message
        """
        action_type = state.get("action_type")
        last_output = state.get("last_output")
        
        # Handle different action types
        if action_type == "submit":
            # Task completed - retain task summary
            self._maybe_retain_learning(state, "task_complete")
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
            # Check if this observation should be retained
            self._maybe_retain_learning(state)
        
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
    
    def _maybe_retain_learning(
        self,
        state: "AgentState",
        override_context: str | None = None,
    ) -> None:
        """Possibly retain a learning from current observation.
        
        Uses heuristics to decide if observation is worth retaining:
        - Error patterns and fixes
        - Successful task completions
        - Significant discoveries
        
        Args:
            state: Current agent state
            override_context: Override the context category
        """
        if not self.memory or not self.memory.is_initialized:
            return
        
        try:
            from agent.memory.utils import should_retain, extract_learning
            
            # Decide if we should retain
            if override_context:
                do_retain = True
                context = override_context
            else:
                do_retain, context = should_retain(state)
            
            if not do_retain:
                return
            
            # Extract learning content
            content = extract_learning(state, context)
            
            if not content or len(content) < 20:
                return  # Too short to be useful
            
            # Build metadata
            metadata = {}
            working_memory = state.get("working_memory", {})
            if "repo" in working_memory:
                metadata["repo"] = working_memory["repo"]
            if "instance_id" in working_memory:
                metadata["instance_id"] = working_memory["instance_id"]
            
            # Retain asynchronously (fire and forget)
            self._retain_async(content, context, metadata)
            
            self.logger.debug(f"Retained learning: [{context}] {content[:50]}...")
            
        except Exception as e:
            self.logger.warning(f"Failed to retain learning: {e}")
    
    def _retain_async(
        self,
        content: str,
        context: str,
        metadata: dict[str, str],
    ) -> None:
        """Retain content to memory.
        
        Uses persistent event loop to avoid "Event loop is closed" errors.
        """
        try:
            from agent.memory.async_utils import run_async
            
            run_async(
                self.memory.retain(content, context=context, metadata=metadata),
                timeout=10
            )
        except Exception as e:
            self.logger.warning(f"Async retain failed: {e}")
    
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
