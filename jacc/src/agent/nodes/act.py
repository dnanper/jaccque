"""Act Node - Execute action in environment.

This node:
1. Takes the parsed action (bash command) from state
2. Executes it in the configured environment
3. Captures output and return code
4. Updates state with execution result

Note: Timeout is managed by the environment itself (DockerConfig.timeout, LocalConfig.timeout).
ActNode just calls execute() and the environment uses its default timeout.
"""

import time
import subprocess
from typing import Any, TYPE_CHECKING

from .base import EnvironmentNode

if TYPE_CHECKING:
    from agent.state import AgentState, ExecutionResult


class ActNode(EnvironmentNode):
    """Action execution node - runs commands in environment.
    
    Executes the current_action in the configured environment
    and captures the result. Uses environment's default timeout.
    """
    
    name = "act"
    
    def __call__(self, state: "AgentState") -> dict[str, Any]:
        """Execute current action in environment.
        
        Args:
            state: Current agent state with current_action
            
        Returns:
            State updates with execution result
        """
        action = state.get("current_action")
        action_type = state.get("action_type")
        
        # Skip execution for invalid or submit actions
        if action_type in ("invalid", "submit", None):
            self.logger.debug(f"Skipping execution for action_type: {action_type}")
            return {}
        
        if not action:
            self.logger.warning("No action to execute")
            return {"last_output": None}
        
        self.logger.info(f"Executing: {action[:100]}...")
        
        # Execute command - environment handles timeout internally
        start_time = time.time()
        timed_out = False
        
        try:
            result = self.environment.execute(command=action)
            duration = time.time() - start_time
            
        except subprocess.TimeoutExpired as e:
            # subprocess.TimeoutExpired is raised by subprocess.run
            duration = time.time() - start_time
            result = {
                "returncode": -1,
                "output": f"Command timed out: {str(e)}",
            }
            timed_out = True
            self.logger.warning(f"Command timed out: {action[:50]}...")
        
        except Exception as e:
            duration = time.time() - start_time
            result = {
                "returncode": -1,
                "output": f"Execution error: {str(e)}",
            }
            self.logger.error(f"Execution error: {e}")
        
        # Build execution result
        execution_result: "ExecutionResult" = {
            "returncode": result.get("returncode", -1),
            "output": result.get("output", ""),
            "timed_out": timed_out,
            "duration": duration,
        }
        
        self.logger.debug(
            f"Execution complete: returncode={execution_result['returncode']}, "
            f"duration={duration:.2f}s"
        )
        
        return {
            "last_output": execution_result,
            "step_count": state["step_count"] + 1,
        }
    
    def validate_input(self, state: "AgentState") -> bool:
        """Validate that we have an action to execute."""
        return state.get("action_type") not in (None, "invalid", "submit")
