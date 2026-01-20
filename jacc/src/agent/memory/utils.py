"""Utility functions for memory integration.

Helpers for formatting facts, building queries, and deciding what to retain.
Uses actual types from memory-api directly.
"""

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.state import AgentState
    # Use actual types from memory-api
    from src.engine.response_models import MemoryFact


def format_facts_for_prompt(facts: list["MemoryFact"], max_chars: int = 4000) -> str:
    """Format memory facts for injection into LLM prompt.
    
    Args:
        facts: List of MemoryFact objects from memory-api
        max_chars: Maximum characters to include
        
    Returns:
        Formatted string ready for prompt injection
    """
    if not facts:
        return ""
    
    lines = []
    total_chars = 0
    
    for i, fact in enumerate(facts, 1):
        # Format each fact with type and optional context
        type_tag = f"[{fact.fact_type.upper()}]"
        context_tag = f" ({fact.context})" if fact.context else ""
        line = f"{i}. {type_tag}{context_tag} {fact.text}"
        
        if total_chars + len(line) > max_chars:
            lines.append(f"... ({len(facts) - i + 1} more facts truncated)")
            break
            
        lines.append(line)
        total_chars += len(line) + 1  # +1 for newline
    
    return "\n".join(lines)


def build_recall_query(state: "AgentState") -> str:
    """Build recall query from current agent state.
    
    Constructs a query that captures:
    - Current error/output (if any)
    - Recent action context
    - Task summary
    
    Args:
        state: Current agent state
        
    Returns:
        Query string for memory recall
    """
    parts = []
    
    # Include error message if last output had non-zero return code
    last_output = state.get("last_output")
    if last_output:
        if last_output.get("returncode", 0) != 0:
            output_text = last_output.get("output", "")[:500]
            # Extract error type if present
            error_match = re.search(
                r"(Error|Exception|Traceback|Failed|error:|warning:)[^\n]*",
                output_text,
                re.IGNORECASE
            )
            if error_match:
                parts.append(f"Error: {error_match.group(0)}")
            else:
                parts.append(f"Command failed: {output_text[:200]}")
    
    # Include current action for context
    current_action = state.get("current_action")
    if current_action:
        parts.append(f"Action: {current_action[:100]}")
    
    # If no specific context, use generic task-related query
    if not parts:
        # Get first user message (task description)
        messages = state.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                task_snippet = msg.get("content", "")[:200]
                parts.append(f"Task context: {task_snippet}")
                break
    
    return " | ".join(parts) if parts else "general programming assistance"


def should_retain(state: "AgentState") -> tuple[bool, str]:
    """Decide whether to retain current observation as memory.
    
    Criteria for retention:
    - Error was encountered and then fixed
    - Task completed successfully
    - Significant discovery (file structure, etc.)
    
    Args:
        state: Current agent state
        
    Returns:
        Tuple of (should_retain, reason/context)
    """
    last_output = state.get("last_output")
    current_action = state.get("current_action", "")
    action_type = state.get("action_type")
    
    # Always retain task completion
    if action_type == "submit":
        return True, "task_complete"
    
    if not last_output:
        return False, ""
    
    returncode = last_output.get("returncode", 0)
    output_text = last_output.get("output", "")
    
    # Don't retain simple navigation commands
    navigation_commands = ["ls", "pwd", "cd", "cat", "head", "tail", "find", "grep"]
    if any(current_action.strip().startswith(cmd) for cmd in navigation_commands):
        # Unless it's a significant discovery
        if "No such file" in output_text or "Permission denied" in output_text:
            return True, "error_discovery"
        return False, ""
    
    # Retain error encounters (for learning)
    if returncode != 0:
        error_patterns = [
            r"Error:", r"Exception:", r"Traceback",
            r"ModuleNotFoundError", r"ImportError", r"AttributeError",
            r"TypeError", r"ValueError", r"KeyError",
            r"FAILED", r"failed", r"error"
        ]
        for pattern in error_patterns:
            if re.search(pattern, output_text, re.IGNORECASE):
                return True, "error_encountered"
        return False, ""
    
    # Retain successful test runs
    if "test" in current_action.lower() and returncode == 0:
        if "passed" in output_text.lower() or "ok" in output_text.lower():
            return True, "test_success"
    
    # Retain git operations  
    if current_action.strip().startswith("git "):
        return True, "git_operation"
    
    return False, ""


def extract_learning(state: "AgentState", context: str) -> str:
    """Extract a learning/observation from current state.
    
    Constructs a memory-worthy text from the action and its result.
    
    Args:
        state: Current agent state
        context: Context type from should_retain
        
    Returns:
        Text content to store as memory
    """
    current_action = state.get("current_action", "")
    last_output = state.get("last_output", {})
    output_text = last_output.get("output", "")[:1000] if last_output else ""
    
    if context == "task_complete":
        working_memory = state.get("working_memory", {})
        if "task_summary" in working_memory:
            return working_memory["task_summary"]
        return f"Completed task. Final action: {current_action}"
    
    elif context == "error_encountered":
        error_match = re.search(
            r"((?:Error|Exception|Traceback)[^\n]*(?:\n.*?(?=\n\n|\Z))?)",
            output_text,
            re.IGNORECASE | re.DOTALL
        )
        if error_match:
            return f"Encountered error:\n{error_match.group(1)[:500]}\nDuring: {current_action}"
        return f"Error during: {current_action}\nOutput: {output_text[:300]}"
    
    elif context == "error_discovery":
        return f"Discovery: {output_text[:300]} when running: {current_action}"
    
    elif context == "test_success":
        return f"Tests passed: {current_action}\nResult: {output_text[:300]}"
    
    elif context == "git_operation":
        return f"Git operation: {current_action}\nResult: {output_text[:300]}"
    
    else:
        return f"Action: {current_action}\nResult: {output_text[:300]}"


def get_error_from_output(output: dict | None) -> str | None:
    """Extract error message from execution output.
    
    Args:
        output: ExecutionResult dict
        
    Returns:
        Error message if present, None otherwise
    """
    if not output:
        return None
    
    if output.get("returncode", 0) == 0:
        return None
    
    output_text = output.get("output", "")
    
    patterns = [
        r"((?:Traceback \(most recent call last\):.*?(?:\n\w+Error:.*?)?\n?))",
        r"((?:Error|Exception):[^\n]+)",
        r"(FAILED[^\n]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()[:500]
    
    lines = output_text.strip().split("\n")[:5]
    return "\n".join(lines)[:500] if lines else None


__all__ = [
    "format_facts_for_prompt",
    "build_recall_query", 
    "should_retain",
    "extract_learning",
    "get_error_from_output",
]
