"""Configuration and template management.

Agent-level configuration only. Model and environment configs
are handled by their respective modules.
"""

from pathlib import Path
from typing import Any
import yaml

from jinja2 import Template, StrictUndefined
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Agent configuration - prompts and execution limits only.
    
    Model configuration: use agent.model.get_model()
    Environment configuration: use agent.environments.get_environment()
    """
    
    # Templates
    system_template: str = Field(..., description="System prompt template")
    instance_template: str = Field(..., description="Task/instance prompt template")
    action_observation_template: str = Field(..., description="Template for execution output")
    format_error_template: str = Field(..., description="Error when action parsing fails")
    timeout_template: str = Field(..., description="Message when command times out")
    
    # Action parsing
    action_regex: str = Field(
        r"```(?:bash|sh)\s*\n(.*?)\n```",
        description="Regex to extract command"
    )
    submit_pattern: str = Field(
        r"COMPLETE_TASK_AND_SUBMIT",
        description="Pattern indicating task completion"
    )
    
    # Limits
    step_limit: int = Field(50, gt=0)
    cost_limit: float = Field(3.0, gt=0.0)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        config_data = data.get("agent", data)
        
        return cls(**config_data)
    
    @classmethod
    def default(cls) -> "AgentConfig":
        """Load default configuration."""
        default_path = Path(__file__).parent / "default.yaml"
        return cls.from_yaml(default_path)


def render_template(template: str, **kwargs) -> str:
    """Render Jinja2 template with strict undefined checking."""
    return Template(template, undefined=StrictUndefined).render(**kwargs)


__all__ = ["AgentConfig", "render_template"]