"""Environments package - Execution backends for agent.

Provides unified interface for running commands in different environments:
- LocalEnvironment: Direct execution on host machine
- DockerEnvironment: Isolated Docker container execution
"""

import copy
import importlib
from typing import TYPE_CHECKING

from .base import Environment

if TYPE_CHECKING:
    pass

# Environment type mappings
_ENVIRONMENT_MAPPING = {
    "local": "agent.environments.local.LocalEnvironment",
    "docker": "agent.environments.docker.DockerEnvironment",
}


def get_environment_class(spec: str) -> type[Environment]:
    """Get environment class by name or full import path.
    
    Args:
        spec: Either a shortcut name ("local", "docker") or full import path
        
    Returns:
        Environment class
        
    Raises:
        ValueError: If environment type not found
    """
    full_path = _ENVIRONMENT_MAPPING.get(spec, spec)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        msg = (
            f"Unknown environment type: {spec} (resolved to {full_path}). "
            f"Available: {list(_ENVIRONMENT_MAPPING.keys())}"
        )
        raise ValueError(msg) from e


def get_environment(config: dict, *, default_type: str = "local") -> Environment:
    """Create environment instance from configuration.
    
    Args:
        config: Environment configuration dict
        default_type: Default environment type if not specified
        
    Returns:
        Configured Environment instance
    """
    config = copy.deepcopy(config)
    environment_type = config.pop("type", default_type)
    environment_class = get_environment_class(environment_type)
    return environment_class(**config)


__all__ = [
    "Environment",
    "get_environment",
    "get_environment_class",
]
