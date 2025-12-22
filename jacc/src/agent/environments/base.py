from abc import ABC, abstractmethod
from typing import Any, Dict

class Environment(ABC):
    """Base class for environments"""
    @abstractmethod
    def execute(self, command: str, cwd: str = "", timeout: int | None = None) -> Dict[str, Any]:
        """Execute a command in the environment"""
        pass

    @abstractmethod
    def get_template_vars(self) -> Dict[str, Any]:
        """Get template variables for the environment"""
        pass

    def cleanup(self):
        pass