from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ToolResult:
    """Represents the outcome of a tool invocation."""

    success: bool
    content: Any
    reasoning: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Simple base interface for agent tools."""

    name: str
    description: str

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    def __call__(self, **kwargs: Any) -> ToolResult:
        return self.run(**kwargs)

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a result."""
        raise NotImplementedError


class ToolExecutionError(RuntimeError):
    """Raised when a tool fails in a recoverable manner."""

    pass
