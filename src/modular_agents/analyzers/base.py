"""Base analyzer interface for different project types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from modular_agents.core.models import ModuleProfile, ProjectType, RepoKnowledge


class BaseAnalyzer(ABC):
    """Abstract base class for project analyzers."""
    
    @property
    @abstractmethod
    def project_type(self) -> ProjectType:
        """The type of project this analyzer handles."""
        ...
    
    @abstractmethod
    def can_analyze(self, path: Path) -> bool:
        """Check if this analyzer can handle the given project."""
        ...
    
    @abstractmethod
    async def analyze(self, path: Path) -> RepoKnowledge:
        """Analyze the project and return knowledge."""
        ...
    
    @abstractmethod
    async def discover_modules(self, path: Path) -> list[ModuleProfile]:
        """Discover all modules in the project."""
        ...
    
    async def build_dependency_graph(
        self, modules: list[ModuleProfile]
    ) -> dict[str, list[str]]:
        """Build a dependency graph between modules."""
        graph = {}
        for module in modules:
            graph[module.name] = module.dependencies
        return graph
    
    def count_lines(self, file_path: Path) -> int:
        """Count lines of code in a file."""
        try:
            return len(file_path.read_text().splitlines())
        except Exception:
            return 0


class AnalyzerRegistry:
    """Registry for project analyzers."""
    
    _analyzers: list[type[BaseAnalyzer]] = []
    
    @classmethod
    def register(cls, analyzer_cls: type[BaseAnalyzer]):
        """Register an analyzer."""
        cls._analyzers.append(analyzer_cls)
        return analyzer_cls
    
    @classmethod
    def get_analyzer(cls, path: Path) -> BaseAnalyzer | None:
        """Get the appropriate analyzer for a project path."""
        for analyzer_cls in cls._analyzers:
            analyzer = analyzer_cls()
            if analyzer.can_analyze(path):
                return analyzer
        return None
    
    @classmethod
    def available(cls) -> list[str]:
        """List available analyzer types."""
        return [a().project_type.value for a in cls._analyzers]
