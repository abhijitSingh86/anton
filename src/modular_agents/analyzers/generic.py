"""Generic analyzer that uses LLM to understand any project structure."""

from __future__ import annotations

import json
from pathlib import Path

from modular_agents.core.models import ModuleProfile, ProjectType, RepoKnowledge
from modular_agents.llm import LLMMessage, LLMProvider

from .base import AnalyzerRegistry, BaseAnalyzer


@AnalyzerRegistry.register
class GenericAnalyzer(BaseAnalyzer):
    """Generic analyzer that uses LLM to understand project structure.
    
    This analyzer can handle any project type by using an LLM to analyze
    the project structure and infer modules.
    """
    
    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm
    
    @property
    def project_type(self) -> ProjectType:
        return ProjectType.UNKNOWN
    
    def can_analyze(self, path: Path) -> bool:
        """Generic analyzer can analyze any directory."""
        return path.is_dir()
    
    async def analyze(self, path: Path) -> RepoKnowledge:
        """Analyze project using LLM assistance."""
        # First, detect project type
        detected_type = self._detect_project_type(path)
        
        # Gather project structure
        structure = self._get_directory_structure(path, max_depth=3)
        
        # If no LLM, use heuristic-based module discovery
        if not self.llm:
            modules = await self._discover_modules_heuristic(path)
        else:
            modules = await self._discover_modules_llm(path, structure)
        
        dep_graph = await self.build_dependency_graph(modules)
        
        return RepoKnowledge(
            root_path=path,
            project_type=detected_type,
            modules=modules,
            dependency_graph=dep_graph,
        )
    
    async def discover_modules(self, path: Path) -> list[ModuleProfile]:
        """Discover modules in the project."""
        if self.llm:
            structure = self._get_directory_structure(path, max_depth=3)
            return await self._discover_modules_llm(path, structure)
        return await self._discover_modules_heuristic(path)
    
    def _detect_project_type(self, path: Path) -> ProjectType:
        """Detect project type from build files."""
        indicators = {
            "build.sbt": ProjectType.SBT,
            "pom.xml": ProjectType.MAVEN,
            "build.gradle": ProjectType.GRADLE,
            "build.gradle.kts": ProjectType.GRADLE,
            "package.json": ProjectType.NPM,
            "Cargo.toml": ProjectType.CARGO,
            "pyproject.toml": ProjectType.POETRY,
            "go.mod": ProjectType.GO,
        }
        
        for filename, ptype in indicators.items():
            if (path / filename).exists():
                return ptype
        
        return ProjectType.UNKNOWN
    
    def _get_directory_structure(self, path: Path, max_depth: int = 3) -> str:
        """Get a text representation of directory structure."""
        lines = []
        
        def walk(current: Path, depth: int, prefix: str = ""):
            if depth > max_depth:
                return
            
            try:
                items = sorted(current.iterdir())
            except PermissionError:
                return
            
            # Filter out common noise
            skip = {'.git', 'node_modules', 'target', '__pycache__', '.idea', 'venv', '.venv'}
            items = [i for i in items if i.name not in skip]
            
            for i, item in enumerate(items[:20]):  # Limit items per directory
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}{item.name}")
                
                if item.is_dir():
                    extension = "    " if is_last else "│   "
                    walk(item, depth + 1, prefix + extension)
        
        lines.append(path.name + "/")
        walk(path, 0)
        
        return "\n".join(lines[:200])  # Limit total lines
    
    async def _discover_modules_heuristic(self, path: Path) -> list[ModuleProfile]:
        """Discover modules using heuristics (no LLM)."""
        modules = []
        
        # Common module directory patterns
        module_indicators = [
            "src",
            "lib",
            "packages",
            "modules",
            "apps",
            "services",
        ]
        
        for indicator in module_indicators:
            indicator_path = path / indicator
            if indicator_path.is_dir():
                for subdir in indicator_path.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith('.'):
                        modules.append(ModuleProfile(
                            name=subdir.name,
                            path=subdir,
                            purpose=f"Module: {subdir.name}",
                        ))
        
        # If no modules found, treat root as single module
        if not modules:
            modules.append(ModuleProfile(
                name="root",
                path=path,
                purpose="Root module",
            ))
        
        return modules
    
    async def _discover_modules_llm(
        self, path: Path, structure: str
    ) -> list[ModuleProfile]:
        """Use LLM to discover and understand modules."""
        prompt = f"""Analyze this project structure and identify distinct modules/components.

Project Structure:
```
{structure}
```

For each module, provide:
1. name: A short identifier
2. path: Relative path from root
3. purpose: Brief description of what this module does

Respond with JSON array:
```json
[
  {{"name": "...", "path": "...", "purpose": "..."}},
  ...
]
```

Only include actual code modules, not config or build directories.
"""
        
        response = await self.llm.complete([
            LLMMessage(role="user", content=prompt)
        ])
        
        # Parse JSON from response
        try:
            # Extract JSON from response
            content = response.content
            start = content.find('[')
            end = content.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                module_data = json.loads(json_str)
                
                return [
                    ModuleProfile(
                        name=m["name"],
                        path=path / m["path"],
                        purpose=m.get("purpose", ""),
                    )
                    for m in module_data
                ]
        except (json.JSONDecodeError, KeyError):
            pass
        
        # Fallback to heuristic
        return await self._discover_modules_heuristic(path)
