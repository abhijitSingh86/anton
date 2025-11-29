"""SBT project analyzer for Scala codebases."""

from __future__ import annotations

import re
from pathlib import Path

from modular_agents.core.models import ModuleProfile, ProjectType, RepoKnowledge

from .base import AnalyzerRegistry, BaseAnalyzer


@AnalyzerRegistry.register
class SBTAnalyzer(BaseAnalyzer):
    """Analyzer for SBT-based Scala projects."""
    
    @property
    def project_type(self) -> ProjectType:
        return ProjectType.SBT
    
    def can_analyze(self, path: Path) -> bool:
        """Check if this is an SBT project."""
        return (path / "build.sbt").exists()
    
    async def analyze(self, path: Path) -> RepoKnowledge:
        """Analyze an SBT project."""
        modules = await self.discover_modules(path)
        dep_graph = await self.build_dependency_graph(modules)
        
        # Compute dependents (reverse dependencies)
        for module in modules:
            module.dependents = [
                m.name for m in modules
                if module.name in m.dependencies
            ]
        
        build_content = ""
        build_file = path / "build.sbt"
        if build_file.exists():
            build_content = build_file.read_text()
        
        return RepoKnowledge(
            root_path=path,
            project_type=ProjectType.SBT,
            modules=modules,
            dependency_graph=dep_graph,
            build_file=build_content,
        )
    
    async def discover_modules(self, path: Path) -> list[ModuleProfile]:
        """Discover SBT modules from build.sbt."""
        modules = []
        build_file = path / "build.sbt"
        
        if not build_file.exists():
            return modules
        
        content = build_file.read_text()
        
        # Parse lazy val definitions for projects
        # Pattern: lazy val moduleName = (project in file("path"))
        project_pattern = r'lazy\s+val\s+(\w+)\s*=\s*\(?\s*project\s+in\s+file\("([^"]+)"\)'
        
        for match in re.finditer(project_pattern, content):
            module_name = match.group(1)
            module_path = path / match.group(2)
            
            if module_path.exists():
                profile = await self._analyze_module(module_name, module_path, content)
                modules.append(profile)
        
        # Also check for single-module project (root project)
        if not modules:
            src_main = path / "src" / "main" / "scala"
            if src_main.exists():
                profile = await self._analyze_module("root", path, content)
                modules.append(profile)
        
        return modules
    
    async def _analyze_module(
        self, name: str, module_path: Path, build_content: str
    ) -> ModuleProfile:
        """Analyze a single SBT module."""
        src_path = module_path / "src" / "main" / "scala"
        test_path = module_path / "src" / "test" / "scala"
        
        # Discover packages
        packages = []
        if src_path.exists():
            packages = self._discover_packages(src_path)
        
        # Count files and LOC
        file_count = 0
        loc = 0
        if src_path.exists():
            for scala_file in src_path.rglob("*.scala"):
                file_count += 1
                loc += self.count_lines(scala_file)
        
        # Extract public API (traits, classes, objects)
        public_api = []
        if src_path.exists():
            public_api = self._extract_public_api(src_path)
        
        # Parse dependencies from build.sbt
        dependencies = self._parse_dependencies(name, build_content)
        external_deps = self._parse_external_deps(name, build_content)
        
        # Infer purpose from package names and files
        purpose = self._infer_purpose(name, packages, public_api)
        
        # Discover test patterns
        test_patterns = []
        if test_path.exists():
            test_patterns = self._discover_test_patterns(test_path)
        
        # Extract code examples for style reference
        code_examples = self._extract_code_examples(src_path)

        # Detect naming patterns
        naming_patterns = self._detect_naming_patterns(src_path)

        # Detect framework from dependencies
        framework = self._detect_framework(external_deps)

        return ModuleProfile(
            name=name,
            path=module_path,
            purpose=purpose,
            language="scala",  # SBT projects are Scala
            framework=framework,
            packages=packages,
            public_api=public_api[:20],  # Limit to top 20
            dependencies=dependencies,
            external_deps=external_deps,
            test_patterns=test_patterns,
            file_count=file_count,
            loc=loc,
            code_examples=code_examples,
            naming_patterns=naming_patterns,
        )
    
    def _discover_packages(self, src_path: Path) -> list[str]:
        """Discover package names in source directory."""
        packages = set()
        
        for scala_file in src_path.rglob("*.scala"):
            content = scala_file.read_text()
            match = re.search(r'^package\s+([\w.]+)', content, re.MULTILINE)
            if match:
                packages.add(match.group(1))
        
        return sorted(packages)
    
    def _extract_public_api(self, src_path: Path) -> list[str]:
        """Extract public traits, classes, and objects."""
        api_items = []
        
        patterns = [
            (r'^\s*(?:sealed\s+)?trait\s+(\w+)', 'trait'),
            (r'^\s*(?:case\s+)?class\s+(\w+)', 'class'),
            (r'^\s*object\s+(\w+)', 'object'),
        ]
        
        for scala_file in src_path.rglob("*.scala"):
            content = scala_file.read_text()
            for pattern, kind in patterns:
                for match in re.finditer(pattern, content, re.MULTILINE):
                    api_items.append(f"{kind} {match.group(1)}")
        
        return api_items
    
    def _parse_dependencies(self, module_name: str, build_content: str) -> list[str]:
        """Parse internal module dependencies from build.sbt."""
        deps = []
        
        # Pattern: moduleName.dependsOn(other1, other2)
        pattern = rf'{module_name}\s*\.\s*dependsOn\s*\(([^)]+)\)'
        match = re.search(pattern, build_content)
        
        if match:
            dep_str = match.group(1)
            # Extract module names
            for dep in re.findall(r'\b(\w+)\b', dep_str):
                if dep not in ['%', 'Test', 'Compile']:
                    deps.append(dep)
        
        return deps
    
    def _parse_external_deps(self, module_name: str, build_content: str) -> list[str]:
        """Parse external library dependencies."""
        deps = []
        
        # Look for libraryDependencies
        # Pattern: "org" %% "name" % "version"
        pattern = r'"([^"]+)"\s*%%?\s*"([^"]+)"\s*%\s*"([^"]+)"'
        
        for match in re.finditer(pattern, build_content):
            org, name, version = match.groups()
            deps.append(f"{org}:{name}:{version}")
        
        return list(set(deps))[:15]  # Limit to 15
    
    def _infer_purpose(
        self, name: str, packages: list[str], api_items: list[str]
    ) -> str:
        """Infer module purpose from its structure."""
        keywords = {
            'api': 'REST API endpoints and HTTP handling',
            'web': 'Web layer and HTTP handling',
            'http': 'HTTP server and routing',
            'domain': 'Domain models and business logic',
            'core': 'Core domain models and shared logic',
            'model': 'Data models and entities',
            'persistence': 'Database access and storage',
            'repository': 'Data access layer',
            'db': 'Database operations',
            'service': 'Business services and operations',
            'common': 'Shared utilities and common code',
            'util': 'Utility functions and helpers',
            'cache': 'Caching layer',
            'auth': 'Authentication and authorization',
            'test': 'Testing utilities',
        }
        
        name_lower = name.lower()
        for keyword, purpose in keywords.items():
            if keyword in name_lower:
                return purpose
        
        # Check packages
        for pkg in packages:
            pkg_lower = pkg.lower()
            for keyword, purpose in keywords.items():
                if keyword in pkg_lower:
                    return purpose
        
        return f"Module: {name}"
    
    def _discover_test_patterns(self, test_path: Path) -> list[str]:
        """Discover testing patterns used."""
        patterns = set()

        framework_indicators = {
            'org.scalatest': 'ScalaTest',
            'org.specs2': 'Specs2',
            'zio.test': 'ZIO Test',
            'munit': 'MUnit',
            'org.mockito': 'Mockito',
            'org.scalamock': 'ScalaMock',
        }

        for test_file in test_path.rglob("*.scala"):
            content = test_file.read_text()
            for indicator, framework in framework_indicators.items():
                if indicator in content:
                    patterns.add(framework)

        return list(patterns)

    def _extract_code_examples(self, src_path: Path) -> list[str]:
        """Extract representative code examples to show style."""
        examples = []

        if not src_path.exists():
            return examples

        # Find a few representative files
        scala_files = list(src_path.rglob("*.scala"))[:3]  # Max 3 files

        for scala_file in scala_files:
            try:
                content = scala_file.read_text()
                lines = content.split('\n')

                # Extract first class/trait/object definition (up to 30 lines)
                in_definition = False
                definition_lines = []
                brace_count = 0

                for line in lines:
                    if not in_definition and any(kw in line for kw in ['class ', 'trait ', 'object ']):
                        in_definition = True

                    if in_definition:
                        definition_lines.append(line)
                        brace_count += line.count('{') - line.count('}')

                        # Stop after balanced braces or max lines
                        if (brace_count == 0 and len(definition_lines) > 5) or len(definition_lines) > 30:
                            break

                if definition_lines:
                    example = '\n'.join(definition_lines)
                    examples.append(f"// From {scala_file.name}\n{example}")

            except Exception:
                pass  # Skip files that can't be read

        return examples[:2]  # Max 2 examples

    def _detect_naming_patterns(self, src_path: Path) -> list[str]:
        """Detect naming conventions from code."""
        patterns = []

        if not src_path.exists():
            return patterns

        # Analyze file and class names
        scala_files = list(src_path.rglob("*.scala"))[:10]

        class_names = []
        method_names = []

        for scala_file in scala_files:
            try:
                content = scala_file.read_text()

                # Extract class/object names
                for match in re.finditer(r'(?:class|object|trait)\s+(\w+)', content):
                    class_names.append(match.group(1))

                # Extract method names (simplified)
                for match in re.finditer(r'def\s+(\w+)', content):
                    method_names.append(match.group(1))

            except Exception:
                pass

        # Detect patterns
        if class_names:
            # Check if PascalCase
            if all(name[0].isupper() for name in class_names if name):
                patterns.append("Classes use PascalCase")

        if method_names:
            # Check if camelCase
            if all(name[0].islower() for name in method_names if name):
                patterns.append("Methods use camelCase")

        # Check for common Scala patterns
        all_files_content = ""
        for scala_file in scala_files[:5]:
            try:
                all_files_content += scala_file.read_text()
            except Exception:
                pass

        if "case class" in all_files_content:
            patterns.append("Uses case classes for data models")
        if "implicit" in all_files_content:
            patterns.append("Uses implicit parameters/conversions")
        if "extends" in all_files_content and "trait" in all_files_content:
            patterns.append("Uses trait-based composition")

        return patterns

    def _detect_framework(self, external_deps: list[str]) -> str:
        """Detect framework from dependencies."""
        dep_str = ' '.join(external_deps).lower()

        frameworks = {
            'akka': 'Akka',
            'play': 'Play Framework',
            'zio': 'ZIO',
            'cats': 'Cats/Cats Effect',
            'http4s': 'Http4s',
            'spark': 'Apache Spark',
            'slick': 'Slick',
            'doobie': 'Doobie',
        }

        for keyword, framework in frameworks.items():
            if keyword in dep_str:
                return framework

        return ""
