"""LLM-assisted code parsing for intelligent chunking."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from modular_agents.core.models import CodeChunk
from modular_agents.knowledge.base import CodeParser

if TYPE_CHECKING:
    from modular_agents.llm import LLMProvider


class LLMCodeParser(CodeParser):
    """Code parser that uses LLM to intelligently chunk and analyze code.

    This parser:
    1. Breaks files into meaningful chunks (classes, functions, etc.)
    2. Generates summaries for each chunk
    3. Identifies the purpose and dependencies
    4. Uses language-specific patterns when available
    """

    def __init__(self, llm: LLMProvider):
        """Initialize the LLM code parser.

        Args:
            llm: LLM provider for analysis
        """
        self.llm = llm

    async def parse_file(
        self,
        file_path: str,
        content: str,
        language: str,
        module_name: str,
        repo_path: str,
    ) -> list[CodeChunk]:
        """Parse a file into code chunks with LLM assistance.

        Args:
            file_path: Relative path to the file
            content: File contents
            language: Programming language
            module_name: Module this file belongs to
            repo_path: Repository root path

        Returns:
            List of code chunks extracted from the file
        """
        # Don't parse empty files
        if not content.strip():
            return []

        # Don't parse very small files (< 10 lines)
        lines = content.split("\n")
        if len(lines) < 10:
            return []

        # Build parsing prompt
        prompt = self._build_parsing_prompt(file_path, content, language, module_name)

        # Get LLM response
        from modular_agents.llm.base import LLMMessage

        messages = [LLMMessage(role="user", content=prompt)]
        response = await self.llm.complete(messages=messages)

        # Parse response
        chunks = self._parse_response(
            response.content,
            file_path,
            content,
            language,
            module_name,
            repo_path,
        )

        return chunks

    def _build_parsing_prompt(
        self,
        file_path: str,
        content: str,
        language: str,
        module_name: str,
    ) -> str:
        """Build the prompt for code parsing."""
        return f"""Analyze this {language} source file and break it into meaningful code chunks.

File: {file_path}
Module: {module_name}
Language: {language}

```{language}
{content}
```

Instructions:
1. Identify all major code entities (classes, functions, interfaces, enums, etc.)
2. For each entity, provide:
   - Type (class, function, interface, enum, constant, etc.)
   - Name
   - Start and end line numbers (1-indexed)
   - Brief summary (1-2 sentences)
   - Purpose (what problem it solves)
   - Dependencies (other entities it uses, if obvious)

3. Focus on:
   - Public APIs and exports
   - Main classes and functions
   - Important types and interfaces
   - Skip trivial getters/setters
   - Skip test utilities unless significant

4. Return as JSON array:

```json
[
  {{
    "type": "class" | "function" | "interface" | "enum" | "constant" | "type",
    "name": "EntityName",
    "start_line": 10,
    "end_line": 45,
    "summary": "Brief description of what this does",
    "purpose": "What problem this solves or role it plays",
    "dependencies": ["OtherClass", "someFunction"]
  }}
]
```

IMPORTANT: Return ONLY the JSON array, no additional text.
"""

    def _parse_response(
        self,
        response: str,
        file_path: str,
        content: str,
        language: str,
        module_name: str,
        repo_path: str,
    ) -> list[CodeChunk]:
        """Parse LLM response into CodeChunk objects."""
        chunks = []

        try:
            # Extract JSON from response
            response = response.strip()
            start = response.find("[")
            end = response.rfind("]") + 1

            if start < 0 or end <= start:
                # No JSON found, try to parse entire file as single chunk
                return self._fallback_parse(
                    file_path, content, language, module_name, repo_path
                )

            json_str = response[start:end]
            entities = json.loads(json_str)

            # Convert to CodeChunk objects
            lines = content.split("\n")
            for entity in entities:
                # Extract chunk content
                start_line = entity.get("start_line", 1)
                end_line = entity.get("end_line", len(lines))

                # Ensure valid line numbers
                start_line = max(1, min(start_line, len(lines)))
                end_line = max(start_line, min(end_line, len(lines)))

                chunk_content = "\n".join(lines[start_line - 1 : end_line])

                chunk = CodeChunk(
                    id=str(uuid.uuid4()),
                    repo_path=repo_path,
                    file_path=file_path,
                    module_name=module_name,
                    language=language,
                    chunk_type=entity.get("type", "unknown"),
                    name=entity.get("name", "unknown"),
                    content=chunk_content,
                    start_line=start_line,
                    end_line=end_line,
                    summary=entity.get("summary", ""),
                    purpose=entity.get("purpose", ""),
                    dependencies=entity.get("dependencies", []),
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )

                chunks.append(chunk)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to simple parsing
            return self._fallback_parse(
                file_path, content, language, module_name, repo_path
            )

        return chunks

    def _fallback_parse(
        self,
        file_path: str,
        content: str,
        language: str,
        module_name: str,
        repo_path: str,
    ) -> list[CodeChunk]:
        """Fallback: Parse file using simple heuristics."""
        chunks = []
        lines = content.split("\n")

        # Language-specific patterns
        patterns = {
            "scala": [
                (r"^\s*class\s+(\w+)", "class"),
                (r"^\s*object\s+(\w+)", "object"),
                (r"^\s*trait\s+(\w+)", "trait"),
                (r"^\s*def\s+(\w+)", "function"),
            ],
            "python": [
                (r"^\s*class\s+(\w+)", "class"),
                (r"^\s*def\s+(\w+)", "function"),
                (r"^\s*async\s+def\s+(\w+)", "function"),
            ],
            "java": [
                (r"^\s*(?:public|private|protected)?\s*class\s+(\w+)", "class"),
                (r"^\s*(?:public|private|protected)?\s*interface\s+(\w+)", "interface"),
                (r"^\s*(?:public|private|protected)?\s*enum\s+(\w+)", "enum"),
            ],
            "typescript": [
                (r"^\s*(?:export\s+)?class\s+(\w+)", "class"),
                (r"^\s*(?:export\s+)?interface\s+(\w+)", "interface"),
                (r"^\s*(?:export\s+)?type\s+(\w+)", "type"),
                (r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)", "function"),
            ],
            "javascript": [
                (r"^\s*(?:export\s+)?class\s+(\w+)", "class"),
                (r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)", "function"),
                (r"^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(", "function"),
            ],
            "go": [
                (r"^\s*type\s+(\w+)\s+struct", "struct"),
                (r"^\s*type\s+(\w+)\s+interface", "interface"),
                (r"^\s*func\s+(\w+)", "function"),
            ],
            "rust": [
                (r"^\s*(?:pub\s+)?struct\s+(\w+)", "struct"),
                (r"^\s*(?:pub\s+)?enum\s+(\w+)", "enum"),
                (r"^\s*(?:pub\s+)?trait\s+(\w+)", "trait"),
                (r"^\s*(?:pub\s+)?fn\s+(\w+)", "function"),
            ],
        }

        lang_patterns = patterns.get(language.lower(), [])

        # Find entities using regex
        current_entity = None
        entity_start = 0

        for i, line in enumerate(lines, 1):
            for pattern, entity_type in lang_patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous entity
                    if current_entity:
                        chunk_content = "\n".join(lines[entity_start - 1 : i - 1])
                        chunks.append(
                            CodeChunk(
                                id=str(uuid.uuid4()),
                                repo_path=repo_path,
                                file_path=file_path,
                                module_name=module_name,
                                language=language,
                                chunk_type=current_entity[1],
                                name=current_entity[0],
                                content=chunk_content,
                                start_line=entity_start,
                                end_line=i - 1,
                                summary=f"{current_entity[1].title()} {current_entity[0]}",
                                purpose="",
                                dependencies=[],
                                created_at=datetime.now(),
                                updated_at=datetime.now(),
                            )
                        )

                    # Start new entity
                    current_entity = (match.group(1), entity_type)
                    entity_start = i
                    break

        # Save last entity
        if current_entity:
            chunk_content = "\n".join(lines[entity_start - 1 :])
            chunks.append(
                CodeChunk(
                    id=str(uuid.uuid4()),
                    repo_path=repo_path,
                    file_path=file_path,
                    module_name=module_name,
                    language=language,
                    chunk_type=current_entity[1],
                    name=current_entity[0],
                    content=chunk_content,
                    start_line=entity_start,
                    end_line=len(lines),
                    summary=f"{current_entity[1].title()} {current_entity[0]}",
                    purpose="",
                    dependencies=[],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
            )

        # If no chunks found, create one for whole file
        if not chunks:
            file_name = Path(file_path).stem
            chunks.append(
                CodeChunk(
                    id=str(uuid.uuid4()),
                    repo_path=repo_path,
                    file_path=file_path,
                    module_name=module_name,
                    language=language,
                    chunk_type="file",
                    name=file_name,
                    content=content,
                    start_line=1,
                    end_line=len(lines),
                    summary=f"File {file_name}",
                    purpose="",
                    dependencies=[],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
            )

        return chunks

    async def analyze_chunk(self, chunk: CodeChunk) -> CodeChunk:
        """Analyze a code chunk to extract summary and purpose.

        This is used when chunks are created without LLM analysis
        (e.g., from fallback parsing).

        Args:
            chunk: Code chunk to analyze

        Returns:
            Code chunk with summary and purpose filled in
        """
        # Skip if already analyzed
        if chunk.summary and chunk.purpose:
            return chunk

        # Build analysis prompt
        prompt = f"""Analyze this {chunk.language} code and provide a concise summary and purpose.

Type: {chunk.chunk_type}
Name: {chunk.name}

```{chunk.language}
{chunk.content}
```

Provide:
1. Summary: Brief 1-2 sentence description of what this code does
2. Purpose: What problem it solves or role it plays in the system
3. Dependencies: Key entities this code depends on (class names, function names, etc.)

Return as JSON:
```json
{{
  "summary": "...",
  "purpose": "...",
  "dependencies": ["Entity1", "Entity2"]
}}
```

Return ONLY the JSON, no additional text.
"""

        try:
            from modular_agents.llm.base import LLMMessage

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm.complete(messages=messages)

            # Parse response
            content = response.content.strip()
            start = content.find("{")
            end = content.rfind("}") + 1

            if start >= 0 and end > start:
                json_str = content[start:end]
                data = json.loads(json_str)

                chunk.summary = data.get("summary", chunk.summary)
                chunk.purpose = data.get("purpose", chunk.purpose)
                chunk.dependencies = data.get("dependencies", chunk.dependencies)
                chunk.updated_at = datetime.now()

        except (json.JSONDecodeError, KeyError, ValueError):
            # Keep existing values
            pass

        return chunk
