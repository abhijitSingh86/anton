"""Simple regex-based code parser that doesn't require LLM."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path

from modular_agents.core.models import CodeChunk
from modular_agents.knowledge.base import CodeParser


class SimpleCodeParser(CodeParser):
    """Simple code parser using regex patterns (no LLM required).

    This parser uses language-specific regex patterns to identify code entities.
    It's faster but less intelligent than LLM-based parsing.
    """

    # Language-specific patterns
    PATTERNS = {
        "scala": [
            (r"^\s*class\s+(\w+)", "class"),
            (r"^\s*object\s+(\w+)", "object"),
            (r"^\s*trait\s+(\w+)", "trait"),
            (r"^\s*case\s+class\s+(\w+)", "case_class"),
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

    def __init__(self):
        """Initialize the simple parser (no LLM needed)."""
        pass

    async def parse_file(
        self,
        file_path: str,
        content: str,
        language: str,
        module_name: str,
        repo_path: str,
    ) -> list[CodeChunk]:
        """Parse file using regex patterns.

        Args:
            file_path: Relative path to the file
            content: File contents
            language: Programming language
            module_name: Module this file belongs to
            repo_path: Repository root path

        Returns:
            List of code chunks
        """
        # Don't parse empty files
        if not content.strip():
            return []

        # Don't parse very small files (< 10 lines)
        lines = content.split("\n")
        if len(lines) < 10:
            return []

        # Get patterns for this language
        lang_patterns = self.PATTERNS.get(language.lower(), [])
        if not lang_patterns:
            # Unknown language, create single chunk for whole file
            return self._create_file_chunk(
                file_path, content, language, module_name, repo_path, lines
            )

        chunks = []
        current_entity = None
        entity_start = 0

        # Find entities using regex
        for i, line in enumerate(lines, 1):
            for pattern, entity_type in lang_patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous entity
                    if current_entity:
                        chunk_content = "\n".join(lines[entity_start - 1 : i - 1])
                        chunks.append(
                            self._create_chunk(
                                file_path,
                                chunk_content,
                                language,
                                module_name,
                                repo_path,
                                current_entity[0],  # name
                                current_entity[1],  # type
                                entity_start,
                                i - 1,
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
                self._create_chunk(
                    file_path,
                    chunk_content,
                    language,
                    module_name,
                    repo_path,
                    current_entity[0],
                    current_entity[1],
                    entity_start,
                    len(lines),
                )
            )

        # If no chunks found, create one for whole file
        if not chunks:
            chunks = [
                self._create_file_chunk(
                    file_path, content, language, module_name, repo_path, lines
                )
            ]

        return chunks

    def _create_chunk(
        self,
        file_path: str,
        content: str,
        language: str,
        module_name: str,
        repo_path: str,
        name: str,
        chunk_type: str,
        start_line: int,
        end_line: int,
    ) -> CodeChunk:
        """Create a code chunk."""
        return CodeChunk(
            id=str(uuid.uuid4()),
            repo_path=repo_path,
            file_path=file_path,
            module_name=module_name,
            language=language,
            chunk_type=chunk_type,
            name=name,
            content=content,
            start_line=start_line,
            end_line=end_line,
            summary=f"{chunk_type.title()}: {name}",
            purpose="",
            dependencies=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    def _create_file_chunk(
        self,
        file_path: str,
        content: str,
        language: str,
        module_name: str,
        repo_path: str,
        lines: list[str],
    ) -> CodeChunk:
        """Create a chunk for the entire file."""
        file_name = Path(file_path).stem
        return CodeChunk(
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
            summary=f"File: {file_name}",
            purpose="",
            dependencies=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    async def analyze_chunk(self, chunk: CodeChunk) -> CodeChunk:
        """Simple parser doesn't analyze chunks (no LLM)."""
        return chunk
