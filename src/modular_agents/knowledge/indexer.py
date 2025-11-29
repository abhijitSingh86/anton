"""Repository indexer for building knowledge base."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from modular_agents.core.models import IndexedRepo, ProjectType

if TYPE_CHECKING:
    from modular_agents.core.models import RepoKnowledge
    from modular_agents.knowledge.base import CodeParser, EmbeddingProvider
    from modular_agents.knowledge.store import KnowledgeStore

console = Console()


class RepositoryIndexer:
    """Indexes a repository by parsing code files and storing chunks."""

    # File extensions to index by language
    LANGUAGE_EXTENSIONS = {
        "scala": [".scala"],
        "python": [".py"],
        "java": [".java"],
        "javascript": [".js", ".jsx"],
        "typescript": [".ts", ".tsx"],
        "go": [".go"],
        "rust": [".rs"],
        "c": [".c", ".h"],
        "cpp": [".cpp", ".hpp", ".cc", ".cxx"],
        "csharp": [".cs"],
        "ruby": [".rb"],
        "php": [".php"],
        "swift": [".swift"],
        "kotlin": [".kt"],
    }

    # Directories to skip
    SKIP_DIRS = {
        ".git",
        ".svn",
        "node_modules",
        "target",
        "build",
        "dist",
        "out",
        "__pycache__",
        ".pytest_cache",
        ".tox",
        "venv",
        ".venv",
        "env",
        ".env",
        "vendor",
    }

    # Files to skip
    SKIP_FILES = {
        ".DS_Store",
        "Thumbs.db",
        ".gitignore",
        ".gitattributes",
    }

    def __init__(
        self,
        store: KnowledgeStore,
        parser: CodeParser,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        """Initialize the repository indexer.

        Args:
            store: Knowledge store for saving chunks
            parser: Code parser for chunking files
            embedding_provider: Optional embedding provider
        """
        self.store = store
        self.parser = parser
        self.embedding_provider = embedding_provider

    async def index_repository(
        self,
        repo_knowledge: RepoKnowledge,
        force: bool = False,
        show_progress: bool = True,
    ) -> dict:
        """Index a repository using its knowledge.

        Args:
            repo_knowledge: Repository knowledge from analyzer
            force: Re-index all files even if unchanged
            show_progress: Show progress bars

        Returns:
            Statistics dict with counts
        """
        repo_path = str(repo_knowledge.root_path)

        # Check if already indexed (but allow incremental updates)
        existing = self.store.get_repo(repo_path)
        is_first_index = existing is None

        # Collect all code files
        all_files = self._collect_files(repo_knowledge)

        if not all_files:
            console.print("[yellow]No code files found to index[/yellow]")
            return {"total_files": 0, "total_chunks": 0}

        # Filter files that need indexing (unless force=True)
        if force or is_first_index:
            # Index everything
            files_to_index = all_files
            skipped_count = 0
            console.print(f"\n[bold]Indexing Repository:[/bold] {repo_path}")
            if force and not is_first_index:
                console.print("[yellow]Force mode: re-indexing all files[/yellow]")
        else:
            # Incremental indexing: only index new/modified files
            files_to_index = []
            skipped_count = 0

            for file_path, module_name, language in all_files:
                # Get relative path
                rel_path = str(file_path.relative_to(repo_knowledge.root_path))

                # Get file modification time
                try:
                    file_mtime = file_path.stat().st_mtime
                except OSError:
                    # File no longer exists, skip
                    continue

                # Check if needs re-indexing
                if self.store.needs_reindex(repo_path, rel_path, file_mtime):
                    files_to_index.append((file_path, module_name, language))
                else:
                    skipped_count += 1

            console.print(f"\n[bold]Updating Repository:[/bold] {repo_path}")
            console.print(f"[green]Skipped {skipped_count} unchanged files[/green]")

        # Handle deleted files (only in incremental mode)
        deleted_count = 0
        if not is_first_index and not force:
            deleted_count = await self._cleanup_deleted_files(
                repo_knowledge, all_files
            )
            if deleted_count > 0:
                console.print(f"[yellow]Removed {deleted_count} deleted files[/yellow]")

        if not files_to_index:
            console.print("[green]All files up to date![/green]")
            return {
                "total_files": 0,
                "total_chunks": 0,
                "skipped_files": skipped_count,
                "deleted_files": deleted_count,
                "up_to_date": True,
            }

        console.print(f"Files to process: {len(files_to_index)}")

        # Index files with progress
        stats = await self._index_files(
            files_to_index,
            repo_knowledge,
            force=force,
            show_progress=show_progress,
        )

        stats["skipped_files"] = skipped_count
        stats["deleted_files"] = deleted_count

        # Update repository metadata
        if is_first_index:
            indexed_repo = IndexedRepo(
                repo_path=repo_path,
                project_type=repo_knowledge.project_type,
                language=self._detect_primary_language(repo_knowledge),
                total_chunks=stats["total_chunks"],
                total_files=stats["total_files"],
                indexed_at=datetime.now(),
                last_updated=datetime.now(),
            )
        else:
            # Update existing repo metadata
            # Get current totals from database
            indexed_files = self.store.list_indexed_files(repo_path)
            total_files = len(indexed_files)
            total_chunks = sum(f["chunk_count"] for f in indexed_files)

            indexed_repo = IndexedRepo(
                repo_path=repo_path,
                project_type=existing.project_type,
                language=existing.language,
                total_chunks=total_chunks,
                total_files=total_files,
                indexed_at=existing.indexed_at,  # Keep original
                last_updated=datetime.now(),
            )

        self.store.add_repo(indexed_repo)

        return stats

    def _collect_files(self, repo_knowledge: RepoKnowledge) -> list[tuple[Path, str, str]]:
        """Collect all code files to index.

        Returns:
            List of (file_path, module_name, language) tuples
        """
        files = []
        repo_path = repo_knowledge.root_path

        # Get primary language from modules
        primary_lang = self._detect_primary_language(repo_knowledge)
        extensions = self.LANGUAGE_EXTENSIONS.get(primary_lang.lower(), [])

        # Walk through modules
        for module in repo_knowledge.modules:
            module_path = module.path
            if not module_path.exists():
                continue

            # Determine language for this module
            module_lang = module.language if module.language != "unknown" else primary_lang

            # Walk directory tree
            for file_path in module_path.rglob("*"):
                # Skip directories
                if file_path.is_dir():
                    continue

                # Skip if in excluded directory
                if any(skip_dir in file_path.parts for skip_dir in self.SKIP_DIRS):
                    continue

                # Skip excluded files
                if file_path.name in self.SKIP_FILES:
                    continue

                # Check extension
                if file_path.suffix in extensions:
                    files.append((file_path, module.name, module_lang))

        return files

    def _detect_primary_language(self, repo_knowledge: RepoKnowledge) -> str:
        """Detect primary language from repository knowledge."""
        # Count language occurrences in modules
        language_counts = {}
        for module in repo_knowledge.modules:
            lang = module.language
            if lang and lang != "unknown":
                language_counts[lang] = language_counts.get(lang, 0) + 1

        # Return most common language
        if language_counts:
            return max(language_counts.items(), key=lambda x: x[1])[0]

        # Fallback based on project type
        type_languages = {
            ProjectType.SBT: "scala",
            ProjectType.MAVEN: "java",
            ProjectType.GRADLE: "java",
            ProjectType.NPM: "javascript",
            ProjectType.CARGO: "rust",
            ProjectType.POETRY: "python",
            ProjectType.GO: "go",
        }
        return type_languages.get(repo_knowledge.project_type, "unknown")

    async def _cleanup_deleted_files(
        self,
        repo_knowledge: RepoKnowledge,
        current_files: list[tuple[Path, str, str]],
    ) -> int:
        """Remove chunks for files that no longer exist.

        Args:
            repo_knowledge: Repository knowledge
            current_files: List of currently existing files

        Returns:
            Number of files cleaned up
        """
        repo_path = str(repo_knowledge.root_path)

        # Get list of indexed files
        indexed_files = self.store.list_indexed_files(repo_path)

        # Build set of current file paths (relative)
        current_paths = set()
        for file_path, _, _ in current_files:
            rel_path = str(file_path.relative_to(repo_knowledge.root_path))
            current_paths.add(rel_path)

        # Find deleted files
        deleted_count = 0
        for indexed_file in indexed_files:
            if indexed_file["file_path"] not in current_paths:
                # File was deleted, remove its chunks
                self.store.delete_file_chunks(repo_path, indexed_file["file_path"])
                self.store.delete_indexed_file(repo_path, indexed_file["file_path"])
                deleted_count += 1

        return deleted_count

    async def _index_files(
        self,
        files: list[tuple[Path, str, str]],
        repo_knowledge: RepoKnowledge,
        force: bool = False,
        show_progress: bool = True,
    ) -> dict:
        """Index a list of files.

        Args:
            files: List of (file_path, module_name, language) tuples
            repo_knowledge: Repository knowledge
            force: Force re-indexing (delete old chunks first)
            show_progress: Show progress bars

        Returns:
            Statistics dict
        """
        stats = {
            "total_files": 0,
            "total_chunks": 0,
            "failed_files": 0,
            "errors": [],
        }

        repo_path = str(repo_knowledge.root_path)

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Indexing files...", total=len(files)
                )

                for file_path, module_name, language in files:
                    try:
                        # Get relative path
                        rel_path = str(file_path.relative_to(repo_knowledge.root_path))

                        # Delete old chunks if file was already indexed
                        if force or self.store.get_indexed_file(repo_path, rel_path):
                            self.store.delete_file_chunks(repo_path, rel_path)

                        chunks = await self._index_file(
                            file_path,
                            module_name,
                            language,
                            repo_path,
                        )
                        stats["total_chunks"] += len(chunks)
                        stats["total_files"] += 1

                    except Exception as e:
                        stats["failed_files"] += 1
                        error_msg = f"{file_path}: {type(e).__name__}: {e}"
                        stats["errors"].append(error_msg)

                    progress.update(task, advance=1)

        else:
            # No progress bar
            for file_path, module_name, language in files:
                try:
                    # Get relative path
                    rel_path = str(file_path.relative_to(repo_knowledge.root_path))

                    # Delete old chunks if file was already indexed
                    if force or self.store.get_indexed_file(repo_path, rel_path):
                        self.store.delete_file_chunks(repo_path, rel_path)

                    chunks = await self._index_file(
                        file_path,
                        module_name,
                        language,
                        repo_path,
                    )
                    stats["total_chunks"] += len(chunks)
                    stats["total_files"] += 1

                except Exception as e:
                    stats["failed_files"] += 1
                    error_msg = f"{file_path}: {type(e).__name__}: {e}"
                    stats["errors"].append(error_msg)

        return stats

    async def _index_file(
        self,
        file_path: Path,
        module_name: str,
        language: str,
        repo_path: str,
    ) -> list:
        """Index a single file.

        Args:
            file_path: Path to the file
            module_name: Module this file belongs to
            language: Programming language
            repo_path: Repository root path

        Returns:
            List of created chunks
        """
        # Read file content
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            # Binary or non-UTF8 file - raise to be counted as failed
            raise ValueError(f"Not a UTF-8 text file: {e}")
        except OSError as e:
            # Permission or IO error
            raise ValueError(f"Cannot read file: {e}")

        # Get relative path and file mtime
        rel_path = str(file_path.relative_to(Path(repo_path)))
        file_mtime = file_path.stat().st_mtime

        # Parse file into chunks
        chunks = await self.parser.parse_file(
            file_path=rel_path,
            content=content,
            language=language,
            module_name=module_name,
            repo_path=repo_path,
        )

        # Store each chunk
        for chunk in chunks:
            await self.store.add_chunk(
                chunk,
                generate_embedding=self.embedding_provider is not None,
            )

        # Track indexed file metadata
        self.store.add_indexed_file(
            repo_path=repo_path,
            file_path=rel_path,
            module_name=module_name,
            language=language,
            file_mtime=file_mtime,
            chunk_count=len(chunks),
        )

        return chunks

    async def update_repository(
        self,
        repo_knowledge: RepoKnowledge,
        changed_files: list[Path] | None = None,
    ) -> dict:
        """Update an indexed repository with changed files.

        Args:
            repo_knowledge: Repository knowledge
            changed_files: Specific files to update (None = detect changes)

        Returns:
            Statistics dict
        """
        # For now, just re-index everything
        # TODO: Implement incremental updates using git diff
        return await self.index_repository(
            repo_knowledge,
            force=True,
            show_progress=True,
        )
