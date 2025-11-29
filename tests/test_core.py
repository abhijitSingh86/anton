"""Tests for modular-agents."""

import pytest
from pathlib import Path

from modular_agents.core import ModuleProfile, ProjectType, RepoKnowledge
from modular_agents.llm import LLMConfig, LLMProviderRegistry


class TestModels:
    """Test core data models."""
    
    def test_module_profile_creation(self):
        profile = ModuleProfile(
            name="test-module",
            path=Path("/test/path"),
            purpose="Test module",
        )
        assert profile.name == "test-module"
        assert profile.purpose == "Test module"
    
    def test_repo_knowledge_get_module(self):
        module1 = ModuleProfile(name="api", path=Path("/api"))
        module2 = ModuleProfile(name="domain", path=Path("/domain"))
        
        knowledge = RepoKnowledge(
            root_path=Path("/repo"),
            project_type=ProjectType.SBT,
            modules=[module1, module2],
        )
        
        assert knowledge.get_module("api") == module1
        assert knowledge.get_module("domain") == module2
        assert knowledge.get_module("nonexistent") is None
    
    def test_repo_knowledge_affected_modules(self):
        module1 = ModuleProfile(name="api", path=Path("/repo/api"))
        module2 = ModuleProfile(name="domain", path=Path("/repo/domain"))
        
        knowledge = RepoKnowledge(
            root_path=Path("/repo"),
            project_type=ProjectType.SBT,
            modules=[module1, module2],
        )
        
        affected = knowledge.get_affected_modules([
            "/repo/api/src/Main.scala",
            "/repo/domain/src/User.scala",
        ])
        
        assert set(affected) == {"api", "domain"}


class TestLLMConfig:
    """Test LLM configuration."""
    
    def test_config_creation(self):
        config = LLMConfig(
            model="claude-sonnet-4-20250514",
            api_key="test-key",
        )
        assert config.model == "claude-sonnet-4-20250514"
        assert config.api_key == "test-key"
        assert config.temperature == 0.7  # default
    
    def test_config_with_base_url(self):
        config = LLMConfig(
            model="local-model",
            base_url="http://localhost:8000",
        )
        assert config.base_url == "http://localhost:8000"


class TestProviderRegistry:
    """Test LLM provider registry."""
    
    def test_available_providers(self):
        available = LLMProviderRegistry.available()
        # At minimum, should be a list (may be empty if no providers installed)
        assert isinstance(available, list)
    
    def test_get_unknown_provider(self):
        provider = LLMProviderRegistry.get("nonexistent-provider")
        assert provider is None


# Integration tests (require actual LLM access)
class TestAnalyzers:
    """Test project analyzers."""
    
    def test_sbt_analyzer_detection(self, tmp_path):
        from modular_agents.analyzers import AnalyzerRegistry
        
        # Create a fake SBT project
        (tmp_path / "build.sbt").write_text("name := \"test\"")
        
        analyzer = AnalyzerRegistry.get_analyzer(tmp_path)
        assert analyzer is not None
        assert analyzer.project_type == ProjectType.SBT
    
    def test_generic_analyzer_fallback(self, tmp_path):
        from modular_agents.analyzers import AnalyzerRegistry
        
        # Create a directory with no recognized build file
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")
        
        analyzer = AnalyzerRegistry.get_analyzer(tmp_path)
        assert analyzer is not None
        # Should get generic analyzer as fallback
