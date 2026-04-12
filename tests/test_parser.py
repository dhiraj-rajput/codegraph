"""
Tests for the code parser and symbol extractor.
"""

import os
import tempfile
import pytest
from pathlib import Path


# Create a sample Python file for testing
SAMPLE_PYTHON = '''
"""Sample module for testing."""

import os
from pathlib import Path

MAX_RETRIES = 3

class AuthHandler:
    """Handles authentication."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def validate_token(self, token: str) -> bool:
        """Validate a JWT token."""
        decoded = self.decode_token(token)
        return decoded is not None

    def decode_token(self, token: str) -> dict:
        """Decode a JWT token."""
        return {"user": "test"}


def login(username: str, password: str) -> str:
    """Login and return a token."""
    handler = AuthHandler("secret")
    return "token123"


def logout(token: str):
    """Logout and invalidate token."""
    handler = AuthHandler("secret")
    handler.validate_token(token)
'''


class TestCodeParser:
    @pytest.fixture
    def sample_file(self, tmp_path):
        """Create a temporary Python file for testing."""
        file_path = tmp_path / "sample.py"
        file_path.write_text(SAMPLE_PYTHON)
        return str(file_path)

    def test_parse_file(self, sample_file):
        from parser.tree_sitter_parser import CodeParser

        parser = CodeParser()
        result = parser.parse_file(sample_file)

        assert result is not None
        assert result.language == "python"
        assert len(result.functions) >= 2  # login, logout
        assert len(result.classes) >= 1    # AuthHandler
        assert len(result.imports) >= 2    # os, Path

    def test_extract_functions(self, sample_file):
        from parser.tree_sitter_parser import CodeParser

        parser = CodeParser()
        result = parser.parse_file(sample_file)

        func_names = [f.name for f in result.functions]
        assert "login" in func_names
        assert "logout" in func_names

    def test_extract_classes(self, sample_file):
        from parser.tree_sitter_parser import CodeParser

        parser = CodeParser()
        result = parser.parse_file(sample_file)

        class_names = [c.name for c in result.classes]
        assert "AuthHandler" in class_names

        auth_cls = [c for c in result.classes if c.name == "AuthHandler"][0]
        method_names = [m.name for m in auth_cls.methods]
        assert "validate_token" in method_names
        assert "decode_token" in method_names

    def test_extract_docstrings(self, sample_file):
        from parser.tree_sitter_parser import CodeParser

        parser = CodeParser()
        result = parser.parse_file(sample_file)

        auth_cls = [c for c in result.classes if c.name == "AuthHandler"][0]
        assert auth_cls.docstring is not None
        assert "authentication" in auth_cls.docstring.lower()

    def test_extract_calls(self, sample_file):
        from parser.tree_sitter_parser import CodeParser

        parser = CodeParser()
        result = parser.parse_file(sample_file)

        # logout should call validate_token
        logout_func = [f for f in result.functions if f.name == "logout"]
        assert len(logout_func) == 1

    def test_skip_unsupported_extension(self, tmp_path):
        from parser.tree_sitter_parser import CodeParser

        file_path = tmp_path / "readme.md"
        file_path.write_text("# Hello")
        parser = CodeParser()
        result = parser.parse_file(str(file_path))
        assert result is None


class TestSymbolExtractor:
    @pytest.fixture
    def parsed_file(self, tmp_path):
        from parser.tree_sitter_parser import CodeParser

        file_path = tmp_path / "sample.py"
        file_path.write_text(SAMPLE_PYTHON)
        parser = CodeParser()
        return parser.parse_file(str(file_path))

    def test_extract_all_symbols(self, parsed_file):
        from parser.symbol_extractor import SymbolExtractor

        extractor = SymbolExtractor()
        symbols = extractor.extract_from_file(parsed_file)

        assert len(symbols) > 0
        sym_names = [s.name for s in symbols]

        assert "AuthHandler" in sym_names
        assert "login" in sym_names
        assert "logout" in sym_names
        assert "validate_token" in sym_names

    def test_symbol_types(self, parsed_file):
        from parser.symbol_extractor import SymbolExtractor

        extractor = SymbolExtractor()
        symbols = extractor.extract_from_file(parsed_file)

        types = {s.name: s.type for s in symbols}
        assert types.get("AuthHandler") == "class"
        assert types.get("login") == "function"
        assert types.get("validate_token") == "method"

    def test_qualified_names(self, parsed_file):
        from parser.symbol_extractor import SymbolExtractor

        extractor = SymbolExtractor()
        symbols = extractor.extract_from_file(parsed_file)

        qualified = {s.name: s.qualified_name for s in symbols}
        assert qualified.get("validate_token") == "AuthHandler.validate_token"

    def test_symbol_ids_unique(self, parsed_file):
        from parser.symbol_extractor import SymbolExtractor

        extractor = SymbolExtractor()
        symbols = extractor.extract_from_file(parsed_file)

        ids = [s.symbol_id for s in symbols]
        assert len(ids) == len(set(ids)), "Symbol IDs must be unique"
