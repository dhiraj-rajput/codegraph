"""
Tests for the retrieval pipelines.
"""

import pytest
from parser.symbol_extractor import CodeSymbol
from graph_builder.code_graph import CodeGraph
from indexer.bm25_index import BM25CodeIndex, tokenize_code
from indexer.symbol_index import SymbolIndex
from indexer.page_index import PageIndex, CodePage


def make_symbol(name, sym_type="function", file_path="test.py",
                calls=None, parent=None, line=1, source="def x(): pass",
                docstring=None):
    return CodeSymbol(
        symbol_id=f"{name}_{line}",
        name=name,
        qualified_name=f"{parent}.{name}" if parent else name,
        type=sym_type,
        file_path=file_path,
        line_start=line,
        line_end=line + 10,
        source_code=source,
        docstring=docstring,
        calls=calls or [],
        parent=parent,
    )


class TestTokenizer:
    def test_camel_case_split(self):
        tokens = tokenize_code("validateJWT")
        assert "validate" in tokens or "validatejwt" in tokens

    def test_snake_case_split(self):
        tokens = tokenize_code("auth_middleware")
        assert "auth" in tokens
        assert "middleware" in tokens

    def test_preserve_identifier(self):
        tokens = tokenize_code("loginHandler")
        assert "loginhandler" in tokens

    def test_stopword_removal(self):
        tokens = tokenize_code("self.return_value = True")
        assert "self" not in tokens


class TestBM25Index:
    @pytest.fixture
    def sample_pages(self):
        return [
            CodePage(
                page_id="p1", symbol_id="s1", symbol_name="loginHandler",
                qualified_name="loginHandler", symbol_type="function",
                file_path="auth/handler.py",
                source_code="def loginHandler(request): return authenticate(request)",
                docstring="Handle user login",
                line_start=1, line_end=10, token_count=20,
            ),
            CodePage(
                page_id="p2", symbol_id="s2", symbol_name="validateJWT",
                qualified_name="validateJWT", symbol_type="function",
                file_path="auth/jwt.py",
                source_code="def validateJWT(token): return decode(token)",
                docstring="Validate JSON Web Token",
                line_start=1, line_end=8, token_count=15,
            ),
            CodePage(
                page_id="p3", symbol_id="s3", symbol_name="createUser",
                qualified_name="createUser", symbol_type="function",
                file_path="users/models.py",
                source_code="def createUser(name, email): return User(name, email)",
                docstring="Create a new user in the database",
                line_start=1, line_end=12, token_count=25,
            ),
        ]

    def test_build_and_search(self, sample_pages):
        index = BM25CodeIndex()
        index.build(sample_pages)
        assert index.page_count == 3

        results = index.search("authentication login", top_k=3)
        assert len(results) > 0
        # loginHandler should rank higher for auth queries
        top_names = [r.page.symbol_name for r in results]
        assert "loginHandler" in top_names

    def test_code_identifier_search(self, sample_pages):
        index = BM25CodeIndex()
        index.build(sample_pages)

        results = index.search("validateJWT", top_k=3)
        assert len(results) > 0
        assert results[0].page.symbol_name == "validateJWT"

    def test_empty_query(self, sample_pages):
        index = BM25CodeIndex()
        index.build(sample_pages)
        results = index.search("", top_k=3)
        assert len(results) == 0


class TestSymbolIndex:
    @pytest.fixture
    def sample_symbols(self):
        return [
            make_symbol("loginHandler", file_path="auth/handler.py", line=1),
            make_symbol("validateJWT", file_path="auth/jwt.py", line=10),
            make_symbol("AuthMiddleware", sym_type="class", file_path="auth/middleware.py", line=1),
        ]

    def test_exact_lookup(self, sample_symbols):
        index = SymbolIndex()
        index.build(sample_symbols)

        results = index.lookup("validateJWT")
        assert len(results) >= 1
        assert results[0]["name"] == "validateJWT"

    def test_fuzzy_lookup(self, sample_symbols):
        index = SymbolIndex()
        index.build(sample_symbols)

        results = index.lookup_fuzzy("JWT")
        assert len(results) >= 1

    def test_not_found(self, sample_symbols):
        index = SymbolIndex()
        index.build(sample_symbols)

        results = index.lookup("nonExistentFunction")
        assert len(results) == 0


class TestPageIndex:
    def test_build_from_symbols(self):
        symbols = [
            make_symbol("foo", source="def foo(): pass"),
            make_symbol("Bar", sym_type="class", source="class Bar: pass"),
        ]
        index = PageIndex()
        index.build(symbols)
        assert index.page_count == 2

    def test_lookup_by_name(self):
        symbols = [
            make_symbol("foo", source="def foo(): pass"),
        ]
        index = PageIndex()
        index.build(symbols)

        pages = index.get_by_name("foo")
        assert len(pages) == 1
        assert pages[0].symbol_name == "foo"
