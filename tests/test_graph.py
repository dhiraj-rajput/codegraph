"""
Tests for the code graph construction.
"""

import pytest
from parser.symbol_extractor import CodeSymbol
from graph_builder.code_graph import CodeGraph


def make_symbol(name, sym_type="function", file_path="test.py",
                calls=None, bases=None, parent=None, line=1):
    """Helper to create test symbols."""
    return CodeSymbol(
        symbol_id=f"{name}_{line}",
        name=name,
        qualified_name=f"{parent}.{name}" if parent else name,
        type=sym_type,
        file_path=file_path,
        line_start=line,
        line_end=line + 10,
        source_code=f"def {name}(): pass",
        calls=calls or [],
        base_classes=bases or [],
        parent=parent,
    )


class TestCodeGraph:
    @pytest.fixture
    def sample_symbols(self):
        return [
            make_symbol("loginHandler", calls=["validateJWT"], line=1),
            make_symbol("validateJWT", calls=["decodeToken"], line=15),
            make_symbol("decodeToken", line=30),
            make_symbol("AuthMiddleware", sym_type="class", line=50, bases=[]),
            make_symbol("check", sym_type="method", parent="AuthMiddleware",
                       calls=["validateJWT"], line=55),
        ]

    def test_build_graph(self, sample_symbols):
        graph = CodeGraph()
        graph.build(sample_symbols)

        stats = graph.stats()
        assert stats["total_nodes"] > 0
        assert stats["total_edges"] > 0
        assert stats["total_symbols"] == 5

    def test_calls_edges(self, sample_symbols):
        graph = CodeGraph()
        graph.build(sample_symbols)

        # loginHandler should connect to validateJWT
        callees = graph.get_callees("loginHandler_1")
        callee_names = [c.name for c in callees]
        assert "validateJWT" in callee_names

    def test_graph_expansion(self, sample_symbols):
        graph = CodeGraph()
        graph.build(sample_symbols)

        # Expand from loginHandler at depth 2
        expanded = graph.expand_graph(["loginHandler_1"], depth=2)
        expanded_names = [s.name for s in expanded]

        # Should reach validateJWT and decodeToken
        assert "loginHandler" in expanded_names
        assert "validateJWT" in expanded_names

    def test_call_chain(self, sample_symbols):
        graph = CodeGraph()
        graph.build(sample_symbols)

        chain = graph.get_call_chain("loginHandler_1", depth=3)
        # Should have loginHandler -> validateJWT
        assert len(chain) > 0
        callers = [c[0] for c in chain]
        assert "loginHandler" in callers

    def test_find_by_name(self, sample_symbols):
        graph = CodeGraph()
        graph.build(sample_symbols)

        results = graph.find_symbols_by_name("validateJWT")
        assert len(results) >= 1
        assert results[0].name == "validateJWT"

    def test_pagerank(self, sample_symbols):
        graph = CodeGraph()
        graph.build(sample_symbols)

        pr = graph.compute_pagerank()
        assert len(pr) > 0
        # validateJWT should have relatively high PageRank (called by 2 functions)

    def test_graph_proximity(self, sample_symbols):
        graph = CodeGraph()
        graph.build(sample_symbols)

        # Adjacent nodes have high proximity
        prox = graph.get_graph_proximity("loginHandler_1", "validateJWT_15")
        assert prox > 0.0

    def test_save_load(self, sample_symbols, tmp_path):
        graph = CodeGraph()
        graph.build(sample_symbols)

        path = str(tmp_path / "graph.pkl")
        graph.save(path)

        graph2 = CodeGraph()
        graph2.load(path)

        assert graph2.stats()["total_nodes"] == graph.stats()["total_nodes"]
        assert graph2.stats()["total_edges"] == graph.stats()["total_edges"]
