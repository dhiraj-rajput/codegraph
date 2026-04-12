"""
Language grammar loaders for Tree-sitter.

Each supported language needs its tree-sitter grammar installed as a pip package.
Currently supported: Python, JavaScript, TypeScript.
"""

from typing import Optional

# Lazy-load grammars to avoid import errors if a grammar is not installed.
_LANGUAGE_CACHE = {}

EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".rs": "rust",
}


def get_language(lang_name: str):
    """
    Load and cache a tree-sitter Language object for the given language.

    Returns None if the grammar package is not installed.
    """
    if lang_name in _LANGUAGE_CACHE:
        return _LANGUAGE_CACHE[lang_name]

    try:
        from tree_sitter import Language

        if lang_name == "python":
            import tree_sitter_python as ts_lang
        elif lang_name == "javascript":
            import tree_sitter_javascript as ts_lang
        elif lang_name == "typescript":
            import tree_sitter_typescript as ts_lang
            # typescript grammar exposes typescript() not language()
            lang_obj = Language(ts_lang.language_typescript())
            _LANGUAGE_CACHE[lang_name] = lang_obj
            return lang_obj
        else:
            # Unsupported language
            _LANGUAGE_CACHE[lang_name] = None
            return None

        lang_obj = Language(ts_lang.language())
        _LANGUAGE_CACHE[lang_name] = lang_obj
        return lang_obj

    except (ImportError, AttributeError, Exception):
        _LANGUAGE_CACHE[lang_name] = None
        return None


def language_for_extension(ext: str) -> Optional[str]:
    """Map a file extension to a language name."""
    return EXTENSION_MAP.get(ext.lower())
