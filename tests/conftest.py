"""Load selected top-level definitions from notebook_src.py without running the notebook.

The notebook installs packages, starts Ollama and loads models at import time, so the tests cannot import
it. Instead they parse the source and execute only the pure helpers they test.
"""
import __future__
import ast
import bisect
import pathlib
import re
import types
import unicodedata
from collections import Counter
from dataclasses import dataclass

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "notebook_src.py"


def load_defs(path: pathlib.Path, names: list, extra: dict = None) -> types.SimpleNamespace:
    """Execute the top-level functions, classes and assignments called `names` from `path`, in that order."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nodes[node.name] = node
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            nodes[node.targets[0].id] = node
    missing = [n for n in names if n not in nodes]
    assert not missing, f"not found in {path.name}: {missing}"
    module = ast.Module(body=[nodes[n] for n in names], type_ignores=[])
    # Annotations such as `-> fitz.Document` are not evaluated, so the tests need only what the code calls.
    code = compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True)
    ns = {"re": re, "bisect": bisect, "unicodedata": unicodedata, "Counter": Counter, "dataclass": dataclass,
          "Optional": None, "CFG": types.SimpleNamespace(chunk_size=1200, chunk_overlap=200, max_pdf_mb=100)}
    ns.update(extra or {})
    exec(code, ns)
    return types.SimpleNamespace(**{n: ns[n] for n in names})


@pytest.fixture(scope="session")
def nb():
    """The helpers under test, taken from notebook_src.py."""
    return load_defs(SRC, [
        "page_label", "_CITE_LIST", "expand_cites", "_CITE", "render_answer", "cited_pages",
        "REFUSAL_RE", "is_refusal", "match_key", "NUM_WORDS", "norm_answer", "DIGIT_WORDS", "keyword_hit",
        "token_f1", "_TABLE_ROW", "find_markdown_tables", "msg_text", "clean_chat",
    ])
