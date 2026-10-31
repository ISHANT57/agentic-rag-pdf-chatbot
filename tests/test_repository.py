"""Repository-level checks: the gold benchmark set, the notebook build step and the shipped notebook."""
import ast
import json
from collections import Counter

from conftest import ROOT, load_defs

GOLD = json.loads((ROOT / "benchmark" / "gold_qa.json").read_text(encoding="utf-8"))
QUESTIONS = GOLD if isinstance(GOLD, list) else GOLD["questions"]
FIELDS = {"id", "doc_id", "category", "question", "gold_answer", "gold_pages", "evidence", "keywords"}


def test_gold_set_shape():
    assert len(QUESTIONS) == 36
    assert len({q["id"] for q in QUESTIONS}) == 36
    for q in QUESTIONS:
        assert FIELDS <= q.keys(), q["id"]
    per_doc = Counter(q["doc_id"] for q in QUESTIONS)
    assert len(per_doc) == 4 and set(per_doc.values()) == {9}


def test_gold_categories_per_document():
    expected = {"factual": 3, "numeric_table": 2, "multi_hop": 2, "summary": 1, "unanswerable": 1}
    for doc in {q["doc_id"] for q in QUESTIONS}:
        assert Counter(q["category"] for q in QUESTIONS if q["doc_id"] == doc) == expected, doc


def test_gold_answerable_questions_have_evidence():
    for q in QUESTIONS:
        if q["category"] == "unanswerable":
            assert q["gold_answer"] == "NOT_IN_DOCUMENT" and not q["gold_pages"]
        elif q["category"] != "summary":
            assert q["gold_pages"] and q["evidence"], q["id"]


def test_build_gates_benchmark_cells():
    build = load_defs(ROOT / "scripts" / "build_notebook.py", ["_STRING_TOKENS", "gate_cell"],
                      extra={"ast": ast, "io": __import__("io"), "tokenize": __import__("tokenize")})
    cell = '# @title 10.9 Demo\nPROMPT = """line one\nline two"""\nresult = len(PROMPT)\nresult'
    gated = build.gate_cell(cell)
    body = gated.split("\n", 1)[1]
    ast.parse(body)                                   # the wrapped cell is valid Python
    assert body.startswith("if RUN_BENCHMARK:")
    assert "\nline two" in gated                       # multi-line string content is not re-indented
    assert "display(_last)" in gated and "if _last is not None" in gated


def test_shipped_notebook_is_valid_and_error_free():
    nb = json.loads((ROOT / "Agentic_RAG_PDF_Chatbot.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) >= 27
    for c in code:
        ast.parse("\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("!", "%"))))
        assert not [o for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert "widgets" not in nb["metadata"]            # GitHub cannot render partial widget state
