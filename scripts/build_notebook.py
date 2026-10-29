"""Build the Colab notebook from notebook_src.py (jupytext percent format).

The gold benchmark set in benchmark/gold_qa.json is embedded into the notebook so it
runs in Colab without any other file.

Usage:
    python scripts/build_notebook.py
"""
import ast
import io
import json
import pathlib
import re
import tokenize

import jupytext

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "notebook_src.py"
GOLD = ROOT / "benchmark" / "gold_qa.json"
OUT = ROOT / "Agentic_RAG_PDF_Chatbot.ipynb"

_STRING_TOKENS = {tokenize.STRING} | {getattr(tokenize, n) for n in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END")
                                      if hasattr(tokenize, n)}
BENCH_CELL = re.compile(r"^# @title 10\.[1-9] ")


def gate_cell(src: str, flag: str = "RUN_BENCHMARK") -> str:
    """Wrap a cell body in `if <flag>:` so 'Run all' skips it cleanly (an exception would stop Run all in Colab).

    Lines inside multi-line strings are left untouched, so prompt text is not changed by the indentation.
    """
    title, _, body = src.partition("\n")
    protected = set()
    for tok in tokenize.generate_tokens(io.StringIO(body).readline):
        if tok.type in _STRING_TOKENS and tok.end[0] > tok.start[0]:
            protected.update(range(tok.start[0] + 1, tok.end[0] + 1))
    lines = [("    " + ln if ln.strip() and i not in protected else ln) for i, ln in enumerate(body.split("\n"), 1)]
    # A bare trailing expression is shown by Jupyter only at top level, so display it explicitly.
    tree = ast.parse(body)
    last = tree.body[-1] if tree.body else None
    if isinstance(last, ast.Expr) and not (isinstance(last.value, ast.Call) and getattr(last.value.func, "id", "") in
                                           ("print", "display")):
        seg = ast.get_source_segment(body, last.value)
        start, end = last.lineno, last.end_lineno
        lines[start - 1:end] = [f"    display({seg})"]
    return "\n".join([title, f"if {flag}:", *lines])


def main():
    source = SRC.read_text()
    gold = json.dumps(json.loads(GOLD.read_text()), ensure_ascii=False, indent=1)
    if "'''" in gold:
        raise ValueError("gold_qa.json must not contain triple single quotes")
    source = source.replace("__GOLD_JSON__", gold)

    nb = jupytext.reads(source, fmt="py:percent")
    for cell in nb.cells:
        if cell.cell_type == "code" and BENCH_CELL.match(cell.source):
            cell.source = gate_cell(cell.source)
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"gpuType": "T4", "provenance": [], "toc_visible": True}
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "name": "python3"}
    # Colab shows "@title" cells as forms. Keep the code visible so reviewers can read it.
    jupytext.write(nb, OUT, fmt="ipynb")
    print(f"Wrote {OUT} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
