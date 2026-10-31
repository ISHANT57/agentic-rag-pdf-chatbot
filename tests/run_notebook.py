"""Execute a notebook top to bottom and fail on the first cell that raises.

Usage:
    python tests/run_notebook.py Agentic_RAG_PDF_Chatbot.ipynb executed.ipynb
"""
import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def main(src: str, out: str) -> int:
    nb = nbformat.read(src, as_version=4)
    client = NotebookClient(nb, timeout=1800, kernel_name="python3")
    try:
        client.execute()
    except CellExecutionError as e:
        print(e)
        return 1
    finally:
        nbformat.write(nb, out)   # keep the outputs for inspection, also on failure
    ran = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("execution_count"))
    print(f"Executed {ran} code cells without errors")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
