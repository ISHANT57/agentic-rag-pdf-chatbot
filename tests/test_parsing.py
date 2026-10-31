"""PDF handling: upload validation, Markdown tables, chunking with page tracking, Gradio message parsing."""
import types

import fitz
import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter

from conftest import SRC, load_defs


@pytest.fixture(scope="module")
def parsing():
    ns = load_defs(SRC, ["open_validated", "_HEADING", "Chunk", "chunk_document"],
                   extra={"fitz": fitz, "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter})
    ns.cfg = types.SimpleNamespace(chunk_size=300, chunk_overlap=50, max_pdf_mb=1)
    return ns


def pdf_bytes(pages: int = 1, text: str = "Hello PDF") -> bytes:
    doc = fitz.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"{text} {i + 1}")
    return doc.tobytes()


def test_valid_pdf_opens(parsing):
    doc = parsing.open_validated(pdf_bytes(2), "ok.pdf", parsing.cfg)
    assert doc.page_count == 2


def test_rejects_non_pdf(parsing):
    with pytest.raises(ValueError, match="not a PDF"):
        parsing.open_validated(b"this is not a pdf", "broken.pdf", parsing.cfg)


def test_rejects_corrupt_pdf(parsing):
    with pytest.raises(ValueError, match="corrupt"):
        parsing.open_validated(b"%PDF-1.7\n" + b"\x00garbage" * 50, "corrupt.pdf", parsing.cfg)


def test_rejects_oversized_file(parsing):
    with pytest.raises(ValueError, match="larger than"):
        parsing.open_validated(b"%PDF" + b"0" * (2 * 1024 * 1024), "big.pdf", parsing.cfg)


def page(n, text, ocr=False, figures="", tables=None):
    return {"page": n, "text": text, "ocr": ocr, "figures": figures, "tables": tables or []}


def test_chunks_track_pages_sections_and_kinds(parsing):
    body1 = "# Introduction\n\n" + "Sales forecasting matters for retail planning. " * 12
    body2 = "## Results\n\n" + "XGBoost reached the lowest RMSE on the test month. " * 12
    pages = [page(1, body1), page(2, body2, ocr=True, figures="A bar chart of RMSE per model.",
                                  tables=[{"summary": "RMSE per model", "markdown": "| a | b |\n|---|---|\n| 1 | 2 |"}])]
    chunks = parsing.chunk_document(pages, "doc", "paper.pdf", "Paper", parsing.cfg)

    text = [c for c in chunks if c.kind in ("text", "ocr")]
    assert {p for c in text for p in c.pages} == {1, 2}
    assert all(c.section == "Introduction" for c in text if c.pages == [1])
    assert all(c.kind == "ocr" for c in text if 2 in c.pages)          # OCR'd page marks its chunks
    assert any(c.section == "Results" for c in text if c.pages == [2])
    assert [c.kind for c in chunks if c.kind in ("figure", "table")] == ["figure", "table"]
    assert all(c.pages == [2] for c in chunks if c.kind in ("figure", "table"))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_spanning_two_pages(parsing):
    pages = [page(1, "Alpha sentence about the first page. " * 3), page(2, "Beta sentence about page two. " * 3)]
    chunks = parsing.chunk_document(pages, "doc", "a.pdf", "A", types.SimpleNamespace(chunk_size=2000, chunk_overlap=0))
    assert chunks[0].pages == [1, 2]


def test_find_markdown_tables(nb):
    text = "Intro\n| a | b |\n|---|---|\n| 1 | 2 |\nAfter\n| lonely | row |\n"
    tables = nb.find_markdown_tables(text)
    assert tables == ["| a | b |\n|---|---|\n| 1 | 2 |"]


def test_gradio_message_parts_are_flattened(nb):
    assert nb.msg_text("plain") == "plain"
    assert nb.msg_text([{"type": "text", "text": "Hello "}, {"type": "text", "text": "there"}]) == "Hello there"
    assert nb.msg_text(None) == ""
    chat = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}, {"role": "assistant", "content": "Hello"}]
    assert nb.clean_chat(chat) == [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
