"""Citation handling: [S#] tags in answers are mapped to the evidence list by code, not by the model."""


def ev(*pages):
    return [{"source": "paper.pdf", "pages": p, "section": "Results", "text": f"passage on page {p[0]}"} for p in pages]


def test_page_label(nb):
    assert nb.page_label([3]) == "p. 3"
    assert nb.page_label([3, 4, 5]) == "pp. 3-5"


def test_expand_cites_lists_and_ranges(nb):
    assert nb.expand_cites("a [S1, S3] b") == "a [S1][S3] b"
    assert nb.expand_cites("a [S2-S4] b") == "a [S2][S3][S4] b"
    assert nb.expand_cites("a [S2–4] b") == "a [S2][S3][S4] b"
    assert nb.expand_cites("single [S5] stays") == "single [S5] stays"
    assert nb.expand_cites(None) == ""


def test_render_answer_renumbers_cited_sources(nb):
    res = {"answer": "XGBoost was best [S3]. Its RMSE was 0.878 [S1].", "status": "answered",
           "evidence": ev([2], [4], [6])}
    out = nb.render_answer(res)
    # Only the cited sources are listed, renumbered in order of their original numbers.
    assert "[S2]" in out and "[S3]" not in out.split("**Sources**")[0]
    sources = out.split("**Sources**")[1]
    assert "p. 2" in sources and "p. 6" in sources and "p. 4" not in sources


def test_render_answer_drops_out_of_range_tags(nb):
    res = {"answer": "Claim [S9] and claim [S1].", "status": "answered", "evidence": ev([1])}
    out = nb.render_answer(res)
    assert "[S9]" not in out and "[S1]" in out


def test_render_answer_leaves_refusals_alone(nb):
    res = {"answer": "I could not find this information.", "status": "not_found", "evidence": ev([1])}
    assert nb.render_answer(res) == "I could not find this information."


def test_render_answer_marks_partial_answers(nb):
    res = {"answer": "Part one [S1].", "status": "partial", "evidence": ev([1])}
    assert "only partially" in nb.render_answer(res)


def test_cited_pages_reads_lists_and_ignores_unknown_tags(nb):
    res = {"answer": "See [S1, S2] and [S7].", "evidence": ev([1], [5, 6])}
    assert nb.cited_pages(res) == {1, 5, 6}
