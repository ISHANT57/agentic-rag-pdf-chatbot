"""Benchmark metric helpers: refusal detection, evidence-quote matching, keyword accuracy and token F1."""


def row(answer, status="answered"):
    return {"answer": answer, "status": status}


def test_refusal_by_status(nb):
    assert nb.is_refusal(row("anything", status="not_found"))


def test_short_uncited_refusal(nb):
    assert nb.is_refusal(row("The document does not mention the ARIMA order."))
    assert nb.is_refusal(row("I could not find this information in the uploaded document(s)."))


def test_partial_answer_with_citation_is_not_a_refusal(nb):
    assert not nb.is_refusal(row("XGBoost had the lowest RMSE, 0.807 [S2]. The validation month is not specified."))


def test_plain_answer_is_not_a_refusal(nb):
    assert not nb.is_refusal(row("The encoder has 6 identical layers [S1]."))


def test_match_key_ignores_markdown_and_html(nb):
    indexed = "**Encoder:** The encoder is composed of a stack of _N_ = 6 identical layers."
    quote = "The encoder is composed of a stack of N = 6 identical layers."
    assert nb.match_key(quote) in nb.match_key(indexed)
    assert nb.match_key("_β_ 1 = 0 _._ 9") == nb.match_key("β1 = 0.9")
    assert nb.match_key("ε = 10<sup>−9</sup>") == nb.match_key("ε = 10−9")


def test_match_key_handles_ligatures(nb):
    assert nb.match_key("ﬁne-tuning") == nb.match_key("fine tuning")


def test_keyword_numbers_match_whole_numbers_only(nb):
    assert nb.keyword_hit("N = 6 identical layers", ["6"]) == 1.0
    assert nb.keyword_hit("published in 2016", ["6"]) == 0.0
    assert nb.keyword_hit("a BLEU of 6.5", ["6"]) == 0.0
    assert nb.keyword_hit("a BLEU of 27.3.", ["27.3"]) == 1.0
    assert nb.keyword_hit("costs 3,611 dollars", ["3611"]) == 1.0


def test_keyword_number_words(nb):
    assert nb.keyword_hit("six layers", ["6"]) == 1.0
    assert nb.keyword_hit("6 layers", ["six"]) == 1.0


def test_keyword_all_must_match(nb):
    assert nb.keyword_hit("XGBoost won", ["xgboost", "0.878"]) == 0.0
    assert nb.keyword_hit("XGBoost won with RMSE 0.878 [S1]", ["xgboost", "0.878"]) == 1.0


def test_token_f1(nb):
    assert nb.token_f1("RMSE was 27.3.", "27.3") > 0
    assert nb.token_f1("same words here", "same words here") == 1.0
    assert nb.token_f1("nothing shared", "completely different") == 0.0
