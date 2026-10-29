# Benchmark report

* **Hardware**: Tesla T4 · **LLM**: `qwen3.5:9b` · **Judge**: `gemma3:4b` · **Embeddings**: `BAAI/bge-m3` · **Reranker**: `BAAI/bge-reranker-v2-m3`
* **Test set**: 36 questions over 4 PDFs (88 pages in total)
* **Peak GPU memory**: ingestion 9.5 GB, answering 9.5 GB (T4 limit: 15 GB)
* **Ingestion**: 408s total, 4.63s per page (includes figure descriptions)

## End-to-end results
|                          |   naive_rag |   advanced_rag |   agentic_rag |
|:-------------------------|------------:|---------------:|--------------:|
| judge_correctness        |       0.875 |          0.917 |         0.931 |
| correctness_answerable   |       0.891 |          0.938 |         0.953 |
| keyword_acc              |       0.786 |          0.929 |         0.929 |
| token_f1                 |       0.229 |          0.262 |         0.258 |
| semantic_sim             |       0.702 |          0.72  |         0.726 |
| faithfulness             |       0.985 |          0.973 |         0.954 |
| citation_page_acc        |       0.963 |          1     |         1     |
| refusal_acc_unanswerable |       1     |          1     |         1     |
| false_refusal_rate       |       0.031 |          0.031 |         0     |
| latency_mean_s           |       6.501 |          6.142 |        26.34  |
| latency_p50_s            |       4.888 |          5.045 |        12.378 |
| latency_p95_s            |      10.173 |          9.311 |       101.244 |
| llm_calls_mean           |       1.222 |          1     |         5.194 |
| output_tokens_mean       |      82.417 |         68.694 |       379.806 |

## 95% bootstrap confidence intervals
|                   | naive_rag            | advanced_rag         | agentic_rag          |
|:------------------|:---------------------|:---------------------|:---------------------|
| judge_correctness | 0.875 [0.778, 0.958] | 0.917 [0.847, 0.972] | 0.931 [0.847, 0.986] |
| keyword_acc       | 0.786 [0.643, 0.929] | 0.929 [0.821, 1.000] | 0.929 [0.821, 1.000] |
| faithfulness      | 0.985 [0.965, 1.000] | 0.973 [0.930, 1.000] | 0.954 [0.907, 0.996] |

## Paired difference (agentic minus baseline; significant = CI excludes 0)
| comparison                 |   mean_diff |   ci95_low |   ci95_high |   n | significant   |
|:---------------------------|------------:|-----------:|------------:|----:|:--------------|
| agentic_rag - naive_rag    |       0.056 |      0     |       0.125 |  36 | False         |
| agentic_rag - advanced_rag |       0.014 |     -0.028 |       0.069 |  36 | False         |

Judge vs keyword-check agreement: 0.89 · judge parse failures: 0.0%

## Judge correctness by category
| category      |   naive_rag |   advanced_rag |   agentic_rag |
|:--------------|------------:|---------------:|--------------:|
| factual       |       1     |          1     |         1     |
| multi_hop     |       0.688 |          0.875 |         0.812 |
| numeric_table |       1     |          0.938 |         1     |
| summary       |       0.75  |          0.875 |         1     |
| unanswerable  |       0.75  |          0.75  |         0.75  |

## Retrieval ablation (page-level)
|               |   hit@1 |   hit@3 |   hit@5 |   mrr@10 |   evidence_recall@5 |
|:--------------|--------:|--------:|--------:|---------:|--------------------:|
| bm25          |   0.893 |   0.929 |   0.964 |    0.918 |               0.875 |
| dense         |   0.786 |   0.929 |   0.964 |    0.872 |               0.804 |
| hybrid        |   0.857 |   0.929 |   1     |    0.907 |               0.893 |
| hybrid_rerank |   0.857 |   0.929 |   1     |    0.911 |               0.964 |

## Ingestion
| doc             |   pages |   chunks |   ocr_pages |   figure_pages |   parse_s |   index_s |   total_s |   s_per_page |
|:----------------|--------:|---------:|------------:|---------------:|----------:|----------:|----------:|-------------:|
| attention       |      15 |       61 |           0 |              5 |      55.5 |       1.4 |      75.9 |         5.06 |
| reference_paper |       6 |       27 |           0 |              2 |      28.5 |       1.7 |      38.7 |         6.45 |
| rag_paper       |      19 |       95 |           0 |              4 |      46.9 |       1.9 |      72.1 |         3.79 |
| nist_ai_rmf     |      48 |      156 |           0 |              6 |      99.8 |       5.2 |     220.8 |         4.6  |

**Best system on answerable-question correctness: `agentic_rag`.**
