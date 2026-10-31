# Benchmark report

* **Hardware**: Tesla T4 · **LLM**: `qwen3.5:9b` · **Judge**: `gemma3:4b` · **Embeddings**: `BAAI/bge-m3` · **Reranker**: `BAAI/bge-reranker-v2-m3`
* **Test set**: 36 questions over 4 PDFs (88 pages in total)
* **Peak GPU memory**: ingestion 9.6 GB, answering 9.6 GB (T4 limit: 15 GB)
* **Ingestion**: 446s total, 5.06s per page (includes figure descriptions)

## End-to-end results
|                          |   naive_rag |   advanced_rag |   agentic_rag |
|:-------------------------|------------:|---------------:|--------------:|
| judge_correctness        |       0.917 |          0.958 |         0.958 |
| correctness_answerable   |       0.906 |          0.953 |         0.953 |
| keyword_acc              |       0.786 |          0.929 |         0.929 |
| token_f1                 |       0.359 |          0.395 |         0.389 |
| semantic_sim             |       0.703 |          0.717 |         0.726 |
| faithfulness             |       0.969 |          0.973 |         0.954 |
| citation_page_acc        |       0.963 |          1     |         1     |
| refusal_acc_unanswerable |       1     |          1     |         1     |
| false_refusal_rate       |       0.031 |          0.031 |         0     |
| latency_mean_s           |       6.449 |          6.855 |        29.775 |
| latency_p50_s            |       5.198 |          5.57  |        15.305 |
| latency_p95_s            |      10.457 |         10.189 |        84.914 |
| llm_calls_mean           |       1     |          1     |         5.278 |
| output_tokens_mean       |      76.167 |         70.111 |       384.667 |

## 95% bootstrap confidence intervals
|                   | naive_rag            | advanced_rag         | agentic_rag          |
|:------------------|:---------------------|:---------------------|:---------------------|
| judge_correctness | 0.917 [0.833, 0.986] | 0.958 [0.917, 1.000] | 0.958 [0.903, 1.000] |
| keyword_acc       | 0.786 [0.643, 0.929] | 0.929 [0.821, 1.000] | 0.929 [0.821, 1.000] |
| faithfulness      | 0.969 [0.929, 1.000] | 0.973 [0.930, 1.000] | 0.954 [0.907, 0.996] |

## Paired difference (agentic minus baseline; significant = CI excludes 0)
| comparison                 |   mean_diff |   ci95_low |   ci95_high |   n | significant   |
|:---------------------------|------------:|-----------:|------------:|----:|:--------------|
| agentic_rag - naive_rag    |       0.042 |     -0.014 |       0.097 |  36 | False         |
| agentic_rag - advanced_rag |       0     |     -0.042 |       0.042 |  36 | False         |

Judge vs keyword-check agreement: 0.89 · judge parse failures: 0.0%

## Judge correctness by category
| category      |   naive_rag |   advanced_rag |   agentic_rag |
|:--------------|------------:|---------------:|--------------:|
| factual       |       1     |          1     |         1     |
| multi_hop     |       0.688 |          0.875 |         0.812 |
| numeric_table |       1     |          1     |         1     |
| summary       |       0.875 |          0.875 |         1     |
| unanswerable  |       1     |          1     |         1     |

## Retrieval ablation (page-level)
|               |   hit@1 |   hit@3 |   hit@5 |   mrr@10 |   evidence_recall@5 |
|:--------------|--------:|--------:|--------:|---------:|--------------------:|
| bm25          |   0.893 |   0.929 |   0.964 |    0.918 |               0.875 |
| dense         |   0.786 |   0.929 |   0.964 |    0.872 |               0.804 |
| hybrid        |   0.857 |   0.929 |   1     |    0.907 |               0.911 |
| hybrid_rerank |   0.857 |   0.929 |   1     |    0.911 |               0.964 |

## Ingestion
| doc             |   pages |   chunks |   ocr_pages |   figure_pages |   parse_s |   index_s |   total_s |   s_per_page |
|:----------------|--------:|---------:|------------:|---------------:|----------:|----------:|----------:|-------------:|
| attention       |      15 |       61 |           0 |              5 |      87.2 |       2.8 |     111.3 |         7.42 |
| reference_paper |       6 |       27 |           0 |              2 |      32.2 |       1.9 |      45.4 |         7.57 |
| rag_paper       |      19 |       95 |           0 |              4 |      58.3 |       2.6 |      87.9 |         4.63 |
| nist_ai_rmf     |      48 |      156 |           0 |              6 |     146.9 |       3.6 |     201.1 |         4.19 |

**Best system on answerable-question correctness: `advanced_rag`.**
