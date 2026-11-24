# EvalForge (scratch build): eval harness for a telecom-FAQ RAG assistant

Corpus (44 Vi FAQ pages) -> chunking -> local embeddings -> retrieval -> answer -> evaluation -> CI gate.
Embeddings run locally (`all-MiniLM-L6-v2`), so retrieval evaluation and its CI gate need **no API key**.
Answer generation, the LLM judge and the red team call an LLM and need a working key (or a local
OpenAI-compatible server, see `.env.example`).

## Where your step files went
| Your step file | Production module |
|---|---|
| `step1_load_corpus.py` | `evalkit/index.py` -> `load_documents` |
| `step2c_chunk_all.py` | `evalkit/chunking.py` -> `fixed` |
| `step3c_index.py` | `evalkit/index.py` -> `embed`, `Retriever.from_chunks` |
| `step4a_search.py` | `evalkit/index.py` -> `Retriever.search` |
| `step4b_metrics.py` | `evalkit/metrics.py`, `evalkit/run_eval.py` |

## Commands (run from this folder, venv active)
| What | Command | Needs key? |
|---|---|---|
| Unit tests | `pytest -q` | no |
| Build index | `python -m evalkit.index` | no |
| Retrieval eval | `python -m evalkit.run_eval` | no |
| Compare chunking | `python -m evalkit.compare` | no |
| Gate | `python -m evalkit.gate` (`--update-baseline` to accept scores) | no |
| Ask one question | `python -m evalkit.rag "How do I check my balance?"` | yes |
| Answers + judge scores | `python -m evalkit.run_eval --answers` (`--limit 10` for a cheap trial) | yes |
| Judge calibration | `python -m evalkit.judge sample` -> fill 50 labels -> `python -m evalkit.judge calibrate` | yes |
| Red team | `npx promptfoo@latest redteam run -c promptfoo/promptfooconfig.yaml -o results/redteam.json` | yes |
| Red-team gate | `python -m evalkit.gate_redteam results/redteam.json results/custom_redteam.json` | no |

For the full generation gate, add `faithfulness` and `answer_relevancy` floors to `thresholds.json` after
your first `--answers` run.

## Results so far (local embeddings, 99-question golden set, 85 answerable)
| Setup | chunks | recall@5 | MRR |
|---|---|---|---|
| fixed 800/100 (default) | 57 | 0.988 | 0.830 |
| fixed 400/50 | 97 | 0.976 | 0.834 |
| recursive 400/50 | 87 | 0.976 | 0.847 |
| token 100/10 | 79 | 0.965 | 0.837 |

These differences are small. With 85 questions, one question moving one rank changes MRR by about 0.005,
so do not claim an improvement from a gap this size without a larger golden set or a repeated paired test.

## CI
`.github/workflows/eval.yml`: the `retrieval` job runs tests, rebuilds the index, evaluates and gates on every pull
request. The `redteam` job needs an `OPENAI_API_KEY` repository secret. The gate only blocks merging if branch
protection requires the `eval-gate` check.

Chunking strategies adapted from the RAG Fitness Agent tutorial by Nisarg Kadam (MIT licence).
