# MEMORA LoCoMo Reproduction

This repo implements an end-to-end LoCoMo reproduction harness for the paper
**MEMORA: A Harmonic Memory Representation Balancing Abstraction and Specificity**.

Implemented methods:

- Full Context
- RAG
- MEMORA(S), semantic retriever
- MEMORA(P), policy retriever

## Setup

The local virtual environment is already created at `.venv`.

```powershell
cd "e:\Causal Dynamics Lab\memora_repro"
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Data

The official LoCoMo repository was cloned under `data/raw/locomo_repo`, and the
run uses `data/raw/locomo_repo/data/locomo10.json`.

The loader excludes adversarial questions and evaluates 1,540 QA items across
10 conversations:

- multi-hop: 282
- temporal: 321
- open-domain: 96
- single-hop: 841

## Run Commands

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\inspect_locomo.py
.\.venv\Scripts\python.exe scripts\build_memories.py
.\.venv\Scripts\python.exe scripts\run_full_context.py
.\.venv\Scripts\python.exe scripts\run_rag.py
.\.venv\Scripts\python.exe scripts\run_memora_s.py
.\.venv\Scripts\python.exe scripts\run_memora_p.py
.\.venv\Scripts\python.exe scripts\evaluate_results.py --with-llm-judge
```

## Current Results

Final artifacts are in `results/`.

| Method | Overall BLEU | Overall F1 | Overall LLM judge |
| --- | ---: | ---: | ---: |
| Full Context | 0.067 | 0.272 | 0.727 |
| RAG | 0.049 | 0.200 | 0.500 |
| MEMORA(S) | 0.049 | 0.221 | 0.594 |
| MEMORA(P) | 0.049 | 0.223 | 0.601 |

See `results/summary_table.json` and `results/reproduction_report.md` for the
full category-level table and notes.

Full prediction dumps are generated locally and intentionally ignored by git.
Small examples are committed under `results/samples/`.

Processed MEMORA stores are also generated locally and ignored because they are
large. A truncated structural example is committed under
`data/processed/samples/`.

## Notes

This is a faithful-from-paper implementation, not an official code release.
Prompts, memory consolidation details, and the policy retriever had to be
inferred from the paper. Exact paper numbers may require the authors' hidden
prompts, hyperparameters, evaluator prompt, model versions, and retrieval policy.
