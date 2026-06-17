# LoCoMo MEMORA Reproduction Report

Completed on 2026-06-17 local time.

## Scope

Implemented and evaluated:

- Full Context
- RAG
- MEMORA(S), semantic retriever
- MEMORA(P), policy retriever

## Dataset

Source: `data/raw/locomo_repo/data/locomo10.json`

Adversarial QA items were excluded. Final evaluation set:

| Category | N |
| --- | ---: |
| multi-hop | 282 |
| temporal | 321 |
| open-domain | 96 |
| single-hop | 841 |
| overall | 1540 |

All four prediction files contain 1,540 rows, 10 conversations, and 1,540
non-null LLM judge scores.

## Models

Configured in `configs/locomo.yaml`:

| Use | Model |
| --- | --- |
| Answer generation | `gpt-4.1-mini` |
| Memory construction | `gpt-4.1-mini` |
| Policy retrieval | `gpt-4.1-mini` |
| LLM judge | `gpt-4o-mini` |
| Embeddings | `text-embedding-3-small` |

## Commands Run

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

## Results

| Method | Category | BLEU | F1 | LLM judge | N |
| --- | --- | ---: | ---: | ---: | ---: |
| Full Context | multi-hop | 0.037 | 0.282 | 0.617 | 282 |
| Full Context | temporal | 0.043 | 0.220 | 0.550 | 321 |
| Full Context | open-domain | 0.011 | 0.089 | 0.396 | 96 |
| Full Context | single-hop | 0.092 | 0.309 | 0.869 | 841 |
| Full Context | overall | 0.067 | 0.272 | 0.727 | 1540 |
| RAG | multi-hop | 0.030 | 0.184 | 0.381 | 282 |
| RAG | temporal | 0.030 | 0.155 | 0.316 | 321 |
| RAG | open-domain | 0.014 | 0.100 | 0.328 | 96 |
| RAG | single-hop | 0.067 | 0.234 | 0.629 | 841 |
| RAG | overall | 0.049 | 0.200 | 0.500 | 1540 |
| MEMORA(S) | multi-hop | 0.029 | 0.203 | 0.447 | 282 |
| MEMORA(S) | temporal | 0.047 | 0.243 | 0.583 | 321 |
| MEMORA(S) | open-domain | 0.012 | 0.089 | 0.318 | 96 |
| MEMORA(S) | single-hop | 0.060 | 0.233 | 0.680 | 841 |
| MEMORA(S) | overall | 0.049 | 0.221 | 0.594 | 1540 |
| MEMORA(P) | multi-hop | 0.029 | 0.206 | 0.452 | 282 |
| MEMORA(P) | temporal | 0.047 | 0.245 | 0.586 | 321 |
| MEMORA(P) | open-domain | 0.013 | 0.094 | 0.344 | 96 |
| MEMORA(P) | single-hop | 0.061 | 0.236 | 0.685 | 841 |
| MEMORA(P) | overall | 0.049 | 0.223 | 0.601 | 1540 |

Machine-readable copy: `results/summary_table.json`.

## Memory Stores

Generated in `data/processed/memora_stores`.

The memory builder creates:

- turn segments from LoCoMo dialogue
- episodic summaries per segment
- atomic semantic entries
- links from semantic memories to episodic source nodes
- embeddings for semantic and episodic retrieval

The resulting stores cover all 10 conversations.

## Query Handling

Full Context:

1. Load the complete conversation transcript.
2. Pass the full transcript plus question to the answer model.
3. Evaluate against the gold answer.

RAG:

1. Chunk the transcript into about 500-token chunks.
2. Embed chunks.
3. Retrieve top 3 chunks by cosine similarity to the question.
4. Pass retrieved chunks plus question to the answer model.

MEMORA(S):

1. Embed the question.
2. Retrieve semantic abstractions and episodic cues from the memory store.
3. Expand selected memory nodes to source-linked context.
4. Pass compact memory context plus question to the answer model.

MEMORA(P):

1. Embed the question and seed an initial memory frontier.
2. Ask a policy model to select memory actions over a capped frontier.
3. Expand chosen memories through semantic and episodic links.
4. Fall back to deterministic semantic retrieval if policy JSON is invalid.
5. Pass selected memory context plus question to the answer model.

## Caveats

The paper does not provide code, full prompts, exact evaluator settings, or all
hyperparameters. This implementation therefore reconstructs the method from the
paper text. Exact paper-table values may require:

- the authors' original prompts
- exact LoCoMo evaluator prompt
- exact model snapshots
- exact memory consolidation thresholds
- exact policy retriever behavior
- any hidden preprocessing choices

The current result is best treated as a complete, inspectable reproduction
baseline that can now be tuned against the paper table.
