# Code Graph RAG

Research-grade system for comparing **Vector**, **Vectorless**, and **Hybrid** RAG on codebase retrieval.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Ingest a repository
python main.py ingest /path/to/repo --name my_repo

# Query with vectorless retrieval
python main.py query "How does authentication work?" --strategy vectorless

# Query with LLM answer generation
python main.py query "What calls loginHandler?" --strategy hybrid --llm

# View repository stats
python main.py stats --repo my_repo
```

## Architecture

```
Git Repository → Tree-sitter Parser → Symbol Extractor → Code Graph
                                                       → BM25 Index
                                                       → Symbol Index
                                                       → Vector Index
                                    → Retriever (Vector | Vectorless | Hybrid)
                                    → Context Builder → LLM Reasoning
```

## Retrieval Strategies

| System | Pipeline | Embedding Required |
|:---|:---|:---|
| **A: Vector** | query → embed → cosine similarity | Yes |
| **B: Vectorless** | query → BM25 + symbol lookup + graph expansion | No |
| **C: Hybrid** | query → BM25 + vector + RRF + graph expansion | Yes |

## Evaluation

Run ablation studies with full IR metrics:

```bash
python scripts/run_experiment.py --config config/experiments.yaml
```

Metrics: Recall@K, Precision@K, MRR, nDCG, Hit@K, EM, F1, Hallucination Rate, Latency, Memory, Token Cost

## Project Structure

See `implementation_plan.md` for the full architecture and development roadmap.
