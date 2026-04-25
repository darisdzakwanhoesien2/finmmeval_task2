Workflow and Implementation
https://scite.ai/assistant/you-are-an-academic-researcher-writing-a-master-s-thesis-thesis-kZDePJ

\section{Workflow}

Step 1: Data ingestion and organization

Ingest 344 PolyFiQA Q&A instances, including English 10-K-like filings and multilingual signals (EN, ZH, JA, ES, EL).
Normalize and structure data into a unified schema suitable for cross-lingual retrieval and evaluation.
Step 2: Multi-document retrieval

Execute cross-lingual retrieval over English filings and parallel multilingual news using a dense embedding index with language-aware re-ranking.
Retrieve context chunks with passage IDs, language tags, and timestamps to support provenance.
Step 3: Cross-lingual prompt construction

Compose prompts that constrain outputs to ≤100 words, enforce citation requirements, and specify language-sensitive grounding instructions for Easy and Expert tiers.
Step 4: Model inference

Run the encoder–decoder LLM prompting with the retrieved evidence, generating concise, citation-backed answers across EN, ZH, JA, ES, and EL.
Step 5: ROUGE-1 evaluation

Compute per-language ROUGE-1 scores against expert references, followed by macro-averaged ROUGE-1 across all five languages.
Apply a length penalty to outputs exceeding 100 words to produce final scores.
Step 6: Grounding and provenance assessment

Evaluate grounding fidelity and provenance integrity by checking explicit citations and passage links to original sources.
Step 7: Reliability checks and diagnostics

Conduct bootstrap confidence intervals for ROUGE-1 and grounding metrics; perform cross-language coherence and memory/contagion diagnostics.
Step 8: Reporting and replication

Document evaluation setup, data splits, prompts, retrieval configurations, and scoring scripts to enable replication and cross-study comparability within CLEF-2026 FinMMEval-Lab Task 2 framework.

\subsection{Implementation}

Evaluation harness and environment

Implement a Python-based evaluation harness that processes all 344 PolyFiQA instances under uniform parameters across Easy and Expert tiers.
Maintain a containerized environment (e.g., Docker) detailing Python version, key libraries (transformers, datasets, numpy, scipy, nltk), and GPU configuration to ensure reproducibility.
Data handling

Load English filings and multilingual signals (EN, ZH, JA, ES, EL) with per-item provenance metadata and language tags.
Normalize inputs to a fixed schema and enforce a 100-word generation constraint during inference.
Model integration

Deploy a multilingual encoder–decoder backbone with retrieval-augmented generation (RAG) capabilities and an explicit prompt layer for citation constraints.
Integrate a provenance module that attaches passage IDs, language codes, and timestamps to each asserted claim in the generated outputs.
Prompt design

Implement a standardized prompt template that enforces ≤100 words, requires explicit citations, and includes language-aware grounding directives for Easy vs Expert tasks.
Retrieval pipeline

Use cross-lingual embeddings to retrieve top-k passages, with language-aware re-ranking and chunking into context windows compatible with the model’s input.
Evaluation metrics

Compute ROUGE-1 per language and macro-average.
Compute grounding fidelity and provenance integrity scores.
Compute cross-language coherence metrics and calibration/uncertainty estimates per language.
Apply length penalty for outputs exceeding 100 words during final scoring.
Reproducibility and governance

Log seeds, prompts, retrieval settings, and model versions; store evaluation artifacts with time-stamped metadata.
Release evaluation scripts and configuration files to facilitate replication and cross-lab comparability within CLEF-2026 FinMMEval-Lab Task 2 framework.
Deliverables

Reproducible evaluation pipeline, language-specific ROUGE-1 results, grounding and provenance diagnostics, and a comprehensive replication package suitable for external validation.