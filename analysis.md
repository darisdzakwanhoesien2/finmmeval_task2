Your current code is a solid analysis front-end, but not yet a research pipeline. Right now it helps you inspect the task well; to reach research-paper quality, you need to turn that inspection into reproducible measurement.

What the current code already gives you:
- [src/data_utils.py:33](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task2/src/data_utils.py:33) is your canonical loader, so every analysis should start from `load_dataset(split)`.
- [src/data_utils.py:62](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task2/src/data_utils.py:62) is the key research helper. `parse_prompt()` decomposes each giant prompt into `instructions`, `financial_statements`, `news_sections`, and `question`. That is exactly what you need for paper-style task analysis because it lets you analyze the task as structured evidence rather than one long string.
- [pages/1_Dataset_Explorer.py:12](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task2/pages/1_Dataset_Explorer.py:12) already supports descriptive analysis: split size, company coverage, prompt/question/answer length, and row-level browsing.
- [pages/3_Model_Playground.py:19](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task2/pages/3_Model_Playground.py:19) already supports controlled qualitative probing: pick one item, inspect the parsed evidence, and test a hosted model with either the raw prompt or the curated wrapper.
- [pages/2_Workflow_Notes.py:16](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task2/pages/2_Workflow_Notes.py:16) already maps your app to a thesis-style methodology, which is useful for the paper’s experimental design section.

How to use this code for research-quality analysis:
- Start with dataset characterization. Use `load_dataset()` plus `parse_prompt()` to produce tables for split size, unique companies, prompt length, answer length, and language coverage per example. This becomes your “Dataset Statistics” section.
- Build a task taxonomy. For each row, classify the question into categories like trend analysis, balance-sheet health, capital allocation, risk, segment performance, or evidence retrieval. Your explorer page already makes this manual coding practical because it exposes the question and gold answer side by side.
- Analyze evidence composition. Because `parse_prompt()` separates financial statements from multilingual news, you can quantify how much of each prompt is structured statement evidence versus multilingual narrative evidence, and whether Easy and Expert differ in where the answer is grounded.
- Use the model playground for qualitative case studies, not final evaluation. It is good for understanding failure modes, prompt behavior, and citation style, but paper-quality results need batch inference across all rows under fixed settings.
- Convert qualitative insights into hypotheses. Example: “Expert questions rely more heavily on multilingual news than Easy questions” or “curated prompts improve grounding over raw prompts.” Then test these in a batch script.

What is still missing for paper-quality work:
- A batch evaluation script. Right now you only have interactive inference in [pages/3_Model_Playground.py:168](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task2/pages/3_Model_Playground.py:168). You need a script that runs every row in both splits and saves predictions with `task_id`, split, model, prompt mode, temperature, max tokens, and timestamp.
- Persistent artifacts. For a paper, every run should save outputs to something like `artifacts/predictions/<model>/<split>.jsonl` plus a config file recording the exact parameters.
- Automatic metrics. Your notes mention ROUGE-1, grounding, provenance, and diagnostics, but the current code does not compute them yet. Add:
  - ROUGE-1 against `answer`
  - output length compliance
  - citation presence / evidence marker detection
  - optional grounding heuristics by checking whether quoted spans appear in the source prompt
- Error analysis. After scoring, manually review a stratified sample of strong and weak cases and label failure types such as unsupported claim, wrong financial comparison, missing multilingual evidence, over-reliance on one language, hallucinated quote, or incomplete answer.
- Statistical rigor. Report mean scores by split, by company, and by question type, then add bootstrap confidence intervals and paired comparisons between prompt variants or models.
- Reproducibility. Save the model ID, provider, token settings, and prompt mode from [src/hf_streaming.py:136](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task2/src/hf_streaming.py:136) and [pages/3_Model_Playground.py:131](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task2/pages/3_Model_Playground.py:131) for every experiment.

If you want a strong paper structure, I’d frame it like this:
1. Dataset analysis: what the task contains and why Easy vs Expert differ.
2. Experimental setup: model, prompt mode, inference settings, evaluation metrics.
3. Quantitative results: aggregate metrics plus split/company/question-type breakdowns.
4. Qualitative analysis: failure taxonomy and multilingual grounding case studies.
5. Ablations: raw prompt vs curated prompt, different HF models, maybe different max token budgets.

The biggest conceptual shift is this: your current app is excellent for exploration, but a paper needs evidence that is systematic, logged, and repeatable. The natural next step is to add one offline evaluation script that reuses `load_dataset()`, `parse_prompt()`, and `stream_chat_completion()` to generate a results table and artifact folder.

If you want, I can take your current code and build that evaluation harness next so the app and the paper workflow line up cleanly.