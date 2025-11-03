# Repository Guidelines

## Project Structure & Module Organization
`main.py` loads `.env`, refreshes the Chroma stores in `chroma_policies/`, then launches the console chat workflow. Core retrieval and dialogue orchestration live in `chatbot_v3.py`, which defines store builders, session memory, and policy filters. Prebuilt embeddings sit in `kwdb/` and `categorydb/`, while source policy data resides in JSON files such as `FINAL_key_cat.json`. Utility scripts like `kwdb_create.py`, `categorydb_create.py`, and `fill_keyword_category.py` regenerate supporting datasets; keep them in sync with any schema changes.

## Build, Test, and Development Commands
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 main.py                  # run the chat loop
python3 kwdb_create.py           # rebuild keyword embeddings
python3 categorydb_create.py     # rebuild category embeddings
```
Re-run the data builders after modifying JSON inputs or when wiping `chroma_policies/` to force a clean vector store.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indents and line lengths ≤ 100. Use `snake_case` for functions and variables, `PascalCase` for classes, and keep module names lowercase (see `chatbot_v3.py`). Shared prompts and system templates live as multiline strings; document complex logic with concise comments only where flow or heuristics need justification. Keep I/O paths relative to the repo root to simplify sandboxed execution.

## Testing Guidelines
No automated tests ship today; introduce `pytest` under `tests/` with files named `test_*.py`. Favor fixture-driven checks that cover retrieval scoring, filtering heuristics (e.g., `is_policy_related_question`), and vector-store regeneration flows. Before submitting, run `pytest` locally and validate a short console chat to ensure embeddings were persisted.

## Commit & Pull Request Guidelines
Recent commits use short, imperative verbs (`Fix url issue`, `server_memory`); continue that style and scope each change narrowly. Reference issue IDs when available and include before/after context for data updates. Pull requests should summarize the intent, list any new scripts or configs, attach console transcripts for behavioral changes, and call out manual steps (rebuilding embeddings, refreshing `.env`).

## Configuration & Security Tips
Store secrets in `.env` (at minimum `OPENAI_API_KEY`) and keep the file out of version control. Document any new environment variable in both `.env.example` (if added) and this guide. When rotating embeddings, delete the corresponding Chroma directory and rerun the builders to avoid stale states.
