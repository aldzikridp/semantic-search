# TASK-001: Project Scaffolding & Dependencies

> **Phase**: 1 | **Priority**: Critical | **Status**: ✅ Done
> **Depends on**: None
> **Blocks**: TASK-002, TASK-003, TASK-004, TASK-005, TASK-006, TASK-007, TASK-008, TASK-009

## Objective

Create the complete directory structure, dependency files, and package boilerplate so that `pip install -e .` succeeds and `semsearch --help` shows a placeholder.

## Files to Create/Modify

### 1. `requirements.txt`

Pin all dependencies per SPEC §2:

```text
langchain==0.2.16
langchain-core==0.2.41
langchain-postgres==0.0.17
langchain-community==0.2.16
langchain-text-splitters==0.2.4
langchain-openai==0.1.22
langchain-ollama==0.1.0
psycopg[binary]==3.2.1
pgvector==0.3.5
sqlalchemy==2.0.32
pydantic==2.8.2
pydantic-settings==2.4.0
typer==0.12.3
pymupdf==1.24.10
python-dotenv==1.0.1
jq==1.7.0
pytest==8.3.2
pytest-asyncio==0.23.8
testcontainers==4.8.1
pytest-cov
```

**Note**: HuggingFace local embeddings removed — rely entirely on API providers (OpenAI, OpenRouter, OpenAI-compatible, Ollama).

### 2. `pyproject.toml` (update existing)

Add:
- `[project]` section: `name = "semsearch"`, `version = "0.1.0"`, `requires-python = ">=3.11"`, `dependencies` (runtime only, not test deps)
- `[project.scripts]` → `semsearch = "semsearch.cli:app"`
- `[tool.pytest.ini_options]` → `testpaths = ["tests"]`
- `[tool.setuptools.packages.find]` → `where = ["src"]`

### 3. `.env.example`

Copy verbatim from SPEC §6.2. Contains all env var names with comments.

### 4. `scripts/init_db.sql`

Copy from SPEC §4.1 — SQL for creating extensions, role, and database.

### 5. Directory structure

```
src/semsearch/__init__.py          # empty, marks as package
src/semsearch/errors.py            # exception hierarchy (see TASK-002)
tests/__init__.py                  # empty
tests/conftest.py                  # placeholder with pass
```

### 6. `.gitignore` (update existing)

Ensure includes: `.venv/`, `__pycache__/`, `*.egg-info/`, `.env`, `.pytest_cache/`, `htmlcov/`, `.coverage`, `dist/`, `build/`

## Verification

- [ ] `pip install -e .` succeeds without errors
- [ ] `semsearch --help` shows help text (even if commands are stubs)
- [ ] `pytest --collect-only` finds no tests (empty test dir)
- [ ] All pinned versions resolve correctly

## Critical Notes

- Runtime dependencies in `pyproject.toml` must NOT include pytest/testcontainers
- Test dependencies go in `[project.optional-dependencies]` under `dev = [...]`
- `src/` layout requires `[tool.setuptools.packages.find] where = ["src"]`
