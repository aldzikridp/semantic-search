# semsearch

Semantic search service over local documents using LangChain and PostgreSQL + pgvector.

## Quick Start

```bash
# 1. Clone & install
git clone <repo-url> && cd semantic-search
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,test]"

# 2. Set up PostgreSQL (one-time)
psql -U postgres -f scripts/init_db.sql
cp .env.example .env   # edit as needed

# 3. Run
semsearch --help
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
