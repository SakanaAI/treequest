# Development setup
## Installation for Development
To install extra dependencies for development, we need to specify all extras and all groups as:

```bash
# Install uv if you don't have it yet
brew install uv

# clone repository
git clone https://github.com/SakanaAI/treequest.git
cd treequest

# uv sync creates .venv and insatall all the dependencies
uv sync --all-extras --all-groups
```

## Adding pre-commit hook
```
uv run pre-commit install
```

## Running test
```
uv run pytest tests -n auto
```

## Formatting
```
uv run ruff format
```