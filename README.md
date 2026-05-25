# Research Agent

Local-first multi-agent CLI for research paper understanding, critical discussion, and literature intelligence.

## Requirements

- Python 3.11+

## Install

```bash
pip install -e ".[dev]"
```

## Configure

```bash
research config set api_key <your-deepseek-key>
research config show
```

Configuration is stored at `~/.research-agent/config.yaml` (file mode `600`).

## Commands

```bash
research --help
research config set api_key sk-xxx
research config show
research read arxiv:2301.12345   # E1.5
research discuss                  # E1.5
```

## Development

```bash
ruff check src tests
ruff format --check src tests
mypy src
pytest
```

## Planning

See [planning/PLAN.md](planning/PLAN.md) and [planning/architecture.md](planning/architecture.md).
