# Research Agent

Local-first multi-agent CLI for research paper understanding, critical discussion, and literature intelligence.

## Requirements

- Python 3.11+

## Install

```bash
pip install -e ".[dev]"
```

## Configure

LLM requests go through [OpenRouter](https://openrouter.ai/) (OpenAI-compatible API).

```bash
# API key from https://openrouter.ai/keys
research config set api_key <your-openrouter-key>

# Default model: deepseek/deepseek-chat — change to any OpenRouter model id
research config set model anthropic/claude-3.5-sonnet
research config set language zh   # or: en (default), chinese, 中文

research config show
```

Configuration is stored at `~/.research-agent/config.yaml` (file mode `600`).

| Key | Default | Description |
|-----|---------|-------------|
| `api_key` | — | OpenRouter API key |
| `model` | `deepseek/deepseek-chat` | Model slug on OpenRouter |
| `base_url` | `https://openrouter.ai/api/v1` | API base (change only if self-hosting a proxy) |
| `app_title` | `Research Agent` | Sent as `X-Title` header to OpenRouter |
| `app_url` | `https://github.com/research-agent` | Sent as `HTTP-Referer` header |
| `language` | `en` | Agent reply language: `en` (English) or `zh` (简体中文) |

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
