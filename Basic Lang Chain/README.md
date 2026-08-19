# Basic LangChain

A minimal Python project for experimenting with LangChain.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)

## Setup

Create the virtual environment and install the project dependencies:

```bash
uv sync
```

## Run

Start the example application with:

```bash
uv run python main.py
```

The current example prints:

```text
Hello from basic-lang-chain!
```

## Project Structure

```text
.
├── main.py          # Application entry point
├── pyproject.toml   # Project metadata and dependencies
└── README.md        # Project documentation
```

## Dependency

This project uses [LangChain](https://www.langchain.com/) to build applications powered by language models.
