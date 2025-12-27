# Project Title

Project description

## Installation

### Requirements

- [UV](https://docs.astral.sh/uv/)
- Project name in [pyproject.toml](pyproject.toml) file must be the same as your module's name in `src` folder.
- [Pytorch Lightning](https://lightning.ai/docs/pytorch/latest/) knowledge.

### Initialize the project:

```bash
  uv sync
```

## Running the project

To run the project locally, run the following command:

```bash
  uv run python -m src.lightning_template
```

## Tests, linting and formatting

```bash
  uv run pytest
```

```bash
  uvx ruff check . --fix
```

```bash
  uvx ruff format .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
