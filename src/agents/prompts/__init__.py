# src/agents/prompts/__init__.py
"""Prompt template loader."""

from pathlib import Path

PROMPT_DIR = Path(__file__).parent


def load_prompt(name: str, **kwargs) -> str:
    """
    Load a prompt template and fill placeholders.

    Args:
        name: filename without extension (e.g., 'reception_system')
        **kwargs: placeholder values (e.g., mcp_sources_section="...")

    Returns:
        Formatted prompt string
    """
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    template = path.read_text()

    for key, value in kwargs.items():
        placeholder = "{" + key + "}"
        template = template.replace(placeholder, value)

    return template
