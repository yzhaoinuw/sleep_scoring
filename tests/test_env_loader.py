"""Tests for local `.env` loading."""

import os
from pathlib import Path


def test_load_local_env_reads_key_value_pairs(monkeypatch):
    """A repo-local `.env` file should populate missing environment variables."""
    from app_src.env_loader import load_local_env

    env_dir = Path("test-artifacts") / "env_loader_basic"
    env_dir.mkdir(parents=True, exist_ok=True)
    env_path = env_dir / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# test env file",
                "OPENAI_API_KEY=sk-test-key",
                "export CHATGPT_MODEL_OVERRIDE='gpt-test-mini'",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CHATGPT_MODEL_OVERRIDE", raising=False)

    loaded_path = load_local_env(search_dirs=[env_dir])

    assert loaded_path == env_path.resolve()
    assert os.environ["OPENAI_API_KEY"] == "sk-test-key"
    assert os.environ["CHATGPT_MODEL_OVERRIDE"] == "gpt-test-mini"


def test_load_local_env_preserves_existing_values_by_default(monkeypatch):
    """Existing environment variables should win unless override=True is requested."""
    from app_src.env_loader import load_local_env

    env_dir = Path("test-artifacts") / "env_loader_preserve"
    env_dir.mkdir(parents=True, exist_ok=True)
    env_path = env_dir / ".env"
    env_path.write_text("OPENAI_API_KEY=sk-file-value\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-existing-value")

    loaded_path = load_local_env(search_dirs=[env_dir])

    assert loaded_path == env_path.resolve()
    assert os.environ["OPENAI_API_KEY"] == "sk-existing-value"
