"""
Safe prompt loader for RepoTriage.

Reads prompts.yaml once at startup, validates the structure against
PromptConfig (a Pydantic v2 model), and caches the result. Fails fast
with a clear ConfigurationError if the file is missing, malformed,
or structurally invalid.

Security: uses yaml.safe_load exclusively — no arbitrary Python object
execution from YAML tags.
"""

import functools
import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from app.core.exceptions import ConfigurationError
from app.models.prompts import PromptConfig

logger = logging.getLogger(__name__)


def _load_and_validate(prompts_file: str) -> PromptConfig:
    """Load and validate prompts.yaml from the given path.

    This is the inner function called by the cached public loader.
    Separated to keep cache key logic simple.

    Raises:
        ConfigurationError: If the file is missing, not valid YAML,
            uses unsafe constructs, or fails Pydantic schema validation.
    """
    path = Path(prompts_file)

    if not path.exists():
        raise ConfigurationError(
            f"Prompts file not found: '{prompts_file}'. "
            "Ensure PROMPTS_FILE points to a valid prompts.yaml."
        )

    if not path.is_file():
        raise ConfigurationError(
            f"Prompts path is not a file: '{prompts_file}'."
        )

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            f"Could not read prompts file '{prompts_file}': {exc}"
        ) from exc

    # safe_load prevents arbitrary Python object execution from YAML tags.
    try:
        raw_data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Prompts file '{prompts_file}' contains invalid YAML: {exc}"
        ) from exc

    if not isinstance(raw_data, dict):
        raise ConfigurationError(
            f"Prompts file '{prompts_file}' must contain a YAML mapping "
            f"at the top level, got {type(raw_data).__name__}."
        )

    try:
        config = PromptConfig.model_validate(raw_data)
    except ValidationError as exc:
        # Format validation errors clearly without leaking file contents.
        error_count = exc.error_count()
        error_summary = "; ".join(
            f"{e['loc']}: {e['msg']}" for e in exc.errors()
        )
        raise ConfigurationError(
            f"Prompts file '{prompts_file}' failed schema validation "
            f"({error_count} error(s)): {error_summary}"
        ) from exc

    logger.info(
        "Loaded prompt configuration",
        extra={
            "prompts_file": prompts_file,
            "prompt_version": config.version,
            "few_shot_count": len(config.few_shot_examples),
        },
    )
    return config


@functools.lru_cache(maxsize=8)
def load_prompt_config(prompts_file: str) -> PromptConfig:
    """Load and cache the prompt configuration from a YAML file.

    Cached by file path — the same path returns the same object without
    re-reading the file. Different paths (e.g., test fixtures) each get
    their own cache entry.

    Call `load_prompt_config.cache_clear()` in tests to reset state.

    Args:
        prompts_file: Absolute or relative path to the prompts YAML file.

    Returns:
        A validated PromptConfig instance.

    Raises:
        ConfigurationError: If loading or validation fails.
    """
    return _load_and_validate(prompts_file)
