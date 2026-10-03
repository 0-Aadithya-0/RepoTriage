"""
Tests for the prompt loader and PromptConfig validation.

All tests use tmp_path fixtures to write YAML to a temporary file —
no network access, no modification of the real prompts.yaml.
The lru_cache is cleared before each test to ensure isolation.
"""

import pytest

from app.core.exceptions import ConfigurationError
from app.models.analysis import IssueCategory, IssuePriority
from app.services.prompt_loader import load_prompt_config

# Path to the actual prompts.yaml generated for the project
REAL_PROMPTS_FILE = "prompts.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_YAML = """\
version: "1.0.0"
system_role: "You are a helpful assistant."
task: "Analyze the GitHub issue below."
classification_rules: "Pick the best category from the list."
priority_rules: "Pick the best priority level from the list."
summary_rules: "Write one concise sentence."
few_shot_examples:
  - title: "App crashes on startup"
    body: "The app crashes immediately when launched on macOS 14."
    category: Bug
    priority: High
    summary: "Application crashes on startup on macOS 14 with no error message."
"""


@pytest.fixture(autouse=True)
def clear_prompt_cache() -> None:
    """Clear the lru_cache before every test to ensure isolation."""
    load_prompt_config.cache_clear()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestValidPromptLoading:
    """Tests for successfully loading and validating a correct prompts.yaml."""

    def test_loads_real_prompts_file(self) -> None:
        """The included prompts.yaml should load and validate correctly."""
        config = load_prompt_config(REAL_PROMPTS_FILE)
        assert config.version == "1.0.0"
        assert len(config.system_role) > 0
        assert len(config.task) > 0
        assert len(config.classification_rules) > 0
        assert len(config.priority_rules) > 0
        assert len(config.summary_rules) > 0
        assert len(config.few_shot_examples) >= 1

    def test_loads_minimal_valid_yaml(self, tmp_path: object) -> None:
        assert isinstance(tmp_path, type(tmp_path))
        import pathlib
        p = pathlib.Path(str(tmp_path)) / "prompts.yaml"
        p.write_text(VALID_YAML, encoding="utf-8")
        config = load_prompt_config(str(p))
        assert config.version == "1.0.0"
        assert config.few_shot_examples[0].title == "App crashes on startup"

    def test_few_shot_example_enums_are_typed(self, tmp_path: object) -> None:
        import pathlib
        p = pathlib.Path(str(tmp_path)) / "prompts.yaml"
        p.write_text(VALID_YAML, encoding="utf-8")
        config = load_prompt_config(str(p))
        example = config.few_shot_examples[0]
        assert example.category == IssueCategory.BUG
        assert example.priority == IssuePriority.HIGH

    def test_result_is_cached(self, tmp_path: object) -> None:
        """Calling the loader twice with the same path returns the same object."""
        import pathlib
        p = pathlib.Path(str(tmp_path)) / "prompts.yaml"
        p.write_text(VALID_YAML, encoding="utf-8")
        config1 = load_prompt_config(str(p))
        config2 = load_prompt_config(str(p))
        assert config1 is config2

    def test_different_paths_get_independent_cache_entries(
        self, tmp_path: object
    ) -> None:
        """Two different files each get their own cached result."""
        import pathlib
        tp = pathlib.Path(str(tmp_path))
        p1 = tp / "p1.yaml"
        p2 = tp / "p2.yaml"
        yaml1 = VALID_YAML.replace('"1.0.0"', '"1.0.0"')
        yaml2 = VALID_YAML.replace('"1.0.0"', '"2.0.0"')
        p1.write_text(yaml1, encoding="utf-8")
        p2.write_text(yaml2, encoding="utf-8")
        c1 = load_prompt_config(str(p1))
        c2 = load_prompt_config(str(p2))
        assert c1 is not c2

    def test_real_file_has_three_few_shot_examples(self) -> None:
        """The shipped prompts.yaml has the minimum required examples."""
        config = load_prompt_config(REAL_PROMPTS_FILE)
        assert len(config.few_shot_examples) >= 3

    def test_real_file_few_shot_summaries_within_bounds(self) -> None:
        """All few-shot summaries respect the 10-256 character constraint."""
        config = load_prompt_config(REAL_PROMPTS_FILE)
        for example in config.few_shot_examples:
            assert 10 <= len(example.summary) <= 256, (
                f"Summary out of bounds: '{example.summary}'"
            )

    def test_real_file_has_all_required_keys(self) -> None:
        config = load_prompt_config(REAL_PROMPTS_FILE)
        assert config.system_role
        assert config.task
        assert config.classification_rules
        assert config.priority_rules
        assert config.summary_rules


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------


class TestMissingFile:
    """Tests for missing or invalid file paths."""

    def test_missing_file_raises_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_prompt_config("/nonexistent/path/prompts.yaml")

    def test_error_message_contains_file_path(self) -> None:
        fake_path = "/no/such/file.yaml"
        with pytest.raises(ConfigurationError) as exc_info:
            load_prompt_config(fake_path)
        assert fake_path in str(exc_info.value)


# ---------------------------------------------------------------------------
# Invalid YAML syntax
# ---------------------------------------------------------------------------


class TestInvalidYAMLSyntax:
    """Tests for malformed YAML content."""

    def test_invalid_yaml_raises_configuration_error(
        self, tmp_path: object
    ) -> None:
        import pathlib
        p = pathlib.Path(str(tmp_path)) / "bad.yaml"
        p.write_text("key: [unclosed bracket", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="invalid YAML"):
            load_prompt_config(str(p))

    def test_yaml_list_at_root_is_rejected(self, tmp_path: object) -> None:
        """Top-level YAML lists are not a valid PromptConfig."""
        import pathlib
        p = pathlib.Path(str(tmp_path)) / "list.yaml"
        p.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="mapping"):
            load_prompt_config(str(p))

    def test_yaml_scalar_at_root_is_rejected(self, tmp_path: object) -> None:
        import pathlib
        p = pathlib.Path(str(tmp_path)) / "scalar.yaml"
        p.write_text("just a string\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="mapping"):
            load_prompt_config(str(p))


# ---------------------------------------------------------------------------
# Missing required keys
# ---------------------------------------------------------------------------


class TestMissingRequiredKeys:
    """Tests for YAML files with incomplete PromptConfig fields."""

    def _make_yaml(self, tmp_path: object, content: str) -> str:
        import pathlib
        p = pathlib.Path(str(tmp_path)) / "prompts.yaml"
        p.write_text(content, encoding="utf-8")
        return str(p)

    def test_missing_version_fails(self, tmp_path: object) -> None:
        yaml_text = "\n".join(
            line for line in VALID_YAML.splitlines()
            if not line.startswith("version:")
        )
        path = self._make_yaml(tmp_path, yaml_text)
        with pytest.raises(ConfigurationError, match="validation"):
            load_prompt_config(path)

    def test_missing_system_role_fails(self, tmp_path: object) -> None:
        yaml_text = "\n".join(
            line for line in VALID_YAML.splitlines()
            if not line.startswith("system_role:")
        )
        path = self._make_yaml(tmp_path, yaml_text)
        with pytest.raises(ConfigurationError, match="validation"):
            load_prompt_config(path)

    def test_missing_task_fails(self, tmp_path: object) -> None:
        yaml_text = "\n".join(
            line for line in VALID_YAML.splitlines()
            if not line.startswith("task:")
        )
        path = self._make_yaml(tmp_path, yaml_text)
        with pytest.raises(ConfigurationError, match="validation"):
            load_prompt_config(path)

    def test_missing_few_shot_examples_fails(self, tmp_path: object) -> None:
        yaml_text = """\
version: "1.0.0"
system_role: "You are a helpful assistant."
task: "Analyze the GitHub issue below."
classification_rules: "Pick the best category."
priority_rules: "Pick the best priority."
summary_rules: "One sentence."
"""
        path = self._make_yaml(tmp_path, yaml_text)
        with pytest.raises(ConfigurationError, match="validation"):
            load_prompt_config(path)

    def test_empty_few_shot_examples_list_fails(
        self, tmp_path: object
    ) -> None:
        yaml_text = """\
version: "1.0.0"
system_role: "You are a helpful assistant."
task: "Analyze the GitHub issue below."
classification_rules: "Pick the best category."
priority_rules: "Pick the best priority."
summary_rules: "One sentence."
few_shot_examples: []
"""
        path = self._make_yaml(tmp_path, yaml_text)
        with pytest.raises(ConfigurationError, match="validation"):
            load_prompt_config(path)

    def test_empty_version_string_fails(self, tmp_path: object) -> None:
        yaml_text = VALID_YAML.replace('version: "1.0.0"', 'version: ""')
        path = self._make_yaml(tmp_path, yaml_text)
        with pytest.raises(ConfigurationError, match="validation"):
            load_prompt_config(path)

    def test_empty_system_role_fails(self, tmp_path: object) -> None:
        yaml_text = VALID_YAML.replace(
            'system_role: "You are a helpful assistant."',
            'system_role: ""',
        )
        path = self._make_yaml(tmp_path, yaml_text)
        with pytest.raises(ConfigurationError, match="validation"):
            load_prompt_config(path)


# ---------------------------------------------------------------------------
# Invalid enum values in few-shot examples
# ---------------------------------------------------------------------------


class TestInvalidEnumValues:
    """Tests for few-shot examples with invalid enum values."""

    def test_invalid_category_in_example_fails(
        self, tmp_path: object
    ) -> None:
        import pathlib
        yaml_text = VALID_YAML.replace("category: Bug", "category: InvalidCategory")
        p = pathlib.Path(str(tmp_path)) / "prompts.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(ConfigurationError, match="validation"):
            load_prompt_config(str(p))

    def test_invalid_priority_in_example_fails(
        self, tmp_path: object
    ) -> None:
        import pathlib
        yaml_text = VALID_YAML.replace("priority: High", "priority: Urgent")
        p = pathlib.Path(str(tmp_path)) / "prompts.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(ConfigurationError, match="validation"):
            load_prompt_config(str(p))

    def test_summary_too_short_in_example_fails(
        self, tmp_path: object
    ) -> None:
        import pathlib
        yaml_text = VALID_YAML.replace(
            'summary: "Application crashes on startup on macOS 14 with no error message."',
            'summary: "Short"',
        )
        p = pathlib.Path(str(tmp_path)) / "prompts.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(ConfigurationError, match="validation"):
            load_prompt_config(str(p))


# ---------------------------------------------------------------------------
# Safe YAML loading
# ---------------------------------------------------------------------------


class TestSafeYAMLLoading:
    """Tests that unsafe YAML constructs are rejected."""

    def test_yaml_python_object_tag_is_rejected(
        self, tmp_path: object
    ) -> None:
        """!!python/object tags must not be executed."""
        import pathlib
        # This is the standard unsafe YAML payload — safe_load must reject it.
        unsafe_yaml = """\
version: !!python/object/apply:os.system ["echo UNSAFE"]
system_role: "role"
task: "task"
classification_rules: "rules"
priority_rules: "rules"
summary_rules: "rules"
few_shot_examples:
  - title: "t"
    body: "b"
    category: Bug
    priority: High
    summary: "A valid long enough summary for testing."
"""
        p = pathlib.Path(str(tmp_path)) / "unsafe.yaml"
        p.write_text(unsafe_yaml, encoding="utf-8")
        # yaml.safe_load raises ConstructorError for !!python/* tags
        with pytest.raises(ConfigurationError):
            load_prompt_config(str(p))
