"""Black-box CLI tests using CliRunner."""

from typer.testing import CliRunner

from neuro_scan.cli import app

runner = CliRunner()


class TestAblateCommand:
    """Tests for the ablate command."""

    def test_ablate_missing_model(self):
        """ablate without --model exits with error."""
        result = runner.invoke(app, ["ablate"])
        assert result.exit_code != 0

    def test_ablate_help(self):
        """ablate --help shows usage."""
        result = runner.invoke(app, ["ablate", "--help"])
        assert result.exit_code == 0
        assert "model" in result.output.lower()


class TestLogitLensCommand:
    """Tests for the logit-lens command."""

    def test_logit_lens_missing_model(self):
        """logit-lens without --model exits with error."""
        result = runner.invoke(app, ["logit-lens"])
        assert result.exit_code != 0

    def test_logit_lens_help(self):
        """logit-lens --help shows usage."""
        result = runner.invoke(app, ["logit-lens", "--help"])
        assert result.exit_code == 0
        assert "model" in result.output.lower()


class TestAttentionCommand:
    """Tests for the attention command."""

    def test_attention_missing_model(self):
        """attention without --model exits with error."""
        result = runner.invoke(app, ["attention"])
        assert result.exit_code != 0

    def test_attention_help(self):
        """attention --help shows usage."""
        result = runner.invoke(app, ["attention", "--help"])
        assert result.exit_code == 0


class TestMapCommand:
    """Tests for the map command."""

    def test_map_missing_model(self):
        """map without --model exits with error."""
        result = runner.invoke(app, ["map"])
        assert result.exit_code != 0

    def test_map_help(self):
        """map --help shows usage."""
        result = runner.invoke(app, ["map", "--help"])
        assert result.exit_code == 0
        assert "model" in result.output.lower()


class TestPromptRepeatCommand:
    """Tests for the prompt-repeat command."""

    def test_prompt_repeat_missing_model(self):
        """prompt-repeat without --model exits with error."""
        result = runner.invoke(app, ["prompt-repeat"])
        assert result.exit_code != 0

    def test_prompt_repeat_help(self):
        """prompt-repeat --help shows usage."""
        result = runner.invoke(app, ["prompt-repeat", "--help"])
        assert result.exit_code == 0
        assert "repeat" in result.output.lower()


class TestProbesCommand:
    """Tests for the probes command."""

    def test_probes_lists_all(self):
        """probes command lists math, eq, json, custom."""
        result = runner.invoke(app, ["probes"])
        assert result.exit_code == 0
        assert "math" in result.output
        assert "eq" in result.output
        assert "json" in result.output
        assert "custom" in result.output

    def test_probes_shows_sample_counts(self):
        """probes command shows sample counts."""
        result = runner.invoke(app, ["probes"])
        assert "Samples:" in result.output


class TestVersionCommand:
    """Tests for the version command."""

    def test_version_format(self):
        """version output contains version number."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "neuro-scan" in result.output
        assert "0.2." in result.output
