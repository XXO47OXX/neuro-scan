"""Smoke tests — critical path validation, <10s total."""

import pytest

pytestmark = pytest.mark.smoke


def test_import():
    """Package imports without error."""
    import neuro_scan

    assert neuro_scan is not None


def test_version():
    """Version string is well-formed."""
    from neuro_scan import __version__

    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_cli_help():
    """--help returns 0 and contains usage info."""
    from typer.testing import CliRunner

    from neuro_scan.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "neuro-scan" in result.output.lower() or "Usage" in result.output


def test_cli_probes():
    """probes command lists built-in probes."""
    from typer.testing import CliRunner

    from neuro_scan.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["probes"])
    assert result.exit_code == 0
    assert "math" in result.output
    assert "eq" in result.output
    assert "json" in result.output


def test_cli_version():
    """version command outputs version number."""
    from typer.testing import CliRunner

    from neuro_scan.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.2." in result.output
