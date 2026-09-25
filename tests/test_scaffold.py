"""Smoke tests: the package and every planned subpackage import, and the CLI responds."""

import importlib

import pytest

import weft
from weft import cli

SUBPACKAGES = [
    "weft.frontend",
    "weft.ir",
    "weft.hardware",
    "weft.mapping",
    "weft.engines",
    "weft.engines.analytical",
    "weft.engines.event",
    "weft.engines.functional",
    "weft.energy",
    "weft.noise",
    "weft.report",
]


def test_version_is_set():
    assert isinstance(weft.__version__, str) and weft.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    importlib.import_module(name)


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert weft.__version__ in capsys.readouterr().out


def test_cli_unimplemented_command_fails_cleanly():
    assert cli.main(["run"]) == 2
