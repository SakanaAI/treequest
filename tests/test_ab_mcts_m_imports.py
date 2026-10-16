import re
import subprocess
import sys
import textwrap

import pytest

from treequest.algos.ab_mcts_m import _ab_mcts_m_imports

# Runs in a fresh interpreter and makes arviz's daily-warning stamp update fail the
# way it does when several processes import arviz at once
# (https://github.com/arviz-devs/arviz/issues/2530): another process renames the
# shared `daily_warning.tmp` first, so this process's own rename raises. With mode
# "always", the rename raises on every attempt instead.
_IMPORT_WHILE_LOSING_DAILY_WARNING_RACE = textwrap.dedent(
    """
    import pathlib
    import sys

    cache_dir, mode = sys.argv[1:]
    try:
        import platformdirs
    except ImportError:
        print("attempts=0")
        sys.exit(0)

    # Start from an empty arviz cache so that arviz writes its daily-warning stamp.
    platformdirs.user_cache_dir = lambda *args, **kwargs: cache_dir

    _replace = pathlib.Path.replace
    attempts = 0

    def _replace_after_another_process(self, target):
        global attempts
        if self.name != "daily_warning.tmp":
            return _replace(self, target)
        attempts += 1
        if mode == "always":
            raise FileNotFoundError(2, "No such file or directory", str(self), None, str(target))
        if attempts == 1:
            _replace(self, target)  # The other process's rename wins.
        return _replace(self, target)

    pathlib.Path.replace = _replace_after_another_process

    try:
        from treequest.algos.ab_mcts_m._ab_mcts_m_imports import _import

        _import.check()
    finally:
        print(f"attempts={attempts}")
    """
)


def _import_while_losing_daily_warning_race(tmp_path, mode):
    if not _ab_mcts_m_imports._import.is_successful():
        pytest.skip("ABMCTSM dependencies are not available")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _IMPORT_WHILE_LOSING_DAILY_WARNING_RACE,
            str(tmp_path),
            mode,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    match = re.search(r"attempts=(\d+)", result.stdout)
    assert match, result.stderr
    attempts = int(match.group(1))
    if attempts == 0:
        pytest.skip("arviz does not write a daily-warning stamp in this environment")
    return result, attempts


def test_import_succeeds_after_losing_arviz_daily_warning_race(tmp_path):
    result, _ = _import_while_losing_daily_warning_race(tmp_path, "once")

    assert result.returncode == 0, result.stderr


def test_import_gives_up_when_arviz_daily_warning_keeps_failing(tmp_path):
    result, attempts = _import_while_losing_daily_warning_race(tmp_path, "always")

    assert attempts > 1
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr


def test_unrelated_file_not_found_error_is_not_retried(monkeypatch):
    imported = []

    def import_module(name):
        imported.append(name)
        raise FileNotFoundError(2, "No such file or directory", "daily_warning/data.nc")

    monkeypatch.setattr(_ab_mcts_m_imports.importlib, "import_module", import_module)

    with pytest.raises(FileNotFoundError):
        _ab_mcts_m_imports._import_arviz()
    assert imported == ["arviz"]
