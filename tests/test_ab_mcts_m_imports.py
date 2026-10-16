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


def _require_abmctsm_dependencies():
    if not _ab_mcts_m_imports._import.is_successful():
        pytest.skip("ABMCTSM dependencies are not available")


def _import_while_losing_daily_warning_race(tmp_path, mode):
    _require_abmctsm_dependencies()
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


# Runs in a fresh interpreter: while the main thread recovers from losing the
# daily-warning race, another thread imports arviz. That import is paused near the
# end of arviz's __init__ (where arviz creates a logging.StreamHandler) until the
# main thread retries, so the recovery happens while the import is in progress.
_IMPORT_IN_ANOTHER_THREAD_WHILE_RETRYING = textwrap.dedent(
    """
    import importlib
    import logging
    import pathlib
    import sys
    import threading

    try:
        import platformdirs
    except ImportError:
        print("race_simulated=False")
        sys.exit(0)

    cache_dir = pathlib.Path(sys.argv[1])
    platformdirs.user_cache_dir = lambda *args, **kwargs: str(cache_dir / "first")

    from treequest.algos.ab_mcts_m import _ab_mcts_m_imports as m

    # Forget arviz so that both threads import it again, and start from an empty
    # cache so that arviz writes its daily-warning stamp.
    for name in [n for n in list(sys.modules) if n.split(".")[0] == "arviz"]:
        del sys.modules[name]
    platformdirs.user_cache_dir = lambda *args, **kwargs: str(cache_dir / "second")

    main = threading.current_thread()
    lost = []
    other_paused = threading.Event()
    retrying = threading.Event()
    other_result = []

    _replace = pathlib.Path.replace

    def _replace_after_another_process(self, target):
        if self.name == "daily_warning.tmp" and not lost:
            lost.append(True)
            _replace(self, target)  # The other process's rename wins.
        return _replace(self, target)

    pathlib.Path.replace = _replace_after_another_process

    def _import_arviz_in_other_thread():
        try:
            import arviz  # noqa: F401

            other_result.append("ok")
        except Exception as e:
            other_result.append(repr(e))

    other = threading.Thread(target=_import_arviz_in_other_thread)

    class _PausingStreamHandler(logging.StreamHandler):
        def __init__(self, *args, **kwargs):
            if threading.current_thread() is other and not other_paused.is_set():
                other_paused.set()
                retrying.wait(60)
            super().__init__(*args, **kwargs)

    logging.StreamHandler = _PausingStreamHandler

    _path = m.Path

    def _start_other_import_after_losing(*args):
        if lost and other.ident is None:
            other.start()
            other_paused.wait(60)
        return _path(*args)

    m.Path = _start_other_import_after_losing

    _import_module = importlib.import_module

    def _resume_other_import_when_retrying(name, *args, **kwargs):
        if threading.current_thread() is main and lost:
            retrying.set()
        return _import_module(name, *args, **kwargs)

    importlib.import_module = _resume_other_import_when_retrying

    try:
        m._import_arviz()
    finally:
        retrying.set()
        if other.ident is not None:
            other.join(60)
        print(f"race_simulated={bool(lost)}")
        print(f"other_paused={other_paused.is_set()}")
        print(f"other_thread={other_result[0] if other_result else 'not finished'}")
    """
)


def test_retry_does_not_break_arviz_import_in_another_thread(tmp_path):
    _require_abmctsm_dependencies()
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_IN_ANOTHER_THREAD_WHILE_RETRYING, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if "race_simulated=False" in result.stdout:
        pytest.skip("arviz does not write a daily-warning stamp in this environment")

    output = result.stdout + result.stderr
    assert "other_paused=True" in result.stdout, output
    assert "other_thread=ok" in result.stdout, output
    assert result.returncode == 0, output


def test_unrelated_file_not_found_error_is_not_retried(monkeypatch):
    imported = []

    def import_module(name):
        imported.append(name)
        raise FileNotFoundError(2, "No such file or directory", "daily_warning/data.nc")

    monkeypatch.setattr(_ab_mcts_m_imports.importlib, "import_module", import_module)

    with pytest.raises(FileNotFoundError):
        _ab_mcts_m_imports._import_arviz()
    assert imported == ["arviz"]
