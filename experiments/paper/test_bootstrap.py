"""Author source setup must work after a fresh clone without overwriting edits."""

from types import SimpleNamespace

import pytest

from . import bootstrap
from .common.safety import OutputGuard


@pytest.fixture
def author(tmp_path):
    item = {"path": str(tmp_path / "author"), "url": "https://example.invalid/author.git", "commit": "a" * 40}
    return item, OutputGuard()


def mock_git(monkeypatch, commit, dirty=False):
    commands = []
    monkeypatch.setattr(bootstrap, "run", lambda command: commands.append(list(map(str, command))))
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))

    def output(command, **kwargs):
        return (" M model.py\n" if dirty else "") if command[-2:] == ["status", "--porcelain"] else commit + "\n"

    monkeypatch.setattr(bootstrap.subprocess, "check_output", output)
    return commands


def test_setup_accepts_empty_directory_left_by_git_submodule_clone(author, monkeypatch):
    from pathlib import Path

    item, guard = author
    Path(item["path"]).mkdir()
    commands = mock_git(monkeypatch, item["commit"])
    bootstrap.prepare_author(item, guard)
    assert any("clone" in command and "--no-checkout" in command for command in commands)


def test_setup_refuses_nonempty_non_repository(author):
    from pathlib import Path

    item, guard = author
    path = Path(item["path"])
    path.mkdir()
    source = path / "user_source.py"
    source.write_text("preserve this file\n")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        bootstrap.prepare_author(item, guard)
    assert source.read_text() == "preserve this file\n"


def test_setup_reuses_existing_pin_without_network_or_checkout(author, monkeypatch):
    from pathlib import Path

    item, guard = author
    (Path(item["path"]) / ".git").mkdir(parents=True)
    commands = mock_git(monkeypatch, item["commit"])
    bootstrap.prepare_author(item, guard)
    assert not commands


def test_setup_does_not_overwrite_dirty_author_source(author, monkeypatch):
    from pathlib import Path

    item, guard = author
    (Path(item["path"]) / ".git").mkdir(parents=True)
    commands = mock_git(monkeypatch, item["commit"], dirty=True)
    with pytest.raises(RuntimeError, match="local edits"):
        bootstrap.prepare_author(item, guard)
    assert commands == []


def test_fresh_real_clone_checks_out_source_without_network(tmp_path):
    import subprocess

    source, destination = tmp_path / "upstream", tmp_path / "checkout"
    source.mkdir()
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    (source / "model.py").write_text("value = 1\n")
    subprocess.run(["git", "-C", str(source), "add", "model.py"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Bootstrap Test", "-c", "user.email=bootstrap@example.invalid",
                    "commit", "-m", "test source"], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    destination.mkdir()
    bootstrap.prepare_author({"path": str(destination), "url": str(source), "commit": commit}, OutputGuard())
    assert (destination / "model.py").read_text() == "value = 1\n"
    assert subprocess.check_output(["git", "-C", str(destination), "status", "--porcelain"], text=True) == ""
