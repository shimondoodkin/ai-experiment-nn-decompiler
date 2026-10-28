"""experiments/equivalence.py: the copy command never deletes the repo or anything around it."""
import pytest

import experiments.equivalence as eq


@pytest.mark.parametrize("where", ["repo", "inside", "parent"])
def test_copy_refuses_destinations_that_hold_the_repo(where, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the guard let a delete through")

    monkeypatch.setattr(eq.shutil, "rmtree", boom)   # a regressed guard must fail, never delete
    monkeypatch.setattr(eq.shutil, "copytree", boom)
    dst = {"repo": eq.REPO, "inside": eq.REPO / "runs", "parent": eq.REPO.parent}[where]
    with pytest.raises(SystemExit, match="refusing"):
        eq.main(["copy", "--copy", str(dst)])


def test_copy_makes_a_copy_elsewhere(tmp_path, monkeypatch):
    src = tmp_path / "src"
    for root in eq.ROOTS:
        (src / root / "a").mkdir(parents=True)
        (src / root / "a" / "f.txt").write_text("x")
    monkeypatch.setattr(eq, "REPO", src)
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "stale").write_text("old")
    assert eq.main(["copy", "--copy", str(dst)]) == 0
    assert (dst / "runs_bugs" / "a" / "f.txt").read_text() == "x" and not (dst / "stale").exists()
