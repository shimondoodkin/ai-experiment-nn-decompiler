"""Bring a refactoring attempt back from its blind folder, audit it, and re-check it independently."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from ..run import agent_attempts, evaluate_run, record_git
from ..spec import Spec
from .prepare import INDEX, load_index

PROJECTS = Path.home() / ".claude" / "projects"
PATH_KEYS = {"file_path", "path", "notebook_path"}


def mangle(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", str(Path(path).resolve()))


def find_transcripts(blind: Path, projects: Path = PROJECTS) -> list[Path]:
    d = Path(projects) / mangle(blind)
    return sorted(d.glob("*.jsonl")) if d.is_dir() else []


# a bare "/" or "~" is usually prose ("+-1 / small", "w0 ~ -w1") in heredoc'd notes, not a path
HARMLESS = {"/dev/null", "/", "~"}
HOMES = ("${HOME}", "$HOME", "~")


def _looks_like_path(tok: str) -> bool:
    if tok in HARMLESS:
        return False
    # "/tmp", "/etc/x", "/c/..." are POSIX paths; "/50" and "/d" are division in prose or code
    return bool(re.match(r"^[A-Za-z]:[\\/]", tok) or re.match(r"^/([A-Za-z_.][^/]{2,}|[A-Za-z_.][^/]*/)", tok)
                or tok.startswith(HOMES)
                or ".." in tok)


def _is_outside(tok: str, blind: Path) -> bool:
    t = tok
    for home in HOMES:
        if t == home or t.startswith((home + "/", home + "\\")):
            t = str(Path.home()) + t[len(home):]
            break
    m = re.match(r"^/([a-zA-Z])(/.*)?$", t)
    if m:   # Git Bash /c/... drive paths
        t = f"{m.group(1)}:/{(m.group(2) or '/')[1:]}"
    elif sys.platform == "win32" and (t == "/tmp" or t.startswith("/tmp/")):   # Git Bash /tmp
        t = tempfile.gettempdir() + t[4:]
    p = Path(t)
    if not p.is_absolute():
        p = blind / p
    try:
        return not p.resolve().is_relative_to(blind.resolve())
    except (OSError, ValueError):
        return True


def _strings(obj, key: str | None = None):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v, key)
    elif isinstance(obj, str):
        yield key, obj


def audit_transcripts(paths: list[Path], blind: Path) -> dict:
    res = {"transcript_found": False, "outside_paths": [], "model": None, "duration_s": None, "tokens": None}
    lines = []
    for p in paths:
        try:
            for raw in Path(p).read_text(encoding="utf-8").splitlines():
                try:
                    lines.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
        except OSError:
            continue
    if not lines:
        return res
    res["transcript_found"] = True
    outside: list[str] = []
    stamps = []
    tokens = {"input": 0, "output": 0}
    seen_ids = set()
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        if ts := ln.get("timestamp"):
            try:
                stamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except ValueError:
                pass
        msg = ln.get("message")
        if not isinstance(msg, dict):
            continue
        model = msg.get("model")
        if res["model"] is None and model and not str(model).startswith("<"):
            res["model"] = model
        usage = msg.get("usage")
        mid = msg.get("id")
        if isinstance(usage, dict) and (mid is None or mid not in seen_ids):
            seen_ids.add(mid)
            tokens["input"] += sum(int(usage.get(k) or 0) for k in
                                   ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
            tokens["output"] += int(usage.get("output_tokens") or 0)
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not (isinstance(block, dict) and block.get("type") == "tool_use"):
                continue
            for key, s in _strings(block.get("input", {})):
                toks = [s] if key in PATH_KEYS else re.split(r"[\s\"'`;|&<>()=,]+", s)
                for tok in toks:
                    tok = tok.rstrip(":.,;!?") if key not in PATH_KEYS else tok   # prose punctuation
                    if tok and _looks_like_path(tok) and _is_outside(tok, blind) and tok not in outside:
                        outside.append(tok)
    res["outside_paths"] = outside
    if stamps:
        res["duration_s"] = (max(stamps) - min(stamps)).total_seconds()
    res["tokens"] = tokens
    return res


def _heuristics_section(notes: str) -> str:
    m = re.search(r"^## Heuristics[ \t]*$(.*?)(?=^## |\Z)", notes, re.M | re.S)
    if m:
        return m.group(1).strip()
    return '(no "## Heuristics" section found)\n\n' + notes.strip()


def collect(blind: Path, index: Path = INDEX, heuristics: Path = Path("HEURISTICS.md"),
            projects: Path = PROJECTS, transcripts: list[Path] | None = None) -> tuple[Path, int | None]:
    blind = Path(blind)
    entry = load_index(index).get(blind.name)
    if entry is None:
        raise KeyError(f"{blind.name} is not in {index}; was it prepared from this repo?")
    rd = Path(entry["run"])
    agent = blind / "agent.py"
    if not agent.exists():
        raise FileNotFoundError(f"{agent} not found: the refactoring run has not written agent.py")
    data = agent.read_bytes()
    attempts = agent_attempts(rd)
    digest = hashlib.sha256(data).hexdigest()
    if any(hashlib.sha256(p.read_bytes()).hexdigest() == digest for _, p in attempts):
        return rd, None
    n = max((k for k, _ in attempts), default=0) + 1
    (rd / f"agent_{n}.py").write_bytes(data)
    for src, dst in (("agent_notes.md", f"agent_{n}_notes.md"), ("agent_checks.log", f"agent_{n}_checks.log")):
        if (blind / src).exists():
            shutil.copyfile(blind / src, rd / dst)
    trs = transcripts if transcripts is not None else find_transcripts(blind, projects)
    meta = {"blind_id": blind.name, "collected": datetime.now().isoformat(timespec="seconds"),
            **audit_transcripts(trs, blind)}
    (rd / f"agent_{n}_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    try:
        evaluate_run(rd)
    except Exception:
        # roll back, so a failed collect never leaves an attempt that breaks the run folder
        for f in rd.glob(f"agent_{n}*"):
            if re.match(rf"^agent_{n}[_.]", f.name):
                f.unlink()
        raise
    record_git(rd, f"agent_{n}")
    notes_path = blind / "agent_notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    spec = Spec.load(rd / "spec.json")
    section = (f"## {spec.task}/{rd.name} — agent_{n} ({blind.name}, {meta['collected'][:10]})\n\n"
               f"{_heuristics_section(notes)}\n\n")
    heuristics = Path(heuristics)
    if not heuristics.exists():
        heuristics.write_text("# Heuristics learned by Claude Code\n\nAppended by `nndc agent collect`, "
                              "one section per refactoring attempt.\n\n", encoding="utf-8")
    with open(heuristics, "a", encoding="utf-8") as f:
        f.write(section)
    return rd, n
