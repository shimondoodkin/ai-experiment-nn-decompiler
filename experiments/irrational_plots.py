"""Plots of the irrational runs, seed 0 (report only): uv run --group plots python -m experiments.irrational_plots

1-D: the net, the true function and ast5's family program, kinks or unit centres marked. Binary 2-D: the net's
decision and the true boundary. 2-D regression: |net − true|. Writes <base>/plots/<task>_<variant>.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nndc.check import net_output
from nndc.lang import evaluate, parse_program
from nndc.run import load_run

from .tasks import get_task

VARIANTS = ("relu", "relu_wide", "sigmoid", "tanh")


def _family_src(rd: Path) -> str | None:
    p = rd / "irr_analysis.json"
    t = rd / "ast5_trace.json"
    if not p.exists() or not t.exists():
        return None
    fam = json.loads(p.read_text(encoding="utf-8"))["family"]
    rows = [r for r in json.loads(t.read_text(encoding="utf-8")).get("families", []) if r.get("src")]
    mine = [r for r in rows if r["family"] == fam] or rows
    return min(mine, key=lambda r: (not r.get("passed"), r.get("nodes") or 10**9))["src"] if mine else None


def plot_run(rd: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec, mlp = load_run(rd)
    task = get_task(spec.task)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    title = f"{spec.task} · {rd.parent.parent.name} · {rd.name}"
    if len(spec.inputs) == 1:
        i = spec.inputs[0]
        x = np.linspace(i.lo, i.hi, 1001)
        X = x[:, None]
        ax.plot(x, task.target(X), color="0.6", lw=3, label="true")
        ax.plot(x, net_output(mlp, spec, X)[0], color="C0", lw=1.5, label="net")
        src = _family_src(rd)
        if src:
            with np.errstate(all="ignore"):
                ax.plot(x, evaluate(parse_program(src, spec.arg_names), X), "--", color="C3", lw=1, label="ast5 family")
        a = json.loads((rd / "irr_analysis.json").read_text(encoding="utf-8")) if (rd / "irr_analysis.json").exists() else {}
        m = a.get("model", {})
        for k in m.get("kinks") or m.get("centres") or []:
            ax.axvline(k, color="C0", alpha=0.25, lw=0.8)
        ax.legend(fontsize=8)
    else:
        g = np.linspace(spec.inputs[0].lo, spec.inputs[0].hi, 301)
        h = np.linspace(spec.inputs[1].lo, spec.inputs[1].hi, 301)
        G, H = np.meshgrid(g, h)
        X = np.stack([G.ravel(), H.ravel()], axis=1)
        ext = (g[0], g[-1], h[0], h[-1])
        if spec.output == "binary":
            ax.imshow(net_output(mlp, spec, X)[0].reshape(G.shape), origin="lower", extent=ext, cmap="Blues", alpha=0.6)
            ax.contour(G, H, task.target(X).reshape(G.shape), levels=[0.5], colors="C3", linewidths=1)
        else:
            err = np.abs(net_output(mlp, spec, X)[0] - task.target(X)).reshape(G.shape)
            im = ax.imshow(err, origin="lower", extent=ext, cmap="viridis")
            fig.colorbar(im, ax=ax, label="|net − true|")
        ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs_irr")
    ap.add_argument("--seed", default="seed0")
    args = ap.parse_args()
    base = Path(args.base)
    (base / "plots").mkdir(parents=True, exist_ok=True)
    for v in VARIANTS:
        for rd in sorted((base / v).glob(f"*/{args.seed}")):
            out = base / "plots" / f"{rd.parent.name}_{v}.png"
            plot_run(rd, out)
            print(out.as_posix(), flush=True)


if __name__ == "__main__":
    main()
