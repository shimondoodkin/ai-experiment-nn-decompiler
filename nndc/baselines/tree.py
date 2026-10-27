"""B3: decision-tree distillation of the net, the smallest tree in a leaf sweep that passes the checker."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..run import load_run, record_git
from ..spec import Spec
from .common import net_samples, smallest_passing, write_info

LEAF_SWEEP = [2, 4, 8, 16, 32, 64, 128, 256, 512]


def tree_to_program(tree, spec: Spec) -> str:
    """A fitted sklearn tree as nested `if x_i <= t:` / `else:` returning the leaf value or class."""
    t = tree.tree_
    classify = hasattr(tree, "classes_")
    if classify and spec.output == "binary":
        assert 0 <= spec.threshold < 1, "binary leaves return 1/0"

    def leaf(node: int) -> str:
        if not classify:
            return repr(float(t.value[node][0][0]))
        return repr(int(tree.classes_[int(np.argmax(t.value[node][0]))]))

    def emit(node: int, depth: int) -> list[str]:
        pad = "    " * depth
        if t.children_left[node] == -1:
            return [f"{pad}return {leaf(node)}"]
        cond = f"x{int(t.feature[node])} <= {float(t.threshold[node])!r}"
        return ([f"{pad}if {cond}:"] + emit(int(t.children_left[node]), depth + 1)
                + [f"{pad}else:"] + emit(int(t.children_right[node]), depth + 1))

    return f"def net({', '.join(spec.arg_names)}):\n" + "\n".join(emit(0, 1)) + "\n"


def run_tree(rd: Path) -> dict:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    t0 = time.perf_counter()
    spec_full, mlp = load_run(rd)
    spec = Spec.from_json(spec_full.blind())
    X, _, dec = net_samples(spec, mlp)
    cands, leaves = [], []
    for k in LEAF_SWEEP:
        model = (DecisionTreeRegressor if spec.output == "regression" else DecisionTreeClassifier)(
            max_leaf_nodes=k, random_state=0).fit(X, dec)
        cands.append(tree_to_program(model, spec))
        leaves.append(int(model.get_n_leaves()))
    best, rows = smallest_passing(cands, mlp, spec)
    pick = best if best is not None else len(cands) - 1
    (Path(rd) / "tree.py").write_text(cands[pick], encoding="utf-8")
    info = {"leaves": leaves[pick], "passed": best is not None,
            "sweep": [{"max_leaf_nodes": k, "leaves": n, **r} for k, n, r in zip(LEAF_SWEEP, leaves, rows)],
            "seconds": round(time.perf_counter() - t0, 2)}
    write_info(rd, "tree", info)
    record_git(rd, "tree")
    return info
