"""Weight-space repair arms: fine-tune the net on the repair set (with or without anchors), and a
PRDNN-style L-inf-minimal change to the last layer that makes every repair point right (an LP)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..model import MLP, from_torch, to_torch
from ..spec import Spec
from .data import ANCHOR_SEED, RepairData, training_points
from .metrics import decide, right

FT_STEPS, FT_LR, N_ANCHOR = 300, 1e-3, 2000
LP_MARGIN = 1e-3
LP_TOL_SHRINK = 0.999


def finetune(mlp: MLP, spec: Spec, X: np.ndarray, y: np.ndarray, seed: int = 0) -> MLP:
    """Full-batch Adam on the task loss (MSE, BCE on out - threshold, or cross-entropy)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    net = to_torch(mlp)
    opt = torch.optim.Adam(net.parameters(), lr=FT_LR)
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    if spec.output == "argmax":
        yt, loss_fn = torch.from_numpy(np.asarray(y)).long(), nn.CrossEntropyLoss()
    else:
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
        loss_fn = nn.MSELoss() if spec.output == "regression" else nn.BCEWithLogitsLoss()
    for _ in range(FT_STEPS):
        out = net(Xt)
        if spec.output == "argmax":
            loss = loss_fn(out, yt)
        else:
            loss = loss_fn(out[:, 0] - (spec.threshold if spec.output == "binary" else 0.0), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return from_torch(net)


def anchors(rd: Path, mlp: MLP, spec: Spec, data: RepairData, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Points that pin the net's behaviour: training-distribution points where the original net is right
    (labelled by the net), or, for dataset runs, uniform points over the box labelled by the net."""
    from experiments.tasks import get_task, uniform_sample

    task = get_task(Spec.load(Path(rd) / "spec.json").task)   # `spec` is the blind one
    rng = np.random.default_rng(ANCHOR_SEED)
    if data.flowers:
        X = uniform_sample(spec, rng, N_ANCHOR)
    elif data.whole_domain:
        X = data.X_heldout
    else:
        X = training_points(task, rng, 2 * N_ANCHOR)
    raw = mlp.forward(X.astype(np.float32)).astype(np.float64)
    label = decide(spec, raw) if spec.output != "regression" else raw[:, 0]
    if not data.flowers:
        keep = right(spec, label, task.target(X).astype(np.float64), tol)
        X, label = X[keep][:N_ANCHOR], label[keep][:N_ANCHOR]
    return X, label


def _hidden(mlp: MLP, X: np.ndarray) -> np.ndarray:
    return mlp.hidden(X)


def lp_repair(mlp: MLP, spec: Spec, X: np.ndarray, y: np.ndarray, tol: float) -> MLP | None:
    """min t s.t. |dW|, |db| <= t on the last layer and every point right; None when infeasible."""
    from scipy.optimize import linprog

    H = _hidden(mlp, X)
    H1 = np.hstack([H, np.ones((len(H), 1))])          # [h, 1] multiplies [dW_i, db_i]
    f = mlp.forward(np.asarray(X, dtype=np.float32)).astype(np.float64)
    k, m1 = f.shape[1], H1.shape[1]
    nv = k * m1 + 1                                     # dW|db per output row, then t

    def coef(out: int, sign: float) -> np.ndarray:
        A = np.zeros((len(H1), nv))
        A[:, out * m1:(out + 1) * m1] = sign * H1
        return A

    rows, rhs = [], []
    if spec.output == "regression":
        rows += [coef(0, 1.0), coef(0, -1.0)]
        tol_lp = LP_TOL_SHRINK * tol   # a little inside the tolerance, so float32 rounding stays within it
        rhs += [tol_lp - (f[:, 0] - y), tol_lp + (f[:, 0] - y)]
    elif spec.output == "binary":
        s = 2 * np.asarray(y) - 1
        score = f[:, 0] - spec.threshold
        g = LP_MARGIN * float(np.percentile(np.abs(score), 95))
        rows.append(-s[:, None] * coef(0, 1.0))
        rhs.append(s * score - g)
    else:
        top = np.sort(f, axis=1)
        g = LP_MARGIN * float(np.percentile(top[:, -1] - top[:, -2], 95))
        yi = np.asarray(y).astype(int)
        for c in range(k):
            A = np.zeros((len(H1), nv))
            for i in range(len(H1)):
                if yi[i] == c:
                    continue
                A[i, yi[i] * m1:(yi[i] + 1) * m1] = -H1[i]
                A[i, c * m1:(c + 1) * m1] = H1[i]
            keep = yi != c
            rows.append(A[keep])
            rhs.append((f[keep, yi[keep]] - f[keep, c]) - g)   # -(f_y - f_c) - (dW_y - dW_c).h <= -g
    box = np.zeros((2 * (nv - 1), nv))
    box[:nv - 1, :nv - 1], box[nv - 1:, :nv - 1] = np.eye(nv - 1), -np.eye(nv - 1)
    box[:, -1] = -1.0
    A_ub = np.vstack(rows + [box])
    b_ub = np.concatenate(rhs + [np.zeros(2 * (nv - 1))])
    c = np.zeros(nv)
    c[-1] = 1.0
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * nv, method="highs")
    if r.status != 0:
        return None
    delta = r.x[:-1].reshape(k, m1)
    W = (mlp.weights[-1].astype(np.float64) + delta[:, :-1]).astype(np.float32)
    b = (mlp.biases[-1].astype(np.float64) + delta[:, -1]).astype(np.float32)
    fixed = MLP(mlp.weights[:-1] + [W], mlp.biases[:-1] + [b], acts=mlp.acts)
    # the LP is solved in float64; on a net whose scores are tiny the margins can vanish in float32
    out = fixed.forward(np.asarray(X, dtype=np.float32)).astype(np.float64)
    pred = out[:, 0] if spec.output == "regression" else decide(spec, out)
    if not right(spec, pred, np.asarray(y, dtype=np.float64), tol * (1 + 1e-6) + 1e-9).all():
        return None
    return fixed
