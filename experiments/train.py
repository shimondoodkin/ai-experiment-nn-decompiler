"""Train a small ReLU MLP on a task, deterministically on CPU."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from nndc.model import MLP, from_torch

from .tasks import Task, uniform_sample

BATCH = 512
N_QUALITY = 20_000


def _sample(task: Task, rng: np.random.Generator, n: int) -> np.ndarray:
    if task.sampler is not None:
        return task.sampler(rng, n)
    return uniform_sample(task.spec, rng, n)


def quality(mlp: MLP, task: Task, seed: int) -> dict:
    X = _sample(task, np.random.default_rng(1000 + seed), N_QUALITY)
    y = task.target(X).astype(np.float64)
    if task.spec.output == "argmax":
        acc = float((np.argmax(mlp.forward(X.astype(np.float32)), axis=1) == y).mean())
        return {"accuracy": acc, "learned": acc >= 0.95}
    out = mlp.forward(X.astype(np.float32))[:, 0].astype(np.float64)
    if task.spec.output == "binary":
        acc = float(((out > task.spec.threshold).astype(int) == y).mean())
        return {"accuracy": acc, "learned": acc >= 0.98}
    mse = float(((out - y) ** 2).mean())
    var = float(y.var())
    r2 = 1 - mse / var if var > 0 else float("nan")
    return {"rmse": mse ** 0.5, "r2": r2, "learned": r2 >= 0.99}


def train(task: Task, seed: int) -> tuple[MLP, dict]:
    if task.hidden is None:
        raise ValueError(f"task {task.name} is hand-only")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    sizes = [len(task.spec.inputs)] + task.hidden + [task.spec.n_outputs]
    mods: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        mods.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            mods.append(nn.ReLU())
    net = nn.Sequential(*mods)
    opt = torch.optim.Adam(net.parameters(), lr=task.lr)
    binary = task.spec.output == "binary"
    argmax = task.spec.output == "argmax"
    loss_fn = nn.CrossEntropyLoss() if argmax else nn.BCEWithLogitsLoss() if binary else nn.MSELoss()
    curve = []
    for step in range(task.steps):
        X = _sample(task, rng, BATCH)
        if argmax:
            loss = loss_fn(net(torch.from_numpy(X.astype(np.float32))), torch.from_numpy(task.target(X)).long())
        else:
            y = torch.from_numpy(task.target(X).astype(np.float32))
            out = net(torch.from_numpy(X.astype(np.float32)))[:, 0]
            loss = loss_fn(out - task.spec.threshold if binary else out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == task.steps - 1:
            curve.append([step, float(loss.item())])
    mlp = from_torch(net)
    q = quality(mlp, task, seed)
    log = {"seed": seed, "steps": task.steps, "lr": task.lr, "hidden": task.hidden, "batch": BATCH,
           "loss_curve": curve, "quality": {k: v for k, v in q.items() if k != "learned"},
           "learned": q["learned"]}
    return mlp, log
