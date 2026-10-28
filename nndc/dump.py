"""Dump an MLP as a raw, readable program: one line per neuron, nothing simplified."""
from __future__ import annotations

import numpy as np

from .model import MLP
from .spec import Spec


def _fmt(c: float) -> str:
    return np.format_float_positional(np.float32(abs(c)), unique=True, trim="-")


def _affine(w: np.ndarray, b: float, names: list[str]) -> str:
    parts: list[str] = []
    for c, name in zip(w, names):
        if c == 0:
            continue
        sign = "-" if c < 0 else "+"
        term = f"{_fmt(c)} * {name}"
        parts.append((f"-{term}" if sign == "-" else term) if not parts else f" {sign} {term}")
    if b != 0:
        sign = "-" if b < 0 else "+"
        parts.append((f"-{_fmt(b)}" if sign == "-" else _fmt(b)) if not parts else f" {sign} {_fmt(b)}")
    return "".join(parts) or "0"


def _header(mlp: MLP, spec: Spec) -> list[str]:
    sizes = [f"{mlp.n_inputs} inputs"] + [str(W.shape[0]) for W in mlp.weights[:-1]] + [f"{mlp.n_outputs} outputs"]
    ins = ", ".join(f"{i.name}: {i.type} [{i.lo:g}, {i.hi:g}]" for i in spec.inputs)
    if spec.output == "regression":
        out = "regression: the returned value"
    elif spec.output == "binary":
        out = f"binary: class 1 if the returned value > {spec.threshold:g}, else class 0"
    else:
        out = f"argmax over {mlp.n_outputs} outputs: the returned class index"
    return [f"# dumped from a neural network: {' -> '.join(sizes)}",
            f"# inputs: {ins}",
            f"# output: {out}"]


def dump_program(mlp: MLP, spec: Spec) -> str:
    if spec.output != "argmax" and mlp.n_outputs != 1:
        raise ValueError(f"{spec.output} output needs exactly 1 network output, got {mlp.n_outputs}")
    names = spec.arg_names
    lines = _header(mlp, spec) + [f"def net({', '.join(names)}):"]
    n_layers = len(mlp.weights)
    for layer in range(n_layers - 1):
        W, b = mlp.weights[layer], mlp.biases[layer]
        act = mlp.acts[layer]
        new = [f"h{layer + 1}_{j}" for j in range(W.shape[0])]
        for j, name in enumerate(new):
            lines.append(f"    {name} = {act}({_affine(W[j], b[j], names)})  # layer {layer + 1} neuron {j}")
        names = new
    W, b = mlp.weights[-1], mlp.biases[-1]
    if spec.output == "argmax":
        outs = [f"o{k}" for k in range(W.shape[0])]
        for k, name in enumerate(outs):
            lines.append(f"    {name} = {_affine(W[k], b[k], names)}  # output {k}")
        lines.append(f"    return argmax({', '.join(outs)})")
    else:
        lines.append(f"    return {_affine(W[0], b[0], names)}")
    return "\n".join(lines) + "\n"
