"""Shared test helpers (handcrafted nets with blind specs, random nets, whitespace normalization)."""
import numpy as np

from experiments.tasks import get_task
from nndc.model import MLP
from nndc.spec import Spec


def hand(name: str) -> tuple[MLP, Spec]:
    t = get_task(name)
    return t.handcrafted(), Spec.from_json(t.spec.blind())


def random_mlp(sizes: list[int], seed: int) -> MLP:
    rng = np.random.default_rng(seed)
    return MLP([rng.normal(size=(o, i)).astype(np.float32) for i, o in zip(sizes, sizes[1:])],
               [rng.normal(size=o).astype(np.float32) for o in sizes[1:]])


def normalize_ws(s: str) -> str:
    return " ".join(s.split())
