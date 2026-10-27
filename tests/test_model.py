import re
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from nndc.model import UnsupportedModel, from_torch, load_npz, save_npz


def test_forward_matches_torch():
    torch.manual_seed(0)
    seq = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 2))
    mlp = from_torch(seq)
    X = np.random.default_rng(0).uniform(-3, 3, (100, 3)).astype(np.float32)
    np.testing.assert_allclose(mlp.forward(X), seq(torch.from_numpy(X)).detach().numpy(), rtol=1e-6, atol=1e-6)


def test_rejects_unsupported():
    with pytest.raises(UnsupportedModel, match="Tanh"):
        from_torch(nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)))
    with pytest.raises(UnsupportedModel):
        from_torch(nn.Sequential(nn.Linear(2, 3), nn.ReLU()))   # must end with Linear


def test_npz_roundtrip(tmp_path):
    mlp = from_torch(nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1)))
    save_npz(mlp, tmp_path / "w.npz")
    m2 = load_npz(tmp_path / "w.npz")
    assert all((a == b).all() for a, b in zip(mlp.weights, m2.weights))
    assert all((a == b).all() for a, b in zip(mlp.biases, m2.biases))


def test_model_module_imports_without_torch():
    src = Path("nndc/model.py").read_text()
    assert not re.search(r"^(import torch|from torch)", src, re.M)
