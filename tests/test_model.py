import re
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from nndc.model import MLP, UnsupportedModel, activate, from_torch, load_npz, save_npz, to_torch


def test_forward_matches_torch():
    torch.manual_seed(0)
    seq = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 2))
    mlp = from_torch(seq)
    X = np.random.default_rng(0).uniform(-3, 3, (100, 3)).astype(np.float32)
    np.testing.assert_allclose(mlp.forward(X), seq(torch.from_numpy(X)).detach().numpy(), rtol=1e-6, atol=1e-6)


def test_rejects_unsupported():
    with pytest.raises(UnsupportedModel, match="GELU"):
        from_torch(nn.Sequential(nn.Linear(2, 3), nn.GELU(), nn.Linear(3, 1)))
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


def _acts_nets():
    torch.manual_seed(0)
    return [
        ("sigmoid", nn.Sequential(nn.Linear(3, 5), nn.Sigmoid(), nn.Linear(5, 2))),
        ("tanh", nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))),
        ("mixed", nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 4), nn.Sigmoid(), nn.Linear(4, 2))),
    ]


@pytest.mark.parametrize("name,seq", _acts_nets(), ids=[n for n, _ in _acts_nets()])
def test_forward_matches_torch_acts(name, seq):
    mlp = from_torch(seq)
    expect = {"sigmoid": ["sigmoid"], "tanh": ["tanh"], "mixed": ["relu", "sigmoid"]}[name]
    assert mlp.acts == expect
    X = np.random.default_rng(0).uniform(-3, 3, (100, 3)).astype(np.float32)
    np.testing.assert_allclose(mlp.forward(X), seq(torch.from_numpy(X)).detach().numpy(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name,seq", _acts_nets(), ids=[n for n, _ in _acts_nets()])
def test_torch_roundtrip_acts(name, seq):
    m = from_torch(seq)
    m2 = from_torch(to_torch(m))
    assert m2.acts == m.acts
    assert all((a == b).all() for a, b in zip(m.weights, m2.weights))
    assert all((a == b).all() for a, b in zip(m.biases, m2.biases))


@pytest.mark.parametrize("name,seq", _acts_nets()[1:], ids=["tanh", "mixed"])
def test_npz_roundtrip_acts(tmp_path, name, seq):
    m = from_torch(seq)
    save_npz(m, tmp_path / "w.npz")
    m2 = load_npz(tmp_path / "w.npz")
    assert m2.acts == m.acts
    assert all((a == b).all() for a, b in zip(m.weights, m2.weights))


def test_npz_relu_has_no_acts_key(tmp_path):
    mlp = from_torch(nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1)))
    save_npz(mlp, tmp_path / "w.npz")
    with np.load(tmp_path / "w.npz") as z:
        assert "acts" not in z.files
    assert load_npz(tmp_path / "w.npz").acts == ["relu"]


def test_sigmoid_extremes():
    with np.errstate(all="raise"):
        out = activate("sigmoid", np.array([-1e4, 0, 1e4], np.float32))
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.array([0, 0.5, 1], np.float32))


def test_mlp_rejects_bad_acts():
    W = [np.zeros((3, 2), np.float32), np.zeros((1, 3), np.float32)]
    b = [np.zeros(3, np.float32), np.zeros(1, np.float32)]
    with pytest.raises(ValueError):
        MLP(W, b, ["relu", "relu"])
    with pytest.raises(ValueError):
        MLP(W, b, ["gelu"])


def test_hidden_relu_and_no_hidden():
    rng = np.random.default_rng(1)
    W = [rng.normal(size=(4, 2)).astype(np.float32), rng.normal(size=(1, 4)).astype(np.float32)]
    b = [rng.normal(size=4).astype(np.float32), rng.normal(size=1).astype(np.float32)]
    m = MLP(W, b)
    assert m.acts == ["relu"]
    X = rng.uniform(-2, 2, (50, 2)).astype(np.float32)
    h = m.hidden(X)
    assert h.dtype == np.float64
    np.testing.assert_array_equal(h, np.maximum(m.preactivations(X)[-2].astype(np.float64), 0.0))
    lin = MLP([W[0]], [b[0]])
    np.testing.assert_array_equal(lin.hidden(X), X.astype(np.float64))
    assert lin.hidden(X).dtype == np.float64


def test_hidden_sigmoid_matches_forward_pass():
    rng = np.random.default_rng(2)
    W = [rng.normal(size=(5, 3)).astype(np.float32), rng.normal(size=(1, 5)).astype(np.float32)]
    b = [rng.normal(size=5).astype(np.float32), rng.normal(size=1).astype(np.float32)]
    m = MLP(W, b, ["sigmoid"])
    X = rng.uniform(-3, 3, (100, 3)).astype(np.float32)
    own = activate("sigmoid", m.preactivations(X)[0])     # the input to the last layer in the forward pass
    np.testing.assert_array_equal(m.hidden(X), own.astype(np.float64))
