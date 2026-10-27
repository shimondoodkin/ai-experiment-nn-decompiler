from dataclasses import replace

import numpy as np
import pytest

from experiments.grade import grade_task
from experiments.tasks import TASKS, expected_program, get_task
from experiments.train import train
from nndc.check import build_input_set


@pytest.mark.parametrize("name", list(TASKS))
def test_expected_formula_is_correct(name):
    t = get_task(name)
    g = grade_task(expected_program(t), t)
    assert g["error"] is None and g["recovered"], g


@pytest.mark.parametrize("name", [n for n, t in TASKS.items() if t.handcrafted])
def test_handcrafted_net_matches_target(name):
    t = get_task(name)
    mlp = t.handcrafted()
    X, _ = build_input_set(mlp, t.spec, seed=0)
    out = mlp.forward(X.astype(np.float32))[:, 0]
    tgt = t.target(X)
    if t.spec.output == "binary":
        off_tie = np.abs(out) > 1e-5   # boundary points sit on the decision line; float32 rounding decides ties
        assert (((out > 0).astype(int)) == tgt)[off_tie].mean() > 0.999
    elif name != "absdiff_perturbed":
        np.testing.assert_allclose(out, tgt, atol=0.01 if name == "sparse8" else 1e-4)


def test_specs_use_neutral_names():
    for t in TASKS.values():
        assert [i.name for i in t.spec.inputs] == [f"x{k}" for k in range(len(t.spec.inputs))]


def test_training_smoke():
    t = replace(get_task("absdiff"), steps=200)
    mlp, log = train(t, seed=0)
    assert mlp.n_inputs == 2 and "quality" in log and "learned" in log
    mlp2, _ = train(t, seed=0)
    assert np.array_equal(mlp.weights[0], mlp2.weights[0])   # deterministic
