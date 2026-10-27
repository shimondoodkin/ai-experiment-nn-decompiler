import json

import numpy as np
import pytest

from experiments.defects import defective_target, make_defect
from experiments.tasks import get_task, uniform_sample
from nndc.run import build_run
from nndc.spec import Spec


@pytest.mark.parametrize("name", ["tax", "absdiff", "median3"])
def test_defect_box_is_ten_percent(name):
    t = get_task(name)
    df = make_defect(t, 0)
    span = np.array([i.hi - i.lo for i in t.spec.inputs])
    vol = np.prod(np.array(df.box_hi) - np.array(df.box_lo)) / np.prod(span)
    assert abs(vol - 0.1) < 1e-9
    assert all(i.lo <= a < b <= i.hi for i, a, b in zip(t.spec.inputs, df.box_lo, df.box_hi))


def test_regression_offset_is_twenty_percent_of_output_range():
    t = get_task("tax")
    y = t.target(uniform_sample(t.spec, np.random.default_rng(0), 20_000))
    assert make_defect(t, 0).offset == pytest.approx(0.2 * (y.max() - y.min()))


def test_defective_target_differs_only_inside():
    t = get_task("absdiff")
    df = make_defect(t, 0)
    X = uniform_sample(t.spec, np.random.default_rng(1), 5000)
    diff = defective_target(t, df)(X) != t.target(X)
    assert diff.any() and (diff == df.inside(X)).all()


def test_binary_defect_flips_labels():
    t = get_task("window")
    df = make_defect(t, 0)
    X = uniform_sample(t.spec, np.random.default_rng(1), 5000)
    inside = df.inside(X)
    assert df.kind == "flip" and (defective_target(t, df)(X)[inside] == 1 - t.target(X)[inside]).all()


def test_xor_flips_one_point():
    t = get_task("xor")
    df = make_defect(t, 0)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    assert (defective_target(t, df)(X) != t.target(X)).sum() == 1


def test_build_run_bug_keeps_true_task(tmp_path):
    rd = build_run("xor", 0, tmp_path, bug=True)
    assert Spec.load(rd / "spec.json").task == "xor" and (rd / "defect.json").exists()
    log = json.loads((rd / "train_log.json").read_text())
    assert log["defect"]["kind"] == "flip" and log["bug_learned"] > 0.5
