import pytest

from experiments.grade import grade_task
from experiments.tasks import get_task
from nndc.check import check_program
from nndc.dump import dump_program
from nndc.heuristics.merge import normalize
from nndc.heuristics.snap import nice, snap_net
from nndc.symnet import from_dump, to_program
from tests.helpers import hand


@pytest.mark.parametrize("v, tol, out", [(0.98, 0.05, 1.0), (-1.03, 0.06, -1.0), (0.124, 0.006, 0.125),
                                          (0.0013, 0.01, 0.0), (0.37, 0.001, 0.37), (0.198, 0.004, 0.2)])
def test_nice(v, tol, out):
    assert nice(v, tol) == pytest.approx(out)


def test_snap_perturbed_absdiff_to_unit_weights():
    mlp, spec = hand("absdiff_perturbed")
    net, trace = snap_net(normalize(from_dump(dump_program(mlp, spec), spec)), spec)
    src = to_program(net)
    assert check_program(src, mlp, spec).snap_pass and grade_task(src, get_task("absdiff_perturbed"))["recovered"]


def _body(src):
    return src.split("def net")[1].split("\n", 1)[1]


def test_small_weights_dropped_only_when_combined_effect_small():
    mlp, spec = hand("sparse8")   # 6 weights of 0.0005 on [0,1]: combined 0.003 -> dropped
    assert "x0" not in _body(to_program(snap_net(from_dump(dump_program(mlp, spec), spec), spec)[0]))
    mlp, spec = hand("sum8")      # 8 weights of 0.02 on [0,255]: combined 40.8 -> kept
    body = _body(to_program(snap_net(from_dump(dump_program(mlp, spec), spec), spec)[0]))
    assert all(f"x{k}" in body for k in range(8))
