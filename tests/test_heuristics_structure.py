import numpy as np

from nndc.check import check_program
from nndc.dump import dump_program
from nndc.heuristics.bounds import prune_dead_and_linear
from nndc.heuristics.merge import merge_units, normalize
from nndc.model import MLP
from nndc.spec import InputSpec, Spec
from nndc.symnet import from_dump, to_program
from tests.helpers import hand


def test_dead_and_linear_units_removed():
    mlp, spec = hand("deadredundant")
    net, trace = prune_dead_and_linear(from_dump(dump_program(mlp, spec), spec), spec)
    names = [u.name for u in net.units if u.act == "relu"]
    assert "h1_1" not in [u.name for u in net.units] and "h1_0" not in names   # dead removed, always-on linearized
    assert check_program(to_program(net), mlp, spec).exact_pass and trace


def test_all_dead_gives_constant():
    mlp = MLP([np.array([[1.0]], np.float32), np.array([[3.0]], np.float32)],
              [np.array([-5.0], np.float32), np.array([0.25], np.float32)])
    spec = Spec(task="", output="regression", inputs=[InputSpec("x0", "x0", "float", 0, 1)])
    net, _ = prune_dead_and_linear(from_dump(dump_program(mlp, spec), spec), spec)
    assert to_program(net).strip().endswith("return 0.25")


def test_duplicates_merge_exactly():
    mlp, spec = hand("deadredundant")
    net, _ = merge_units(normalize(from_dump(dump_program(mlp, spec), spec)), spec, near=False)
    assert sum(u.act == "relu" for u in net.units) == 3          # h1_2 and h1_3 became one
    assert check_program(to_program(net), mlp, spec).exact_pass


def test_near_merge_is_approximate_but_passes_snap():
    mlp, spec = hand("absdiff_perturbed")
    net, _ = merge_units(normalize(from_dump(dump_program(mlp, spec), spec)), spec, near=True)
    assert check_program(to_program(net), mlp, spec).snap_pass
