import pytest

from experiments.grade import grade_task
from experiments.tasks import get_task
from nndc.ast_tool import refactor
from nndc.check import check_program
from nndc.dump import dump_program
from nndc.spec import InputSpec, Spec
from tests.helpers import random_mlp


@pytest.mark.parametrize("name", ["absdiff", "absdiff_perturbed", "deadredundant", "max", "ramp", "triangle",
                                  "window", "neareq", "xor", "sparse8", "sum8"])
def test_refactor_handcrafted_recovers(name):
    t = get_task(name)
    mlp = t.handcrafted()
    spec = Spec.from_json(t.spec.blind())
    src, trace = refactor(dump_program(mlp, spec), spec, mlp)
    assert check_program(src, mlp, spec).snap_pass and grade_task(src, t)["recovered"], (name, src)
    assert trace["winner"] and trace["candidates"]


def test_refactor_requires_blind_spec():
    t = get_task("absdiff")
    mlp = t.handcrafted()
    with pytest.raises(ValueError, match="blind"):
        refactor(dump_program(mlp, t.spec), t.spec, mlp)


def test_two_layer_and_argmax_nets_do_not_crash():
    for sizes, kind in [([3, 6, 6, 1], "regression"), ([2, 4, 3], "argmax")]:
        mlp = random_mlp(sizes, seed=3)
        spec = Spec(task="", output=kind, n_outputs=sizes[-1],
                    inputs=[InputSpec(f"x{i}", f"x{i}", "float", -1, 1) for i in range(sizes[0])])
        src, trace = refactor(dump_program(mlp, spec), spec, mlp)
        assert check_program(src, mlp, spec).snap_pass
