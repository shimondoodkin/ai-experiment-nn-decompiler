from nndc.baselines.regions import regions_program, run_regions
from nndc.check import check_program
from nndc.run import build_run, evaluate_run
from nndc.spec import InputSpec, Spec
from tests.helpers import hand, random_mlp


def test_absdiff_regions_exact():
    mlp, spec = hand("absdiff")
    src, info = regions_program(mlp, spec)
    assert info["regions"] >= 2 and check_program(src, mlp, spec).exact_pass


def test_iris_like_argmax_regions_exact():
    mlp = random_mlp([2, 6, 3], seed=4)
    spec = Spec(task="", output="argmax", n_outputs=3,
                inputs=[InputSpec("x0", "x0", "float", -1, 1), InputSpec("x1", "x1", "float", -1, 1)])
    src, _ = regions_program(mlp, spec)
    assert check_program(src, mlp, spec).exact_pass


def test_two_hidden_layers_exact():
    mlp = random_mlp([2, 5, 4, 1], seed=1)
    spec = Spec(task="", output="regression",
                inputs=[InputSpec("x0", "x0", "float", -2, 2), InputSpec("x1", "x1", "float", -2, 2)])
    src, _ = regions_program(mlp, spec)
    assert check_program(src, mlp, spec).exact_pass


def test_budget_exceeded_gives_no_program():
    mlp = random_mlp([3, 16, 16, 1], seed=0)
    spec = Spec(task="", output="regression",
                inputs=[InputSpec(f"x{i}", f"x{i}", "float", -3, 3) for i in range(3)])
    src, info = regions_program(mlp, spec, max_regions=4)
    assert src is None and info["budget_exceeded"]


def test_run_regions_writes_version(tmp_path):
    rd = build_run("window", None, tmp_path)
    info = run_regions(rd)
    assert not info["budget_exceeded"] and evaluate_run(rd)["versions"]["regions"]["check"]["exact_pass"]
