"""End to end: train, save, load, dump, check and refactor a sigmoid net and a tanh net."""
from experiments.tasks import get_task
from experiments.train import train
from nndc.ast_tool import refactor
from nndc.check import check_program
from nndc.dump import dump_program
from nndc.model import load_npz, save_npz
from nndc.spec import Spec


def test_sigmoid_and_tanh_end_to_end(tmp_path):
    task = get_task("window")
    spec = Spec.from_json(task.spec.blind())
    for act in ["sigmoid", "tanh"]:
        mlp, _ = train(task, 0, act=act)
        save_npz(mlp, tmp_path / f"{act}.npz")
        mlp = load_npz(tmp_path / f"{act}.npz")
        assert set(mlp.acts) == {act}
        dump = dump_program(mlp, spec)
        assert check_program(dump, mlp, spec).exact_pass, act
        src, _ = refactor(dump, spec, mlp)
        assert check_program(src, mlp, spec).snap_pass, (act, src)
