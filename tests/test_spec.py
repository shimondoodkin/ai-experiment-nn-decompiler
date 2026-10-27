import json

from nndc.spec import InputSpec, Spec


def make():
    return Spec(task="absdiff", output="regression",
                inputs=[InputSpec("x0", "a", "float", -3, 3), InputSpec("x1", "b", "float", -3, 3)])


def test_roundtrip(tmp_path):
    s = make()
    s.save(tmp_path / "spec.json")
    assert Spec.load(tmp_path / "spec.json") == s


def test_blind_hides_task_and_real_names():
    b = make().blind()
    text = json.dumps(b)
    assert "absdiff" not in text and '"a"' not in text and "real_name" not in text
    assert [i["name"] for i in b["inputs"]] == ["x0", "x1"]


def test_blind_loads():
    s = Spec.from_json(make().blind())
    assert s.task == "" and s.inputs[0].name == "x0"


def test_domain_size():
    s = Spec(task="t", output="binary",
             inputs=[InputSpec("x0", "p", "bool", 0, 1), InputSpec("x1", "q", "byte", 0, 255)])
    assert s.is_discrete() and s.domain_size() == 2 * 256
    assert not make().is_discrete()
