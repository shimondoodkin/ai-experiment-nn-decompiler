"""Model specification: inputs, output kind and checker tolerances.

Only stdlib imports: this module is copied into blind folders.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

InputType = Literal["float", "int", "byte", "bool"]
OutputKind = Literal["regression", "binary", "argmax"]

DISCRETE_TYPES = ("int", "byte", "bool")
HIDDEN_FIELDS = ("task", "override_reason")


@dataclass
class InputSpec:
    name: str          # neutral program name: "x0", "x1", ...
    real_name: str     # shown only in REPORT/README
    type: InputType
    lo: float
    hi: float


@dataclass
class Spec:
    task: str
    output: OutputKind
    inputs: list[InputSpec] = field(default_factory=list)
    n_outputs: int = 1
    threshold: float = 0.0
    atol: float = 1e-5
    rtol: float = 1e-4
    margin_frac: float = 0.05
    snap_tol_frac: float = 0.02
    override_reason: str | None = None

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> Spec:
        d = dict(d)
        d.setdefault("task", "")
        inputs = [InputSpec(i["name"], i.get("real_name", i["name"]), i["type"], i["lo"], i["hi"])
                  for i in d.pop("inputs")]
        return Spec(inputs=inputs, **d)

    def blind(self) -> dict:
        d = self.to_json()
        for k in HIDDEN_FIELDS:
            d.pop(k)
        for i in d["inputs"]:
            i.pop("real_name")
        return d

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def load(path: Path) -> Spec:
        return Spec.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def is_discrete(self) -> bool:
        return all(i.type in DISCRETE_TYPES for i in self.inputs)

    def domain_size(self) -> int:
        if not self.is_discrete():
            raise ValueError("domain_size() needs all inputs discrete")
        return math.prod(int(i.hi) - int(i.lo) + 1 for i in self.inputs)

    @property
    def arg_names(self) -> list[str]:
        return [i.name for i in self.inputs]
