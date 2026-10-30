import numpy as np
import pytest

from nndc.gate_view import groups, tolerance, unit_table
from nndc.model import MLP
from nndc.spec import InputSpec, Spec

SPEC2 = Spec(task="", output="regression",
             inputs=[InputSpec("x0", "x0", "float", -3, 3), InputSpec("x1", "x1", "float", -3, 3)])


def ring_units(n=6, r=1.0, k=4.0, mag=1.0, phase=0.0):
    """n (w0, w1, b, out) units whose gates sit at distance r around the origin, switching on outside."""
    out = []
    for i in range(n):
        t = phase + 2 * np.pi * i / n
        out.append((k * np.cos(t), k * np.sin(t), -k * r, mag))
    return out


def net(units, act="sigmoid"):
    a = np.array(units, dtype=np.float64)
    return MLP([a[:, :2].astype(np.float32), a[None, :, 3].astype(np.float32)],
               [a[:, 2].astype(np.float32), np.zeros(1, np.float32)], [act])


def ring_net(n=6, r=1.0, k=4.0, mag=1.0, extra=()):
    return net(ring_units(n, r, k, mag) + list(extra))


def _view(mlp, spec=SPEC2):
    return unit_table(spec, mlp), tolerance(spec, mlp), 6.0


def kinds(g):
    return {u: x["kind"] for x in g for u in x["units"]}


def test_ring_found():
    g = groups(*_view(ring_net()))
    assert [len(x["units"]) for x in g if x["kind"] == "ring"] == [6]


def test_two_rings_split_by_offset():
    g = groups(*_view(ring_net(extra=ring_units(n=5, r=2.5, phase=0.3))))
    assert sorted(len(x["units"]) for x in g if x["kind"] == "ring") == [5, 6]


def test_negative_offset_units_never_join_a_ring():
    units = [(4 * np.cos(t), 4 * np.sin(t), 4 * 0.7, 1.0) for t in (0.0, 1.6, 3.2, 4.8)]
    rows, tol, width = _view(net(units))
    assert all(r["offset"] == pytest.approx(-0.7, abs=1e-4) for r in rows)
    assert not any(x["kind"] == "ring" for x in groups(rows, tol, width))


def test_fewer_than_three_units_no_ring():
    g = groups(*_view(net(ring_units(n=3, r=1.0)[:2])))
    assert set(kinds(g).values()) == {"rest"}


def test_negligible_and_lines_first():
    tiny = (4.0, 0.0, -4.0, 1e-4)
    flat = (0.1, 0.1, 0.0, 1.0)            # |z| <= 0.6 on the box: 0.25 z + 0.5 is within 0.005
    g = kinds(groups(*_view(ring_net(extra=[tiny, flat]))))
    assert g[6] == "negligible" and g[7] == "lines"
    assert [g[u] for u in range(6)] == ["ring"] * 6


def test_same_line_pair():
    g = kinds(groups(*_view(net([(4.0, 0.0, -2.0, 1.0), (-4.0, 0.0, 2.0, -0.5)]))))
    assert g == {0: "same line", 1: "same line"}


def test_zones_of_a_steep_unit():
    (row,) = unit_table(SPEC2, net([(20.0, 0.0, 0.0, 1.0)]))
    assert row["zones"]["off"] == pytest.approx(0.5, abs=0.05) and row["zones"]["on"] == pytest.approx(0.5, abs=0.05)
    assert set(row["zones"]) == {"off", "low knee", "middle", "high knee", "on"}
    assert row["angle"] == pytest.approx(0.0) and row["width"] == pytest.approx(0.2) and row["sign"] == 1


def test_relu_units_two_zones():
    rows = unit_table(SPEC2, net([(1.0, 0.0, 0.0, 1.0), (0.0, -1.0, 0.5, -2.0)], act="relu"))
    assert all(set(r["zones"]) == {"off", "active"} for r in rows)
    assert rows[1]["sign"] == -1 and rows[1]["angle"] == pytest.approx(-90.0)


def test_tanh_net_read_as_sigmoid():
    rows = unit_table(SPEC2, net([(2.0, 0.0, -2.0, 1.0)], act="tanh"))
    assert rows[0]["offset"] == pytest.approx(1.0) and rows[0]["width"] == pytest.approx(1.0)   # sigmoid(2z): 4/4
    assert rows[0]["magnitude"] == pytest.approx(2 * (np.tanh(4) - np.tanh(-8)) / 2, rel=1e-3)


def test_two_layer_net_refused():
    W = [np.ones((2, 2), np.float32), np.ones((2, 2), np.float32), np.ones((1, 2), np.float32)]
    b = [np.zeros(2, np.float32), np.zeros(2, np.float32), np.zeros(1, np.float32)]
    with pytest.raises(ValueError):
        unit_table(SPEC2, MLP(W, b, ["sigmoid", "sigmoid"]))
