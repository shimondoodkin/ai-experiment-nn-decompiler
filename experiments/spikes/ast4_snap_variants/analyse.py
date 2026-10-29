"""Compare route policies on the spike grids. Tool size = min(other passing routes in ast3_trace, kink policy)."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "out"
TOLS = (0.05, 0.02, 0.01, 0.005, 0.0)
ORDER = [(m, off, tol) for m in range(1, 9) for off in (0.02, 0.1) for tol in TOLS]


def first(rows, pick):
    """First (m, off, tol) cell in the route's order where pick() returns a row."""
    by = {}
    for r in rows:
        by.setdefault((r["m"], r["off"], r["tol"]), []).append(r)
    for key in ORDER:
        hit = pick(by.get(key, []))
        if hit is not None:
            return hit
    return None


def var(name):
    return lambda cell: next((r for r in cell if r["var"] == name and r["pass"]), None)


def try_all(cell):
    ok = [r for r in cell if r["pass"]]
    return min(ok, key=lambda r: r["nodes"]) if ok else None


def best_in_m(rows):
    ok = [r for r in rows if r["pass"]]
    if not ok:
        return None
    m = min(r["m"] for r in ok)
    return min((r for r in ok if r["m"] == m), key=lambda r: r["nodes"])


def global_min(rows):
    ok = [r for r in rows if r["pass"]]
    return min(ok, key=lambda r: r["nodes"]) if ok else None


POLICIES = {
    "ast3": lambda rows: first(rows, var("ast3")),
    "both(~ast2)": lambda rows: first(rows, var("both")),
    "try4/level": lambda rows: first(rows, try_all),
    "best-in-m": best_in_m,
    "grid-min": global_min,
}


def other_routes(rd):
    t = json.loads((rd / "ast3_trace.json").read_text())
    ok = [c["nodes"] for c in t["candidates"] if c["route"] != "kink-merge" and c["check"]["snap_pass"]]
    return min(ok) if ok else None


def nodes_of(rd, name):
    res = json.loads((rd / "results.json").read_text())
    v = res["versions"][name] if "versions" in res else res[name]
    return v.get("nodes") or v.get("size", {}).get("nodes")


print(f"{'run':20} {'other':>5} {'ast2':>5} " + " ".join(f"{p:>11}" for p in POLICIES))
tot = {p: [0, 0, 0] for p in POLICIES}  # vs ast2: smaller, same, larger
for f in sorted(OUT.glob("*.json")):
    rows = json.loads(f.read_text())
    rel = f.stem.replace("runs_sigmoid_", "runs_sigmoid/").replace("_seed", "/seed")
    rd = REPO / rel
    if rows is None:
        continue
    oth = other_routes(rd)
    a2 = nodes_of(rd, "ast2")
    cells = []
    for p, fn in POLICIES.items():
        hit = fn(rows)
        k = hit["nodes"] if hit else None
        tool = min(x for x in (k, oth) if x is not None) if (k or oth) else None
        cells.append(f"{tool}" + ("*" if hit and (oth is None or k < oth) else ""))
        if a2 and tool:
            tot[p][0 if tool < a2 else 1 if tool == a2 else 2] += 1
    print(f"{rel[13:]:20} {str(oth):>5} {str(a2):>5} " + " ".join(f"{c:>11}" for c in cells))
print("\nvs ast2 (smaller / same / larger):")
for p, (s, e, l) in tot.items():
    print(f"  {p:12} {s:2} / {e:2} / {l:2}")
print("* = kink merge wins the tool")
