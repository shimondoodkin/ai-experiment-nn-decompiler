import json
from pathlib import Path

OUT = Path(__file__).parent / "out"
for f in sorted(OUT.glob("*.json")):
    d = json.loads(f.read_text())
    a = d.get("ast4", {})
    head = f"{d['task']:8s} {d['act']:9s} learned={d['learned']!s:5s} ast4={a.get('nodes')}/{a.get('dump_nodes')} {a.get('route', a.get('error', ''))[:30]}"
    if "families" in d:
        fams = d["families"]
        best = min(fams.items(), key=lambda kv: kv[1]["fit_err_frac"])
        line = (f" | true-vs-net {d['intended_err_frac']:.3f} | best fam {best[0]} {best[1]['fit_err_frac']:.4f}"
                f" snapped={best[1].get('snapped')} pass={best[1].get('snapped_pass')}"
                f" hinges+fam={best[1].get('hinges_to_pass')} hinges-only={d.get('linear_only_hinges_to_pass')}")
        if "n_kinks_in_box" in d:
            ks = [u["centre"] for u in d["units"] if u["in_box"]]
            line += f"\n{'':20s} kinks/centres in box ({d['n_kinks_in_box']}): {ks}  optimal: {d['optimal_knots_same_count']}"
    else:
        line = (f" | intended {d['intended_check']} | snapped {d['snapped_circle']} | fitted-circle {d['fitted_circle']}"
                f"\n{'':20s} circle_fit {d['circle_fit']} radius {d['radius_min_max_mean']} pieces {d.get('boundary_pieces_relu')}")
    print(head + line)
