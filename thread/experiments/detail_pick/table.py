"""The judge's numbers as tables.
  table.py [--md] work/judge_WORLD.json [...]      (what judge.py printed; work/plans/ holds the planners' own output)
One row per plan: what it costs, the eval cameras (near = close + floor), the walk set, seams and flagged views.
"claimed" is the near median r the planner itself printed (no paint counted as 0); "r0" is the judge's by that count.
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {"A": "a_paid", "Afree": "a_free", "B": "b", "Bfree": "b_free", "C": "c", "M": "m", "H": "h"}
def claimed(world, tag):
    """The planner's own predicted near median r for this plan."""
    try: d = json.load(open(f"{HERE}/work/plans/{FILES[tag.rstrip('0123456789') + ('free' if tag.endswith('free') else '')]}_{world}_{''.join(c for c in tag if c.isdigit())}.json"))
    except (OSError, KeyError): return None
    return d["score"]["after"].get("near_r", d["score"]["after"].get("near_median_r")) if "score" in d else d["after"]["near_r_median"]
def cells(world, tag, r, base):
    e, w, wn = r["eval"]["near"], r["walk"]["all"], r["walk"]["near"]; n = len(r.get("views", [])); c = claimed(world, tag) if n else None
    per = "" if not n else "%+.4f" % ((w["sharp"] - base["walk"]["all"]["sharp"]) / n)
    return [tag, n, f"{r['ring']['views']} at {r['ring']['size'][0]}x{r['ring']['size'][1]}", f"{r['paint_seconds']:.0f}", "" if c is None else f"{c:.3f}", f"{e['median_r0']:.3f}", f"{e['median_r']:.3f}", f"{e['r_lt_1']:.3f}", f"{e['r_lt_0.5']:.3f}",
            f"{e['unpainted']:.3f}", f"{r['eval']['worst_close'][1]:.3f}", f"{e['sharp']:.3f}", f"{w['sharp']:.3f}", per, f"{wn['sharp']:.3f}", f"{wn['median_r']:.3f}", f"{w['unpainted']:.3f}", f"{r['steps']['walk']:.4f}",
            f"{sum(any(not f.startswith('claims') for f in v['flags']) for v in r.get('views', []))}/{n}" if n else ""]
HEAD = ["plan", "N", "ring kept", "paint s", "claimed", "r0", "near med r", "r<1", "r<0.5", "unpainted", "worst close", "sharp", "walk sharp", "per picture", "walk <2.5 m sharp", "walk <2.5 m med r", "walk unpainted", "steps", "flagged"]
md = "--md" in sys.argv
for f in [a for a in sys.argv[1:] if a != "--md"]:
    d = json.load(open(f)); rows = [cells(d["world"], "baseline", d["baseline"], d["baseline"])] + [cells(d["world"], t, r, d["baseline"]) for t, r in d.get("plans", {}).items()]
    print(f"\n{'### ' if md else ''}{d['world']}: {d['room'][0]:g} x {d['room'][2]:g} m, {'outdoor' if d['outdoor'] else 'indoor'}, {d['ring_views']} painted views, {d['walk_cameras']} walk cameras\n")
    if md: print("| " + " | ".join(HEAD) + " |\n|" + "---|" * len(HEAD)); [print("| " + " | ".join(map(str, r)) + " |") for r in rows]
    else:
        wd = [max(len(str(x)) for x in col) for col in zip(HEAD, *rows)]
        for r in [HEAD] + rows: print("  ".join(str(x).ljust(w) for x, w in zip(r, wd)))
