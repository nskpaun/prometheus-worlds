"""The predicted score for budgets 2, 4, 6, 8 on the three cached worlds, each way of paying: one line per plan.
  table.py [WORLD ...] [--pay down,shrink,none]      (cwd thread/; about 6 s per plan; writes work/table.json)
Columns: near_r median, soft share (r < 1), share with r < 0.5, worst close camera's median r, unpainted share (all
over the close and floor eval cameras), the centre cameras' median r, the walkers' mean softness in octaves, the share of
near eval pixels a detail view owns, and the share of painted pixels in use (ring / mean detail view).
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import plan as PL
a = sys.argv[1:]; modes = a[a.index("--pay") + 1].split(",") if "--pay" in a else ["down", "shrink", "none"]
worlds = [w for i, w in enumerate(a) if not w.startswith("--") and (i == 0 or a[i - 1] != "--pay")] or ["ex-tower", "garage-shapes", "ex-blossom"]
row = lambda s: f"near_r {s['near_r_median']:.3f}  soft {s['soft_share']:.3f}  r<.5 {s['near']['r_lt_0.5']:.3f}  worst {s['worst_close'][1]:.3f} ({s['worst_close'][0]})  unpainted {s['near']['unpainted']:.3f}  centre_r {s['centre']['median_r']:.3f}"
out = []
for w in worlds:
    for mode in modes:
        for n in (2, 4, 6, 8):
            d = PL.plan(w, n, mode, step=4); out.append(d)
            if n == 2 and mode == modes[0]: print(f"{w:14s} before         {row(d['before'])}  walk {d['walk']['before']['mean_octaves_soft'] if mode == 'none' else '-'}", flush=True)
            p = d["pay"]; ring = "x".join(map(str, d["ring"])) + (f" -{len(d['drop'])}" if d["drop"] else "")
            print(f"{w:14s} N={len(d['views'])} {mode:6s} ring {ring:10s} paint {p['paint_seconds_before']}->{p['paint_seconds_after']} s  {row(d['after'])}  walk {d['walk']['before']['mean_octaves_soft']}->{d['walk']['after']['mean_octaves_soft']}  owned {d['after']['near']['detail_owned']:.2f}  used {d['walk']['ring_pixels_used']:.2f}/{sum(v['pixels_used'] for v in d['views']) / max(len(d['views']), 1):.2f}  {d['seconds']} s", flush=True)
os.makedirs(f"{HERE}/work", exist_ok=True); json.dump(out, open(f"{HERE}/work/table.json", "w"))
