"""The numbers behind NOTES.md: who owns what today, what a cheaper or dearer ring alone does to the score, and how
best to split 17 minutes of painting between the ring and detail views. No GPU. Takes a few minutes.
  study.py [WORLD ...]            (default ex-tower garage-shapes ex-blossom) -> work/study.json, and a summary to stdout
  study.py --ring8 WORLD          one world with the 8-view ring the Quick level uses (run by the above in a process of
                                  its own: chisel_ring holds one ring at a time)
"""
import sys, os, json, subprocess, importlib, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import score as Z, plan as PL
K = Z.K; BUDGET = 1020.0
RATE = Z.BASE[0] * Z.BASE[1] / Z.SECS      # pixels painted per second at today's size

def ceiling(C):
    """r from 0.8 m if every painted pixel of 17 minutes fell square-on, once, on the walls' lower 2.6 m and the floor:
    what no placing of views can beat at today's painting speed."""
    area = C.RW * C.RD + (0 if C.OUT else 2 * (C.RW + C.RD) * min(C.RH, PL.BAND))
    return {"area_m2": round(area, 1), "pixels": int(BUDGET * RATE), "px_per_m": round(float(np.sqrt(BUDGET * RATE / area)), 1),
            "r_at_0.8m": round(float(np.sqrt(BUDGET * RATE / area) / (Z.FE / 0.8)), 3)}

def owners(C, world, src, rank, lifted):
    """Share of what is seen that each painted view owns: by the walker, and over the training cameras consistent.py wrote."""
    S, R, dead, _ = PL.demand(C, src, rank, lifted, []); _, own = PL.ring_r(S, R, dead, range(C.N), Z.BASE)
    walk = np.bincount(own[own >= 0], minlength=C.N) / max(1, (own >= 0).sum()); train = np.zeros(C.N)
    try: tc = json.load(open(f"{world}/gen_c/cams.json"))[::3]
    except OSError: tc = []
    for e, l in tc:
        P, sid, n = Z.seen(C, (tuple(e), tuple(l)), lifted, 8); S1 = Z.scores(C, src, rank, lifted, P, sid, n)
        o = Z.ring_owner(C, S1, np.arange(C.N), Z.shade(C, lifted, P, sid) | ((sid == 4) if C.OUT else False)); train += np.bincount(o[o >= 0], minlength=C.N)
    train /= max(1, train.sum())
    return {"walker_share": [round(float(v), 3) for v in walk], "training_share": [round(float(v), 3) for v in train],
            "views_owning_under_2pct_of_training": int((train < 0.02).sum()), "views": C.N, "ring": C.RING,
            "order_least_first": [int(v) for v in np.argsort(walk, kind="stable")]}

def ring_only(C, world, src, rank, lifted, order, cams=None):
    """The score with no detail views: other sizes, and the least-owning views left out."""
    rows = [("as painted 768x576", list(range(C.N)), (768, 576))] + [(f"all views at {w}x{h}", list(range(C.N)), (w, h)) for w, h in ((1024, 768), (704, 528), (640, 480), (512, 384))]
    drop = [v for v in order if v != 0]
    rows += [(f"768x576, {k} least-owning views left out", sorted(set(range(C.N)) - set(drop[:k])), (768, 576)) for k in (2, 4, 6, 8)]
    rows.append(("768x576, the downward views left out", list(range(C.RING)), (768, 576)))
    res = Z.evaluate(C, world, src, rank, lifted, [(k, s, []) for _, k, s in rows], cams)
    return [{"what": w, "views": len(k), "seconds": round(len(k) * Z.seconds(s), 0), **PL.brief(r)} for (w, k, s), r in zip(rows, res)]

def splits(world, ring=None, budget=BUDGET, ns=(0, 2, 4, 6, 8, 10, 12)):
    out = []
    for n in ns:
        for tag, kw in (("drop and shrink", {}), ("shrink only", {"keep_ring": True}), ("big floor allowed", {"big_floor": True})):
            if n == 0 and tag == "big floor allowed": continue
            p = PL.plan(world, n, budget, ring=ring, **kw)
            out.append({"n": n, "pay": tag, "views": [(v["name"], v["size"], v["seconds"]) for v in p["views"]], "ring": p["pay"]["ring"], "total_seconds": p["pay"]["total_seconds"], "after": p["score"]["after"], "before": p["score"]["before"], "seconds": p["seconds"]})
    return out

def ring8(world_dir):
    """The 8-view ring: its cameras were never prepared for this world, so its views are traced here. Quick paints it in
    6 steps, about 49/57 of a view's time (docs/how-it-works.md), so 12 views cost 12 x 56 s = 672 s. plan.py counts 65 s
    a view, so it is given 12 x 9 s more to spend: the same 1020 s."""
    C, world = K.layout(world_dir); cams = K.layout_cams(world)          # the eval cameras do not depend on the ring
    os.environ.update(WORLD_DIR=world, WORLD_VIEWS="8"); C = importlib.reload(C)
    src, rank, lifted = Z.sources(C, world, fresh=True); ring = (C, world, (src, rank, lifted), cams)
    base = Z.evaluate(C, world, src, rank, lifted, [(list(range(C.N)), (768, 576), [])], cams)[0]
    extra = C.N * (Z.SECS - Z.SECS * 49 / 57)
    return {"views": C.N, "ring_seconds": round(C.N * Z.SECS * 49 / 57, 0), "ring_only": PL.brief(base),
            "splits": [{k: v for k, v in s.items() if k != "before"} for s in splits(world, ring, BUDGET + extra, (2, 4, 5, 6)) if s["pay"] == "shrink only"]}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--ring8"]: print(json.dumps(ring8(sys.argv[2]))); sys.exit()
    out = {}
    for w in sys.argv[1:] or ["ex-tower", "garage-shapes", "ex-blossom"]:
        C, world = K.layout(w); src, rank, lifted = Z.sources(C, world)
        o = owners(C, world, src, rank, lifted)
        r8 = subprocess.run([sys.executable, os.path.abspath(__file__), "--ring8", w], capture_output=True, text=True)
        out[os.path.basename(world)] = {"room": [C.RW, C.RH, C.RD], "outdoor": bool(C.OUT), "ceiling": ceiling(C), "owners": o,
                                        "ring_only": ring_only(C, world, src, rank, lifted, o["order_least_first"]), "splits": splits(w),
                                        "ring8": json.loads(r8.stdout.strip().splitlines()[-1]) if r8.returncode == 0 and r8.stdout.strip() else {"error": r8.stderr[-800:]}}
        print(w, "done", file=sys.stderr, flush=True)
    os.makedirs(f"{HERE}/work", exist_ok=True); json.dump(out, open(f"{HERE}/work/study.json", "w"), indent=1)
    for w, d in out.items():
        print(f"\n== {w}  room {d['room']}  ceiling {d['ceiling']}")
        print("owners:", d["owners"])
        for r in d["ring_only"]: print("  ring only |", r["what"], "|", r["seconds"], "s | near", r["near_median_r"], "soft", r["soft_share_r_lt_1"], "<0.5", r["r_lt_0.5"], "unpainted", r["unpainted"], "worst", r["worst_close_cam"])
        for s in d["splits"]:
            a = s["after"]; print(f"  N={s['n']:2d} {s['pay']:17s} | ring {len(s['ring']['views_kept'])} at {s['ring']['size']} | total {s['total_seconds']} s | near {a['near_median_r']} soft {a['soft_share_r_lt_1']} <0.5 {a['r_lt_0.5']} unpainted {a['unpainted']} close {a['close_median_r']} floor {a['floor_median_r']} centre {a['centre_median_r']} worst {a['worst_close_cam']} | {[v[0] for v in s['views']]}")
        print("  ring8:", json.dumps(d["ring8"])[:3000])
