"""The merged detail-pass planner: detail_b's views, paid for the way detail_c found cheapest. No pictures, no GPU.
  pick.py WORLD --budget N [--high 0]     (run from thread/ with .venv/bin/python; prints one JSON object, detail_b's format)
One out, one in: each detail picture takes the place of the painted view a walker would miss least, so painting takes
exactly as long as today and no picture is painted at a size that was never timed.
- Which views go (order()): the one whose loss costs a walker the least of min(r, 1), taken one at a time and counted
  again after each, since two neighbours can stand in for each other but not both go. View 0 stays: it is painted first
  and sets the look. (detail_c left out the views that own least, all at once; detail_b only the look-down ones.)
- Which views come: detail_b's chooser (ordinary photos from standing spots, a wall with its floor or the ground ahead,
  taken greedily for the softness they remove given the ring that is left), with two changes. They are chosen under
  the judge's ownership rule (within 60 degrees of head-on, 5% rim), not detail_b's looser one. And each spot is also
  offered from HIGH metres up looking 60 degrees down: the ground seen nearly square-on, as detail_a's and detail_c's
  floor tiles see it, but still a picture with a near and a far edge, like the look-down views the ring already has.
  --high 0 leaves those out.
Nothing in detail_b is edited: its functions are called from here and two of its constants are set in memory.
"""
import sys, os, json, time, multiprocessing as mp, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import judge as J
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "detail_b"))
import plan as Bp                              # brings detail_b/score.py in as Bp.S
S = Bp.S; S.COS, S.EDGE = J.HEAD, J.EDGE
HIGH, STEEP = 3.0, -60                         # metres up and degrees down of the raised candidates (indoors: no nearer the ceiling than 0.4 m)

def order(C, src, rank, lifted, n):
    """The n painted views to leave out, least missed first, and what a walker loses with each (mean of min(r, 1))."""
    B = C.B; N = len(src); fo = J.focal(J.BASE, B.FOV); Ss, Rs = [], []
    for si, e in enumerate(Bp.spots(C, 30)):   # a coarse walker: 30 spots, two level looks and two down
        for i in range(4):
            cam = (tuple(map(float, e)), Bp.aim(e, 90 * i + 45 * (si % 2), -30 * (i % 2)))
            P, sid, nr = J.seen(C, cam, lifted, 64, 36, J.SFOV); P, sid, nr = P.reshape(-1, 3), sid.ravel(), nr.reshape(-1, 3)
            live = ~(J.shade(C, lifted, P, sid) | ((sid == 4) if C.OUT else False))
            P, sid, nr = P[live], sid[live], nr[live]
            Sc = J.scores(C, src, rank, lifted, P, sid, nr); T = J.tangents(nr); Je = J.spread(B, cam, J.FE, P, T); R = np.zeros(Sc.shape, np.float32)
            for v in range(N):
                m = Sc[v] < 1e5
                if m.any(): R[v, m] = np.minimum(J.mix(J.spread(B, src[v][0], fo, P[m], T[m]), Je[m])[0], 1)
            Ss.append(Sc); Rs.append(R)
    Sc, R = np.concatenate(Ss, 1), np.concatenate(Rs, 1); ar = np.arange(Sc.shape[1])
    def sharp():
        j = Sc.argmin(0); return float(np.where(Sc[j, ar] < 1e5, R[j, ar], 0).mean())
    out, now = [], sharp()
    for _ in range(min(n, N - 1)):
        best = None
        for v in range(1, N):
            if v in [o[0] for o in out]: continue
            keep = Sc[v].copy(); Sc[v] = 1e6; s = sharp(); Sc[v] = keep
            if best is None or s > best[1]: best = (v, s)
        Sc[best[0]] = 1e6; out.append((best[0], round(now - best[1], 4))); now = best[1]
    return out

def _cands(job):
    """detail_b's candidates from one spot; a raised spot looks only steeply down."""
    Bp.PITCH = (STEEP,) if job[0] >= Bp.G["level"] else (-10, -40)
    return Bp._cands(job)

def plan(world_dir, budget, high=HIGH):
    t0 = time.time(); C, world = J.K.layout(world_dir); B = C.B; src, count, lifted = S.sources(C, world)
    gone = order(C, src, S.ranks(count), lifted, budget); drop = sorted(v for v, _ in gone); was = C.N * J.SECS
    bill = {"mode": "one out, one in", "ring_views": C.N - len(drop), "ring_paint": list(J.BASE), "dropped_views": drop, "walker_loss_each": dict(gone),
            "detail_views": budget, "paint_seconds_before": round(was), "paint_seconds_after": round(was), "extra_seconds": 0}
    stand = Bp.spots(C, 60); up = high if C.OUT else min(high, C.RH - 0.4)
    raised = [e + [0, up - C.EYE, 0] for e in stand if up >= C.EYE + 0.5 and not any(B.inside(e + [0, up - C.EYE, 0], p, 0.3) for p in C.KEEP_OUT)]
    walk = [(tuple(map(float, e)), Bp.aim(e, 90 * i + 45 * (si % 2), 0)) for si, e in enumerate(stand) for i in range(4)]
    walk += [(tuple(map(float, e)), Bp.aim(e, 180 * i + 45 * (si % 4), -30)) for si, e in enumerate(stand) for i in range(2)]
    Bp.G.update(C=C, src=src, rank=S.ranks(count, drop), lifted=lifted, k=1.0, walk=walk, spots=stand + raised, level=len(stand), detail=S.BASE)
    pool = mp.get_context("fork").Pool(max(1, min(8, (os.cpu_count() or 4) - 2)))      # as detail_b: one set of workers, made after G is filled
    P, n, sid, q, Wt, _ = Bp.patches(pool)
    cands = [c for cs in pool.map(_cands, [(si, P, n) for si in range(len(stand) + len(raised))], chunksize=1) for c in cs]
    views = []
    for c, g, j, qj in Bp.choose(cands, q, Wt, budget):
        cam = cands[c][0]; d = np.array(cam[1]) - cam[0]
        views.append({"cam": [list(cam[0]), list(cam[1])], "eye": list(cam[0]), "look": list(cam[1]), "heading_deg": round(float(np.degrees(np.arctan2(d[2], d[0]))) % 360, 1),
                      "pitch_deg": round(float(np.degrees(np.arcsin(d[1]))), 1), "fov": B.FOV, "paint": list(S.BASE), "seconds": J.SECS, "owns": Bp.owns(C, sid, P, j, qj)})
    out = {"world": os.path.basename(world), "budget": budget, "views": views, "ring": list(J.BASE), "detail": list(S.BASE), "drop": drop, "pay": bill, "candidates": len(cands)}
    dsrc = pool.map(Bp._render, [(tuple(v["eye"]), tuple(v["look"])) for v in views], chunksize=1); pool.close()
    sc = S.score(world, out, 8, (C, world, src, count, lifted, dsrc))                   # detail_b's own scorer; judge.py is the one to trust
    out["before"], out["after"] = ({k: v for k, v in sc[s].items() if k != "cams"} for s in ("before", "after"))
    out["seconds"] = round(time.time() - t0, 2)
    return out

if __name__ == "__main__":
    a = sys.argv[1:]; val = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d
    print(json.dumps(plan(a[0], val("--budget", 4), val("--high", HIGH))))
