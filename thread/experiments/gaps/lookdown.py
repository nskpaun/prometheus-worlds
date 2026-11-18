"""Would choosing the look-down painted views by coverage leave less surface unpainted? Geometry only, nothing is painted.
  experiments/gaps/lookdown.py [WORLD...] [--plan]     (run from thread/)
Today the 4 or 6 views painted after the ring stand on an ellipse and all look across the middle of the room. Here the
same number is chosen one by one from standing spots on a 1 m grid at the same height (8 headings, 35 and 55 degrees
down): each time the one that shows the most of what a walker comes close to (owner.cover's walker) and no chosen view
yet sees. Printed: the share of the close and floor eval cameras' pixels without paint, and of the walker's points,
for the ring alone, today's views, the chosen ones ("chosen": each pick checked in full, slow) and the ones the
pipeline's own quick planner picks ("planned": owner.lookdown, with the seconds it took; --plan prints these). The shadow of moving things is taken as today's paint has it.
--plan also prints the chosen cameras as JSON."""
import sys, os, json, numpy as np
sys.path[:0] = ["evals", "."]
import cams as K, density as D, owner as O

def walker(C, lifted):
    B = C.B; keep = [j for j, oi in enumerate(C.PRIM_OBJ) if oi not in lifted]; full, was = B.PRIMS, (B.W, B.H)
    clear = lambda e: 0.5 < e[0] < C.RW - 0.5 and 0.5 < e[2] < C.RD - 0.5 and not any(B.inside(e, p, 0.3) for p in C.KEEP_OUT)
    spots = [np.array([x, C.EYE, z]) for x in np.arange(0.75, C.RW - 0.5, 1.0) for z in np.arange(0.75, C.RD - 0.5, 1.0) if clear([x, C.EYE, z])]
    B.PRIMS = [full[j] for j in keep]; B.W, B.H = 64, 48; out = []
    try:
        for e in spots:
            for i in range(8):
                for down in (0.0, 35.0):
                    a, p = np.pi * i / 4, np.radians(down)
                    t, P, pid, nrm = B.trace((tuple(e), tuple(e + [np.cos(a) * np.cos(p), -np.sin(p), np.sin(a) * np.cos(p)])), keep, True)
                    sid = pid.copy()
                    for j in keep: sid[pid == 6 + j] = 6 + C.PRIM_OBJ[j]
                    m = (t < O.COVER_NEAR) & (sid != 4)
                    if C.OUT: m &= ~np.isin(sid, C.BACKDROP)
                    out.append(np.concatenate([P[m], sid[m][:, None], nrm[m]], 1))
    finally: B.PRIMS = full; B.W, B.H = was
    pts = np.concatenate(out); _, first = np.unique(np.concatenate([np.floor(pts[:, :3] / 0.2), pts[:, 3:4]], 1).astype(np.int64), axis=0, return_index=True)
    pts = pts[np.sort(first)]
    return spots, pts[:, :3], pts[:, 3].astype(int), pts[:, 4:]

def views(C, cams, lifted, shadow):
    B = C.B; NS = 6 + len(C.objs); shaped = [6 + oi for oi in {*C.SHAPES, *(C.SLIM if C.NEW else ())}]; src = D.Views(); src.shadow = shadow
    for cam in cams:
        t, P, sid = B.trace(cam, C.PRIM_OBJ); on = np.isin(sid, shaped); src.append(O.view(cam, t, sid, O.big(on, C.HALO) & ~on, lifted))
    return src, O.rank(np.stack([np.bincount(s[2].ravel(), minlength=NS) for s in src], 1).astype(float))

def score(C, world, src, rank, lifted, Q, qs, qn):
    tot = un = 0
    for name, kind, eye, look in K.layout_cams(world):
        if kind not in ("close", "floor"): continue
        P, sid, n = D.seen(C, (tuple(eye), tuple(look)), lifted); own = D.owner(C, src, rank, lifted, P, sid, n)
        sky = (sid == 4) if C.OUT else np.zeros(sid.shape, bool); tot += (~sky).sum(); un += ((own < 0) & ~sky).sum()
    return round(float(un / tot), 4), round(float((D.owner(C, src, rank, lifted, Q, qs, qn) < 0).mean()), 4)

def main(w, plan):
    C, world = K.layout(w); B = C.B; now, rank0, lifted = D.sources(C, world); ring = [B.CAMS[str(i)] for i in range(C.RING)]; n_extra = C.N - C.RING
    spots, Q, qs, qn = walker(C, lifted); h = min(C.RH - 0.3, C.EYE + 0.6)
    out = {"world": os.path.basename(world), "walker_points": len(Q), "today": score(C, world, now, rank0, lifted, Q, qs, qn)}
    src, rank = views(C, ring, lifted, now.shadow); out["ring_alone"] = score(C, world, src, rank, lifted, Q, qs, qn)
    cand = [((e[0], h, e[2]), (e[0] + np.cos(a) * np.cos(p), h - np.sin(p), e[2] + np.sin(a) * np.cos(p))) for e in spots if not C.crowded(e)
            for a in np.pi * np.arange(8) / 4 for p in np.radians([35.0, 55.0])]
    chosen = []
    for _ in range(n_extra):
        un = D.owner(C, src, rank, lifted, Q, qs, qn) < 0
        if not un.any(): break
        Qu = Q[un]; gain = np.array([O.sees(B, cam, Qu, 1e9).sum() for cam in cand])       # a quick count picks the 6 to try in full
        best = None
        for k in np.argsort(-gain)[:6]:
            s2, r2 = views(C, ring + chosen + [cand[k]], lifted, now.shadow); left = (D.owner(C, s2, r2, lifted, Q, qs, qn) < 0).sum()
            if best is None or left < best[0]: best = (left, k, s2, r2)
        chosen.append(cand[best[1]]); src, rank = best[2], best[3]
    out["chosen"] = score(C, world, src, rank, lifted, Q, qs, qn)
    import time; t0 = time.time(); s0, r0 = views(C, ring, lifted, now.shadow); quick = O.lookdown(C, s0, r0, n_extra, lifted); out["plan_s"] = round(time.time() - t0, 1)
    quick += [B.CAMS[str(C.RING + i)] for i in range(len(quick), n_extra)]      # slots the plan leaves open keep today's view
    s2, r2 = views(C, ring + quick, lifted, now.shadow); out["planned"] = score(C, world, s2, r2, lifted, Q, qs, qn); chosen = quick
    if plan: out["cams"] = [[list(map(float, e)), list(map(float, l))] for e, l in chosen]
    print(json.dumps(out), flush=True)

if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--plan"]
    for w in a or ["ex-tower", "garage-shapes", "ex-blossom"]: main(w, "--plan" in sys.argv)
