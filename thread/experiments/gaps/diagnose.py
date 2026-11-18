"""Why is a near pixel unpainted? experiments/gaps/diagnose.py WORLD   (run from thread/)
For every close and floor eval camera pixel that density.owner leaves without a painted view, the first reason that applies:
  shade     floor in the shadow of a thing that moves away (owner.shade; WORLD_GAPS=0: the whole band round it)
  frame     no painted view has the point in its picture
  hidden    in some picture, but something stands in front of it in every one
  lift      seen, but only inside the band round a moving thing that is never taken (16 px; WORLD_GAPS=0: 40 px)
  behind    seen only from behind a shape's surface
  small     seen only by views that show under 400 px of that surface
  outline   seen, but the picture shows another surface there and at all 8 neighbours
Printed per surface as share of all near pixels."""
import sys, os, json, numpy as np
sys.path.insert(0, "evals")
import cams as K, density as D
import owner as O
def main(world):
    C, world = K.layout(world); B = C.B
    src, rank, lifted = D.sources(C, world); N = len(src); H, W = src[0][2].shape
    names = ["x0", "floor", "z0", "x1", "ceil", "z1"] + [o["name"] for o in C.objs]
    tot = 0; tally = {}; percam = {}
    for name, kind, eye, look in K.layout_cams(world):
        if kind not in ("close", "floor"): continue
        cam = (tuple(eye), tuple(look)); P, sid, n = D.seen(C, cam, lifted); own = D.owner(C, src, rank, lifted, P, sid, n)
        sky = (sid == 4) if C.OUT else np.zeros(sid.shape, bool)
        tot += (~sky).sum(); un = (own < 0) & ~sky; percam[name] = round(float(un.sum() / (~sky).sum()), 3)
        if not un.any(): continue
        Pu, su, nu = P[un], sid[un], n[un]; shaped = np.isin(su, [6 + oi for oi in C.SHAPES])
        frame = np.zeros(len(Pu), bool); vis = frame.copy(); nolift = frame.copy(); front = frame.copy(); big = frame.copy()
        for v in range(N):
            cm, st, ssid, sedge, shalo, sfar, slift = src[v]
            if not O.GAPS: slift = sfar
            with np.errstate(all="ignore"):
                px, py, dist, z = B.project(cm, Pu); xi, yi = np.round(px).astype(int), np.round(py).astype(int)
            ins = (z > 0.05) & (xi >= 1) & (xi < W - 1) & (yi >= 1) & (yi < H - 1); xi, yi = np.clip(xi, 0, W - 1), np.clip(yi, 0, H - 1)
            see = ins & B.visible(cm[0], Pu)
            lf = slift[yi, xi] if lifted else np.zeros(len(Pu), bool)
            to = np.array(cm[0]) - Pu; cos = (to * nu).sum(1) / np.maximum(np.linalg.norm(to, axis=1), 1e-9)
            fr = ~shaped | (cos > 0)
            frame |= ins; vis |= see; nolift |= see & ~lf; front |= see & ~lf & fr; big |= see & ~lf & fr & (rank[su, v] < 1e6)
        if getattr(src, "shadow", None): shade = (su == 1) & src.shadow(Pu)
        else:
            shade = np.zeros(len(Pu), bool)
            for lo, hi, bnd in O.shade_bands(C, lifted): shade |= (su == 1) & (O.near(Pu, lo, hi) < bnd)
        why = np.where(shade & big, "shade", np.where(~frame, "frame", np.where(~vis, "hidden", np.where(~nolift, "lift", np.where(~front, "behind", np.where(~big, "small", "outline"))))))
        for s, w in zip(su, why): tally[(names[s], w)] = tally.get((names[s], w), 0) + 1
    print(os.path.basename(world), "lifted", lifted, "unpainted total %.4f" % (sum(tally.values()) / tot))
    for (s, w), c in sorted(tally.items(), key=lambda kv: -kv[1]): print("  %-18s %-8s %.4f" % (s, w, c / tot))
    by = {}
    for (s, w), c in tally.items(): by[w] = by.get(w, 0) + c
    print("  by cause:", {w: round(c / tot, 4) for w, c in sorted(by.items(), key=lambda kv: -kv[1])})
    print("  per camera:", {k: v for k, v in percam.items() if v > 0.01})
if __name__ == "__main__":
    for w in sys.argv[1:] or ["ex-tower", "garage-shapes", "ex-blossom"]: main(w)
