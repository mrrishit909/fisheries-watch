"""Detectors. Everything here sees only what a monitoring centre would: received AIS, the registry, SAR detections. The
generator's truth is used to train on past cases (training seeds stand in for past investigations) and to score.

- coverage: how often AIS from each 20 km cell reaches the ground, estimated from the history itself
- gaps: every silence over two hours, with the chance of that silence under the cell's coverage, its distance from the
  reserve, whether it crossed it, how far the boat got, and whether a carrier was loitering within reach; a logistic
  model on those, against the rule "any gap over six hours"
- fusion: SAR detections matched to where AIS says ships were; unmatched ones over 18 m are dark targets; each is
  attributed to the boats that were silent at the time and could have been there
- encounters: a carrier and a fishing boat both slow within a kilometre for two hours; a carrier loitering offshore while
  a silent boat could have reached it; against the rule "any two vessels within a kilometre for an hour"
- risk: a logistic score per fishing boat over those signals and the registry, with each factor's contribution
"""
import functools
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import world as W

VERSION = "fw-1.0"
CELL = 20.0
GAP_SLOTS = 12                    # a gap is two hours or more without a message
BASELINE_GAP_H = 6.0
TRAIN_SEEDS = (101, 102, 103)
GAP_FEATURES = ["improbability", "log_hours", "near_reserve", "crosses_reserve", "progress_knots", "carrier_in_reach", "at_region_edge", "fishing_vessel"]
RISK_FEATURES = ["worst_gap", "dark_sightings_near_reserve", "undeclared_carrier_encounters", "dark_transship_partner", "unlicensed", "open_registry", "prior_violations", "owner_network"]


def by_vessel(ais, n, upto=W.SLOTS):
    """-> list of (slots, xy, speed) per vessel from the received messages before `upto`."""
    a = np.array([m for m in ais if m[1] < upto], dtype=float).reshape(-1, 5)
    out = []
    for i in range(n):
        k = a[:, 0] == i
        out.append((a[k, 1].astype(int), a[k, 2:4], a[k, 4]))
    return out


def coverage(tracks_rx, upto):
    """Received over expected messages per cell, from silences under six hours (longer ones are left out: they may be on purpose)."""
    nx, ny = int(W.W_KM // CELL) + 1, int(W.H_KM // CELL) + 1
    got, missed = np.ones((nx, ny)), np.ones((nx, ny))
    for slots, xy, _ in tracks_rx:
        if len(slots) < 2:
            continue
        cx, cy = (xy[:, 0] // CELL).astype(int).clip(0, nx - 1), (xy[:, 1] // CELL).astype(int).clip(0, ny - 1)
        np.add.at(got, (cx, cy), 1)
        d = np.diff(slots) - 1
        k = (d > 0) & (d < 36)
        np.add.at(missed, (cx[1:][k], cy[1:][k]), d[k])
    return got / (got + missed)


def segment_cells(p, q, n):
    t = (np.arange(n) + 0.5) / max(n, 1)
    pts = p[None, :] + (q - p)[None, :] * t[:, None]
    return pts


def dist_to_mpa(pts):
    return np.hypot(pts[:, 0] - W.MPA["x"], pts[:, 1] - W.MPA["y"]) - W.MPA["r"]


def loiters(tracks_rx, vessels):
    """Carrier loitering spells: under 2 knots, over 30 km from port, at least three hours. -> list of (carrier, s0, s1, x, y)."""
    out = []
    for v in vessels:
        if v["kind"] != "carrier":
            continue
        slots, xy, sp = tracks_rx[v["idx"]]
        slow = (sp < 2.0) & (W.dist_port(xy[:, 0], xy[:, 1]) > 30)
        start = None
        for j in range(len(slots) + 1):
            ok = j < len(slots) and slow[j] and (start is None or slots[j] - slots[j - 1] <= 18)
            if ok and start is None:
                start = j
            elif not ok and start is not None:
                if slots[j - 1] - slots[start] >= 18:
                    out.append((v["idx"], int(slots[start]), int(slots[j - 1]), float(xy[start:j, 0].mean()), float(xy[start:j, 1].mean())))
                start = j if (j < len(slots) and slow[j]) else None
    return out


def reachable(g, lx, ly, l0, l1, kn=15.0):
    """Could the silent boat have been at (lx, ly) at some time during [l0, l1]? From its last fix, and back to its next."""
    v = kn * W.KM_PER_SLOT_PER_KN
    t_early = g["s0"] + math.hypot(lx - g["x0"], ly - g["y0"]) / v
    t_late = g["s1"] if g["open"] else g["s1"] - math.hypot(lx - g["x1"], ly - g["y1"]) / v
    return t_early <= t_late and t_early <= l1 and t_late >= l0


def gaps(tracks_rx, vessels, cov, loit, upto):
    """Every silence of two hours or more (including one still open at `upto`), with its features."""
    rows = []
    nx, ny = cov.shape
    for v in vessels:
        slots, xy, sp = tracks_rx[v["idx"]]
        if not len(slots):
            continue
        ends = list(zip(slots[:-1], slots[1:], range(len(slots) - 1)))
        if upto - slots[-1] > GAP_SLOTS:
            ends.append((slots[-1], upto, len(slots) - 1))
        for s0, s1, j in ends:
            if s1 - s0 <= GAP_SLOTS:
                continue
            p, q = xy[j], (xy[j + 1] if s1 < upto or j + 1 < len(slots) and slots[j + 1] == s1 else xy[j])
            open_gap = bool(s1 == upto and not (j + 1 < len(slots) and slots[j + 1] == s1))
            n = int(s1 - s0 - 1)
            pts = segment_cells(p, q, n)
            r = cov[(pts[:, 0] // CELL).astype(int).clip(0, nx - 1), (pts[:, 1] // CELL).astype(int).clip(0, ny - 1)]
            logp = float(np.log10(np.clip(1 - r, 1e-6, 1)).sum())
            hours = (s1 - s0) / 6
            km = float(np.hypot(*(q - p)))
            dm = dist_to_mpa(np.vstack([p, q, pts]))
            g0 = {"s0": int(s0), "s1": int(s1), "open": open_gap, "x0": float(p[0]), "y0": float(p[1]), "x1": float(q[0]), "y1": float(q[1])}
            reach = int(any(reachable(g0, lx, ly, l0, l1) for c, l0, l1, lx, ly in loit))
            edge = min(p[0], W.W_KM - p[0], p[1], W.H_KM - p[1], q[0], W.W_KM - q[0], q[1], W.H_KM - q[1])
            rows.append({"vessel": v["idx"], "s0": int(s0), "s1": int(s1), "open": open_gap, "x0": float(p[0]), "y0": float(p[1]), "x1": float(q[0]), "y1": float(q[1]),
                         "hours": hours, "logp": logp, "mpa_km": float(dm.min()), "km": km,
                         "f": [min(-logp, 30.0) / 10, math.log(hours), float(dm.min() < 20), float(dm.min() < 0), min(km / max(hours, 1e-9) / 1.852, 15.0) / 10, float(reach), float(edge < 10), float(v["kind"] == "fishing")]})
    return rows


def label_gaps(rows, truth):
    dark = {}
    for i, a, b, _ in truth["dark"]:
        dark.setdefault(i, []).append((a, b))
    for g in rows:
        miss = g["s1"] - g["s0"] - 1
        cover = sum(max(0, min(g["s1"], b) - max(g["s0"] + 1, a)) for a, b in dark.get(g["vessel"], []))
        g["deliberate"] = cover >= 0.5 * miss
    return rows


def fuse(w, tracks_rx, upto, since=0):
    """SAR detections against AIS. -> list of detections with matched vessel or None."""
    out = []
    by_slot = {}
    for d in w["sar"]:
        if since <= d["slot"] < upto:
            by_slot.setdefault(d["slot"], []).append(d)
    for s, dets in sorted(by_slot.items()):
        pred, ids = [], []
        for v in w["vessels"]:
            slots, xy, _ = tracks_rx[v["idx"]]
            j = np.searchsorted(slots, s)
            p = None
            if j < len(slots) and slots[j] == s:
                p = xy[j]
            elif 0 < j < len(slots) and slots[j] - slots[j - 1] <= GAP_SLOTS:
                f = (s - slots[j - 1]) / (slots[j] - slots[j - 1])
                p = xy[j - 1] + f * (xy[j] - xy[j - 1])
            elif j > 0 and s - slots[j - 1] <= 3:
                p = xy[j - 1]
            elif j < len(slots) and slots[j] - s <= 3:
                p = xy[j]
            if p is not None:
                pred.append(p)
                ids.append(v["idx"])
        L = [w["vessels"][i]["length"] for i in ids]
        # gate on distance and on length: a 20 m return is not a 130 m carrier, however close
        D = np.array([[math.hypot(d["x"] - p[0], d["y"] - p[1]) if 0.4 <= d["length"] / ln <= 2.5 else 1e6 for p, ln in zip(pred, L)] for d in dets]) if pred else np.zeros((len(dets), 0))
        match = {}
        if D.size:
            ri, ci = linear_sum_assignment(np.where(D < 3.0, D, 1e6))
            match = {a: ids[b] for a, b in zip(ri, ci) if D[a, b] < 3.0}
        for k, d in enumerate(dets):
            out.append({**d, "matched": match.get(k), "dark": k not in match and d["length"] >= 18})
    return out


def attribute(det, tracks_rx, vessels, gap_p):
    """Boats silent at the detection's time that could have been there (15 knots from the last and to the next message) and
    are of a size the radar's length estimate allows; ranked by how suspicious their silence already looked."""
    s = det["slot"]
    cands = []
    for v in vessels:
        if v["kind"] != "fishing":
            continue
        slots, xy, _ = tracks_rx[v["idx"]]
        j = np.searchsorted(slots, s)
        if j == 0 or (j < len(slots) and slots[j] - slots[j - 1] <= GAP_SLOTS) or (j < len(slots) and slots[j] == s):
            continue
        a = xy[j - 1]
        if math.hypot(det["x"] - a[0], det["y"] - a[1]) > 15 * W.KM_PER_SLOT_PER_KN * (s - slots[j - 1]) + 2:
            continue
        if j < len(slots) and math.hypot(det["x"] - xy[j][0], det["y"] - xy[j][1]) > 15 * W.KM_PER_SLOT_PER_KN * (slots[j] - s) + 2:
            continue
        if not 0.5 <= det["length"] / v["length"] <= 2.0:
            continue
        cands.append((gap_p.get((v["idx"], int(slots[j - 1])), 0.0), v["idx"]))
    cands.sort(reverse=True)
    return [i for _, i in cands]


def positions(tracks_rx, n):
    """Known position per vessel per slot: received messages, with silences up to 30 minutes interpolated. NaN otherwise."""
    P = np.full((n, W.SLOTS, 2), np.nan)
    S = np.full((n, W.SLOTS), np.nan)
    for i, (slots, xy, sp) in enumerate(tracks_rx):
        if not len(slots):
            continue
        P[i, slots] = xy
        S[i, slots] = sp
        for k in np.flatnonzero((np.diff(slots) > 1) & (np.diff(slots) <= 3)):
            a, b = slots[k], slots[k + 1]
            for t in range(a + 1, b):
                f = (t - a) / (b - a)
                P[i, t] = xy[k] + f * (xy[k + 1] - xy[k])
                S[i, t] = sp[k] + f * (sp[k + 1] - sp[k])
    return P, S


def encounters(P, S, vessels, upto, rule="carrier", since=0):
    """rule='carrier': a carrier and a fishing boat, both under 2 knots, within 1 km for two hours or more, over 20 km from
    port. rule='distance': any two vessels within 1 km for an hour or more. -> list of (a, b, s0, s1, x, y)."""
    kinds = np.array([v["kind"] for v in vessels])
    if rule == "carrier":
        A, B = np.flatnonzero(kinds == "carrier"), np.flatnonzero(kinds == "fishing")
        need = 12
    else:
        A = B = np.arange(len(vessels))
        need = 6
    out = []
    away = W.dist_port(np.nan_to_num(P[:, since:upto, 0], nan=-999), np.nan_to_num(P[:, since:upto, 1], nan=-999)) > 5
    for a in A:
        pa = P[a, since:upto]
        d = np.hypot(P[B, since:upto, 0] - pa[None, :, 0], P[B, since:upto, 1] - pa[None, :, 1])
        close = d < 1.0
        close &= away[a][None, :] & away[B]                                   # not in port
        if rule == "carrier":
            close &= (S[a, since:upto][None, :] < 2.0) & (S[B, since:upto] < 2.0) & (W.dist_port(pa[:, 0], pa[:, 1])[None, :] > 20)
        for row, b in enumerate(B):
            if rule != "carrier" and b <= a:
                continue
            idx = np.flatnonzero(close[row])
            if not len(idx):
                continue
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 6) + 1)       # an hour without a sighting may be missing data, not a parting
            for run in runs:
                if run[-1] - run[0] >= need:
                    out.append((int(a), int(b), since + int(run[0]), since + int(run[-1]), float(pa[run, 0].mean()), float(pa[run, 1].mean())))
    return out


def analyse(w, upto, gap_model=None):
    """Everything the centre can compute at `upto`."""
    V = w["vessels"]
    rx = by_vessel(w["ais"], len(V), upto)
    cov = coverage(rx, upto)
    loit = loiters(rx, V)
    G = gaps(rx, V, cov, loit, upto)
    if gap_model is not None and G:
        pr = gap_model.predict_proba(np.array([g["f"] for g in G]))[:, 1]
        for g, p in zip(G, pr):
            g["p"] = float(p)
    gap_p = {(g["vessel"], g["s0"]): g.get("p", 0.0) for g in G}
    F = fuse(w, rx, upto)
    for d in F:
        d["candidates"] = attribute(d, rx, V, gap_p) if d["dark"] else []
    P, S = positions(rx, len(V))
    E = encounters(P, S, V, upto)
    dark_ts = []
    for c, l0, l1, lx, ly in loit:
        if any(e[0] == c and e[2] <= l1 and e[3] >= l0 for e in E):
            continue                                                           # its partner was on AIS: already an encounter
        partners = []
        for g in G:
            if V[g["vessel"]]["kind"] != "fishing" or g["s1"] < l0 or g["s0"] > l1:
                continue
            if reachable(g, lx, ly, max(l0, l1 - 12), l1):        # a carrier leaves when loading is done: the partner is there at the end
                partners.append((g.get("p", 0.0), g["vessel"]))
        seen = [d for d in F if d["dark"] and l0 <= d["slot"] <= l1 and math.hypot(d["x"] - lx, d["y"] - ly) < 2.0]
        dark_ts.append({"carrier": c, "s0": l0, "s1": l1, "x": lx, "y": ly, "partners": [i for _, i in sorted(partners, reverse=True)], "sar_corroborated": bool(seen)})
    return {"rx": rx, "coverage": cov, "loiters": loit, "gaps": G, "sar": F, "encounters": E, "dark_transship": dark_ts, "P": P, "S": S}


def declared(w, carrier, vessel, slot):
    return any(a["carrier"] == carrier and a["vessel"] == vessel and abs(a["day"] - slot // 144) <= 1 for a in w["authorisations"])


def risk_features(w, A):
    V = w["vessels"]
    prior_owners = {v["owner"] for v in V if v["prior_violations"]}
    X = {}
    for v in V:
        if v["kind"] != "fishing":
            continue
        i = v["idx"]
        mine = [g for g in A["gaps"] if g["vessel"] == i]
        sightings = [d for d in A["sar"] if d["dark"] and d["candidates"][:1] == [i] and dist_to_mpa(np.array([[d["x"], d["y"]]]))[0] < 10]
        X[i] = [max([g.get("p", 0) for g in mine], default=0.0), len(sightings),
                sum(1 for e in A["encounters"] if e[1] == i and not declared(w, e[0], i, e[2])), sum(1 for t in A["dark_transship"] if t["partners"][:1] == [i]),
                float(not v["licensed"]), float(v["flag"] in W.OPEN_REGISTRIES), float(v["prior_violations"]),
                float(any(o["owner"] == v["owner"] and o["prior_violations"] for o in V if o["idx"] != i))]
    return X


@functools.lru_cache(maxsize=2)
def models():
    """Gap classifier and risk model, trained on the training seeds' past cases."""
    Xg, yg, Xr, yr = [], [], [], []
    for seed in TRAIN_SEEDS:
        w = W.build(seed, demo=False)
        A = analyse(w, W.SLOTS)
        G = label_gaps(A["gaps"], w["truth"])
        Xg += [g["f"] for g in G]
        yg += [g["deliberate"] for g in G]
    gm = LogisticRegression(C=1.0, max_iter=2000).fit(np.array(Xg), np.array(yg))
    for seed in TRAIN_SEEDS:
        w = W.build(seed, demo=False)
        A = analyse(w, W.SLOTS, gm)
        X = risk_features(w, A)
        ill = set(w["truth"]["illicit"])
        Xr += list(X.values())
        yr += [i in ill for i in X]
    Xr = np.array(Xr, float)
    mu, sd = Xr.mean(0), Xr.std(0) + 1e-9
    rm = LogisticRegression(C=0.5, max_iter=2000).fit((Xr - mu) / sd, np.array(yr))
    return {"gap": gm, "risk": rm, "mu": mu, "sd": sd, "gap_examples": len(yg), "gap_positive": int(sum(yg)), "risk_examples": len(yr), "risk_positive": int(sum(yr))}


def score_risk(w, A, m):
    X = risk_features(w, A)
    ids = list(X)
    Z = (np.array([X[i] for i in ids], float) - m["mu"]) / m["sd"]
    p = m["risk"].predict_proba(Z)[:, 1]
    contrib = Z * m["risk"].coef_[0]
    out = []
    for k, i in enumerate(ids):
        out.append({"vessel": i, "score": float(p[k]), "features": dict(zip(RISK_FEATURES, X[i])),
                    "contributions": {f: float(c) for f, c in zip(RISK_FEATURES, contrib[k]) if abs(c) > 0.05},
                    "gap_hours": float(sum(g["hours"] for g in A["gaps"] if g["vessel"] == i))})
    out.sort(key=lambda r: -r["score"])
    return out


def precision_at(ranked, truth_set, k):
    top = ranked[:k]
    hit = sum(1 for i in top if i in truth_set)
    return hit / k, hit / max(len(truth_set), 1)


def auc(scores, labels):
    return float(roc_auc_score(labels, scores)) if 0 < sum(labels) < len(labels) else float("nan")
