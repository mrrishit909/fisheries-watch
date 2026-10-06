"""The sea, the detectors and their scoring, without a database."""
import numpy as np

from fw import engine as E, world as W


def test_world_is_reproducible_and_dark_means_silent():
    a, b = W.build(15), W.build.__wrapped__(15)
    assert a["ais"][:50] == b["ais"][:50] and len(a["sar"]) == len(b["sar"])
    i = a["names"]["Southern Star 7"]
    tr = a["tracks"][i]
    got = {s for v, s, *_ in a["ais"] if v == i}
    assert not (got & set(np.flatnonzero(tr.dark).tolist()))                 # no message while dark
    assert W.in_mpa(*tr.pos[13 * 144 + 6 * 6]) and tr.dark[13 * 144 + 6 * 6]  # inside the reserve, dark, at the 06:00 pass


def test_lawful_boats_do_not_fish_inside_the_reserve():
    w = W.build(21, demo=False)
    ill = set(w["truth"]["illicit"])
    for v in w["vessels"]:
        if v["kind"] == "fishing" and v["idx"] not in ill:
            tr = w["tracks"][v["idx"]]
            assert not (tr.fishing & W.in_mpa(tr.pos[:, 0], tr.pos[:, 1])).any(), v["name"]


def test_coverage_is_learned_from_the_data():
    w = W.build(21, demo=False)
    rx = E.by_vessel(w["ais"], len(w["vessels"]))
    cov = E.coverage(rx, W.SLOTS)
    near, east = cov[:3].mean(), cov[16:].mean()
    assert near > 0.7 and east < 0.4 and near > east


def test_reachability():
    g = {"s0": 0, "s1": 60, "open": False, "x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}
    v = 15 * W.KM_PER_SLOT_PER_KN
    assert E.reachable(g, 25 * v, 0, 0, 60) and not E.reachable(g, 35 * v, 0, 0, 60)
    assert not E.reachable(g, 25 * v, 0, 0, 10)                             # it could not be that far that soon


def test_detectors_beat_the_simple_rules_on_an_unseen_fortnight():
    m = E.models()
    w = W.build(41, demo=False)
    A = E.analyse(w, W.SLOTS, m["gap"])
    G = E.label_gaps(A["gaps"], w["truth"])
    y = np.array([g["deliberate"] for g in G])
    p = np.array([g["p"] for g in G])
    h = np.array([g["hours"] for g in G])
    prec = lambda f: (f & y).sum() / max(f.sum(), 1)     # noqa: E731
    assert prec(p > 0.5) > 3 * prec(h >= E.BASELINE_GAP_H) and (p > 0.5).sum() < (h >= E.BASELINE_GAP_H).sum() / 4
    R = E.score_risk(w, A, m)
    ill = set(w["truth"]["illicit"])
    assert E.precision_at([r["vessel"] for r in R], ill, 10)[0] > E.precision_at([r["vessel"] for r in sorted(R, key=lambda r: -r["gap_hours"])], ill, 10)[0]
    assert all(set(r["contributions"]) <= set(E.RISK_FEATURES) for r in R)


def test_fusion_matches_ais_and_leaves_dark_targets():
    w = W.build(15)
    rx = E.by_vessel(w["ais"], len(w["vessels"]))
    F = E.fuse(w, rx, W.SLOTS)
    matched = [d for d in F if d["matched"] is not None]
    assert sum(d["matched"] == d["truth"] for d in matched) / len(matched) > 0.97
    i = w["names"]["Southern Star 7"]
    seen = [d for d in F if d["truth"] == i and d["slot"] in (13 * 144 + 36, 13 * 144 + 108)]
    assert len(seen) == 2 and all(d["dark"] and d["matched"] is None for d in seen)      # including alongside a carrier whose own return was missed
