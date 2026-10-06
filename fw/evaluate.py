"""Held-out evaluation: three simulated fortnights the models never trained on and the demo never uses (seeds 21, 31, 41,
no scripted demo vessels). Each detector against the generator's truth and against the simple rule a centre might use.

    python -m fw.evaluate > docs/evaluation.md
"""
import time

import numpy as np

from . import engine as E, world as W

SEEDS = (21, 31, 41)


def pr(tp, alerts, positives):
    return f"{tp / alerts:.2f}" if alerts else "–", f"{tp / positives:.2f}" if positives else "–"


def main():
    t0 = time.time()
    m = E.models()
    out = ["# Evaluation", "", f"Three simulated fortnights the models never trained on (seeds {', '.join(map(str, SEEDS))}; the models learned from past cases in seeds",
           f"{', '.join(map(str, E.TRAIN_SEEDS))}). Each has about 110 fishing boats, four carriers, eight cargo ships and roughly a dozen rule-breakers. Reproduce with",
           "`python -m fw.evaluate`.", ""]
    gap_rows, sar_rows, ts_rows, risk_rows = [], [], [], []
    for seed in SEEDS:
        w = W.build(seed, demo=False)
        V = w["vessels"]
        A = E.analyse(w, W.SLOTS, m["gap"])
        G = E.label_gaps(A["gaps"], w["truth"])
        y = np.array([g["deliberate"] for g in G])
        p = np.array([g["p"] for g in G])
        h = np.array([g["hours"] for g in G])
        for name, flag in (("Gap model, score over 0.5", p > 0.5), ("Gap model, score over 0.2", p > 0.2), ("Rule: any silence of 6 hours or more", h >= E.BASELINE_GAP_H)):
            tp = int((flag & y).sum())
            gap_rows.append((seed, name, int(flag.sum()), tp, int(y.sum()), *pr(tp, int(flag.sum()), int(y.sum())), f"{flag.sum() / W.DAYS:.1f}"))
        # SAR: what the unmatched targets were, and whether the silent boat named first was the right one
        dark_slots = {}
        for i, a, b, _ in w["truth"]["dark"]:
            dark_slots.setdefault(i, []).append((a, b))
        is_dark = lambda i, s: any(a <= s < b for a, b in dark_slots.get(i, []))     # noqa: E731
        F = A["sar"]
        flagged = [d for d in F if d["dark"]]
        unmatched = [d for d in F if d["matched"] is None]
        real = [d for d in flagged if d["truth"] is not None]
        deliberate = [d for d in real if is_dark(d["truth"], d["slot"])]
        named = [d for d in deliberate if d["candidates"]]
        right = [d for d in named if d["candidates"][0] == d["truth"]]
        seen_dark = sum(1 for d in F if d["truth"] is not None and is_dark(d["truth"], d["slot"]))
        sar_rows.append((seed, len(F), len(unmatched), len(flagged), len(real), len(deliberate), seen_dark, len(named), len(right)))
        # transshipments that were not declared
        ts = [x for x in w["truth"]["transship"] if not x[5]]
        overlaps = lambda c, a, b: [x for x in ts if x[1] == c and x[2] <= b and x[3] >= a]     # noqa: E731
        enc = [e for e in A["encounters"] if not E.declared(w, e[0], e[1], e[2])]
        enc_hit = [e for e in enc if any(x[0] == e[1] for x in overlaps(e[0], e[2], e[3]))]
        dts = A["dark_transship"]
        dts_hit = [d for d in dts if overlaps(d["carrier"], d["s0"], d["s1"])]
        dts_partner = [d for d in dts_hit if d["partners"][:1] and d["partners"][0] in {x[0] for x in overlaps(d["carrier"], d["s0"], d["s1"])}]
        found = {x[:4] for x in ts if any(e[0] == x[1] and e[1] == x[0] and e[2] <= x[3] and e[3] >= x[2] for e in enc) or any(d["carrier"] == x[1] and d["s0"] <= x[3] and d["s1"] >= x[2] for d in dts)}
        dist = E.encounters(A["P"], A["S"], V, W.SLOTS, rule="distance")
        dist_hit = [e for e in dist if any((x[0], x[1]) in ((e[0], e[1]), (e[1], e[0])) and x[2] <= e[3] and x[3] >= e[2] for x in ts)]
        declared_seen = sum(1 for e in A["encounters"] if E.declared(w, e[0], e[1], e[2]))
        ts_rows.append((seed, len(ts), len(enc), len(enc_hit), len(dts), len(dts_hit), len(dts_partner), len(found), len(dist), len(dist_hit), declared_seen))
        R = E.score_risk(w, A, m)
        ill = set(w["truth"]["illicit"])
        by_h = sorted(R, key=lambda r: -r["gap_hours"])
        p10, r10 = E.precision_at([r["vessel"] for r in R], ill, 10)
        b10, br10 = E.precision_at([r["vessel"] for r in by_h], ill, 10)
        risk_rows.append((seed, len(ill), p10, r10, E.auc([r["score"] for r in R], [r["vessel"] in ill for r in R]), b10, br10, E.auc([r["gap_hours"] for r in R], [r["vessel"] in ill for r in R])))
    out += ["## AIS silences", "", "A silence is two hours or more without a message. Deliberate means the boat switched AIS off on purpose for at least half of it.", "",
            "| Fortnight | Detector | Alerts | Deliberate among them | Deliberate in all | Precision | Recall | Alerts a day |", "|---|---|---|---|---|---|---|---|"]
    out += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} |" for r in gap_rows]
    out += ["", "## SAR targets with no AIS", "", "| Fortnight | SAR detections | Unmatched to AIS | Flagged dark (18 m or longer) | Real vessels among them | Deliberately dark among them | Deliberately dark boats the radar saw | Named a candidate | Named the right boat first |",
            "|---|---|---|---|---|---|---|---|---|"]
    out += [f"| {' | '.join(map(str, r))} |" for r in sar_rows]
    out += ["", "## Undeclared transshipments", "", "| Fortnight | Undeclared transshipments | AIS encounters flagged | of them real | Loitering carriers flagged as possible dark meetings | of them real | partner named first correctly | Transshipments found either way | Distance-only rule events | of them real | Declared transshipments seen and set aside |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    out += [f"| {' | '.join(map(str, r))} |" for r in ts_rows]
    out += ["", "## Ranking the fleet", "", "Precision and recall in the top ten fishing boats, and the area under the ROC curve over all of them.", "",
            "| Fortnight | Rule-breakers | Risk model P@10 | R@10 | AUC | By silent hours P@10 | R@10 | AUC |", "|---|---|---|---|---|---|---|---|"]
    out += [f"| {r[0]} | {r[1]} | {r[2]:.1f} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.1f} | {r[6]:.2f} | {r[7]:.2f} |" for r in risk_rows]
    out += ["", f"Run time {time.time() - t0:.0f} s on one core, training included."]
    print("\n".join(out))


if __name__ == "__main__":
    main()
