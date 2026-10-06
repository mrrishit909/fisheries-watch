# Evaluation

Three simulated fortnights the models never trained on (seeds 21, 31, 41; the models learned from past cases in seeds
101, 102, 103). Each has about 110 fishing boats, four carriers, eight cargo ships and roughly a dozen rule-breakers. Reproduce with
`python -m fw.evaluate`.

## AIS silences

A silence is two hours or more without a message. Deliberate means the boat switched AIS off on purpose for at least half of it.

| Fortnight | Detector | Alerts | Deliberate among them | Deliberate in all | Precision | Recall | Alerts a day |
|---|---|---|---|---|---|---|---|
| 21 | Gap model, score over 0.5 | 8 | 4 | 14 | 0.50 | 0.29 | 0.6 |
| 21 | Gap model, score over 0.2 | 22 | 12 | 14 | 0.55 | 0.86 | 1.6 |
| 21 | Rule: any silence of 6 hours or more | 169 | 14 | 14 | 0.08 | 1.00 | 12.1 |
| 31 | Gap model, score over 0.5 | 11 | 6 | 13 | 0.55 | 0.46 | 0.8 |
| 31 | Gap model, score over 0.2 | 25 | 13 | 13 | 0.52 | 1.00 | 1.8 |
| 31 | Rule: any silence of 6 hours or more | 167 | 13 | 13 | 0.08 | 1.00 | 11.9 |
| 41 | Gap model, score over 0.5 | 13 | 11 | 16 | 0.85 | 0.69 | 0.9 |
| 41 | Gap model, score over 0.2 | 28 | 15 | 16 | 0.54 | 0.94 | 2.0 |
| 41 | Rule: any silence of 6 hours or more | 179 | 15 | 16 | 0.08 | 0.94 | 12.8 |

## SAR targets with no AIS

| Fortnight | SAR detections | Unmatched to AIS | Flagged dark (18 m or longer) | Real vessels among them | Deliberately dark among them | Deliberately dark boats the radar saw | Named a candidate | Named the right boat first |
|---|---|---|---|---|---|---|---|---|
| 21 | 860 | 123 | 58 | 34 | 9 | 11 | 9 | 6 |
| 31 | 866 | 127 | 58 | 39 | 7 | 8 | 7 | 6 |
| 41 | 880 | 131 | 66 | 41 | 10 | 10 | 9 | 5 |

## Undeclared transshipments

| Fortnight | Undeclared transshipments | AIS encounters flagged | of them real | Loitering carriers flagged as possible dark meetings | of them real | partner named first correctly | Transshipments found either way | Distance-only rule events | of them real | Declared transshipments seen and set aside |
|---|---|---|---|---|---|---|---|---|---|---|
| 21 | 8 | 1 | 1 | 21 | 6 | 4 | 7 | 133 | 1 | 4 |
| 31 | 7 | 3 | 3 | 19 | 3 | 2 | 6 | 168 | 3 | 4 |
| 41 | 5 | 2 | 2 | 23 | 3 | 1 | 5 | 147 | 2 | 4 |

## Ranking the fleet

Precision and recall in the top ten fishing boats, and the area under the ROC curve over all of them.

| Fortnight | Rule-breakers | Risk model P@10 | R@10 | AUC | By silent hours P@10 | R@10 | AUC |
|---|---|---|---|---|---|---|---|
| 21 | 11 | 0.7 | 0.64 | 0.96 | 0.6 | 0.55 | 0.88 |
| 31 | 11 | 0.8 | 0.73 | 0.99 | 0.4 | 0.36 | 0.69 |
| 41 | 11 | 1.0 | 0.91 | 1.00 | 0.4 | 0.36 | 0.83 |

Run time 12 s on one core, training included.
