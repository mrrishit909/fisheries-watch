"""The simulated sea. A 400 x 300 km region off a coast with two ports, a marine protected area over the best fishing ground,
a fleet of fishing vessels, reefer carriers and passing cargo ships, fourteen days at 10-minute steps. Each vessel follows
an itinerary; a few do something illegal (fish inside the protected area with AIS off, or meet a carrier at sea to
transship). AIS reaches the ground only some of the time: well near the coast, worse offshore and badly in the east, and
transponders also fail on their own. A SAR satellite passes twice a day, misses small boats, misplaces what it sees and
reports wave clutter. The truth (who was dark on purpose, who met whom) is kept apart and never shown to a detector.

Units: km (x east from the coast, y north), knots, 10-minute slots from day 0 00:00.
"""
import functools
import math

import numpy as np

W_KM, H_KM = 400.0, 300.0
DAYS = 14
SLOT_MIN = 10
SLOTS = DAYS * 24 * 60 // SLOT_MIN
KM_PER_SLOT_PER_KN = 1.852 / 6          # one knot for ten minutes
PORTS = [(4.0, 80.0), (4.0, 220.0)]
MPA = {"name": "Coral Bank Marine Protected Area", "x": 230.0, "y": 150.0, "r": 35.0}
GROUNDS = [(232.0, 150.0, 26.0), (120.0, 70.0, 18.0), (150.0, 235.0, 18.0), (335.0, 95.0, 22.0)]   # (x, y, spread); ground 0 sits on the reserve
HOLE_X = 300.0                          # east of here satellite AIS is poor
LOAD_SLOT = 12 * 144                    # the region loads at day 12 00:00; the last 48 hours arrive by poll
SAR_HOURS = (6, 18)
OWNERS = ["Meridian Holdings", "Bluewater Fisheries", "Kestrel Marine", "Atlas Seafood", "Northline Fishing", "Pelagic Partners", "Sunward Maritime",
          "Coastal Harvest", "Tidewater Co-op", "Ironbay Trawlers", "Silverfin Ltd", "Harbor Light Fishing", "Oceanic Venture", "Delta Catch",
          "Granite Shoals", "Albatross Shipping"]
FLAGS = ["Domestic"] * 7 + ["Norland", "Vestria", "Calvania", "Port Reyes (open registry)", "Sao Marco (open registry)"]
OPEN_REGISTRIES = {"Port Reyes (open registry)", "Sao Marco (open registry)"}
NAMES = ["Morning Tide", "Southern Star", "Bay Sister", "Sea Lark", "Cape Runner", "Silver Wake", "North Wind", "Blue Heron", "Gull Rock",
         "Kelp Queen", "Storm Petrel", "Amber Reef", "Grey Seal", "Harbor Rose", "Ocean Pearl", "Saltwind"]
DEMO = {"name": "Southern Star 7", "carrier": "Polar Crown", "benign": "Morning Tide 3", "pair": ("Bay Sister 1", "Bay Sister 2")}


def reception(x, y):
    """Chance that one AIS message from (x, y) reaches the ground: terrestrial near the coast, satellite beyond, a poor patch east."""
    return np.where(x < 70, 0.92, np.where(x > HOLE_X, 0.12, 0.6))


def in_mpa(x, y):
    return (x - MPA["x"]) ** 2 + (y - MPA["y"]) ** 2 < MPA["r"] ** 2


def dist_port(x, y):
    return np.min([np.hypot(x - px, y - py) for px, py in PORTS], axis=0)


class Track:
    """Renders an itinerary slot by slot."""

    def __init__(self, r, start):
        self.r = r
        self.pos = np.zeros((SLOTS, 2))
        self.speed = np.zeros(SLOTS)
        self.dark = np.zeros(SLOTS, bool)
        self.fishing = np.zeros(SLOTS, bool)
        self.away = np.zeros(SLOTS, bool)        # outside the region: no AIS reaches this station, no SAR sees it
        self.t = 0
        self.p = np.array(start, float)
        self.is_dark = False

    def _put(self, kn, fishing=False, away=False):
        if self.t < SLOTS:
            self.pos[self.t], self.speed[self.t], self.dark[self.t], self.fishing[self.t], self.away[self.t] = self.p, kn, self.is_dark, fishing, away
            self.t += 1

    def leave(self, n, entry):
        """Out of the region for n slots, then back in at `entry`."""
        for _ in range(int(n)):
            self._put(0.0, away=True)
        self.p = np.array(entry, float)

    def go(self, target, kn, dark_after_km=None, light_after_km=None):
        target = np.array(target, float)
        step = kn * KM_PER_SLOT_PER_KN
        start = self.p.copy()
        while self.t < SLOTS:
            d = np.linalg.norm(target - self.p)
            if dark_after_km is not None and np.linalg.norm(self.p - start) >= dark_after_km:
                self.is_dark = True
            if light_after_km is not None and np.linalg.norm(self.p - start) >= light_after_km:
                self.is_dark = False
            if d <= step:
                self.p = target.copy()
                self._put(d / KM_PER_SLOT_PER_KN)
                return
            self.p = self.p + (target - self.p) / d * step
            self._put(kn)

    def wait(self, n, drift_kn=0.3):
        for _ in range(int(n)):
            self.p = self.p + self.r.normal(0, drift_kn * KM_PER_SLOT_PER_KN * 0.7, 2)
            self._put(drift_kn * abs(self.r.normal(1, 0.3)))

    def wait_until(self, t, drift_kn=0.3):
        self.wait(max(0, t - self.t), drift_kn)

    def fish(self, centre, spread, n, avoid_mpa=True):
        """A slow, turning walk around a centre; a lawful boat turns back at the reserve's edge."""
        heading = self.r.uniform(0, 2 * math.pi)
        centre = np.array(centre, float)
        for _ in range(int(n)):
            kn = self.r.uniform(2.0, 4.0)
            pull = centre - self.p
            heading += self.r.normal(0, 0.5) + 0.15 * math.sin(math.atan2(pull[1], pull[0]) - heading) * min(1, np.linalg.norm(pull) / spread)
            nxt = self.p + kn * KM_PER_SLOT_PER_KN * np.array([math.cos(heading), math.sin(heading)])
            if avoid_mpa and math.hypot(nxt[0] - MPA["x"], nxt[1] - MPA["y"]) < MPA["r"] + 1.5:     # a margin, so a pair partner stays out too
                heading += math.pi
                nxt = self.p + kn * KM_PER_SLOT_PER_KN * np.array([math.cos(heading), math.sin(heading)])
            nxt = np.clip(nxt, [8, 2], [W_KM - 2, H_KM - 2])
            self.p = nxt
            self._put(kn, True)

    def follow(self, other, until, offset):
        while self.t < min(until, SLOTS):
            self.p = other.pos[self.t] + offset + self.r.normal(0, 0.1, 2)
            self._put(other.speed[self.t], other.fishing[self.t])

    def done(self):
        while self.t < SLOTS:
            self.wait(1, 0.05)


def ground_point(r, g, outside_mpa=True):
    x, y, s = GROUNDS[g]
    for _ in range(100):
        p = (x + r.normal(0, s), y + r.normal(0, s))
        if 10 < p[0] < W_KM - 5 and 5 < p[1] < H_KM - 5 and (not outside_mpa or not in_mpa(*p) and math.hypot(p[0] - MPA["x"], p[1] - MPA["y"]) > MPA["r"] + 3):
            return p
    return (x + s * 2, y)


def mpa_edge(r, angle, out_km):
    return (MPA["x"] + (MPA["r"] + out_km) * math.cos(angle), MPA["y"] + (MPA["r"] + out_km) * math.sin(angle))


@functools.lru_cache(maxsize=16)
def build(seed=15, demo=True):
    """The world. -> dict with vessels (registry), tracks, received AIS, SAR detections and the truth."""
    r = np.random.default_rng(seed)
    n_fish, n_carrier, n_cargo = 110, 4, 8
    vessels = []
    owners_used = r.choice(len(OWNERS), n_fish)
    for i in range(n_fish):
        flag = FLAGS[r.integers(len(FLAGS))]
        vessels.append({"kind": "fishing", "name": f"{NAMES[i % len(NAMES)]} {i // len(NAMES) + 1}", "length": float(np.clip(r.lognormal(math.log(28), 0.35), 12, 70)),
                        "flag": flag, "owner": OWNERS[owners_used[i]], "gear": r.choice(["trawl", "longline", "purse seine"]), "licensed": bool(r.random() < (0.92 if flag == "Domestic" else 0.45))})
    for i in range(n_carrier):
        vessels.append({"kind": "carrier", "name": ["Polar Crown", "Ice Maiden", "Frost Harbor", "Nordic Reefer"][i], "length": float(r.uniform(90, 130)), "flag": FLAGS[r.integers(7, len(FLAGS))],
                        "owner": OWNERS[r.integers(len(OWNERS))], "gear": "reefer", "licensed": True})
    for i in range(n_cargo):
        vessels.append({"kind": "cargo", "name": f"Cargo {['Atlas', 'Borealis', 'Corsair', 'Dorado', 'Elan', 'Fjord', 'Galleon', 'Halcyon'][i]}", "length": float(r.uniform(150, 260)),
                        "flag": FLAGS[r.integers(len(FLAGS))], "owner": "Liner Services", "gear": "cargo", "licensed": True})
    for k, v in enumerate(vessels):
        v["idx"] = k
        v["mmsi"] = int(200000000 + seed * 1000 + k)
    fish = [v for v in vessels if v["kind"] == "fishing"]
    names = {v["name"]: v["idx"] for v in vessels}
    # who breaks the rules: tilted towards unlicensed boats, open registries and two owner networks
    bad_owners = set(r.choice(OWNERS[:8], 2, replace=False).tolist())
    weight = np.array([1 + 3 * (not v["licensed"]) + 2 * (v["flag"] in OPEN_REGISTRIES) + 3 * (v["owner"] in bad_owners) for v in fish], float)
    illicit = set(r.choice([v["idx"] for v in fish], 11, replace=False, p=weight / weight.sum()).tolist())
    if demo:
        illicit.add(names[DEMO["name"]])
        illicit.discard(names[DEMO["benign"]])
        illicit.discard(names[DEMO["pair"][0]])
        illicit.discard(names[DEMO["pair"][1]])
    # the registry's record of past violations: noisy, more likely for networks that do break the rules
    for v in vessels:
        v["prior_violations"] = int(r.random() < (0.35 if v["idx"] in illicit else 0.06))
    truth = {"dark": [], "transship": [], "incursion": [], "illicit": sorted(illicit), "outages": []}
    tracks = {}
    carriers = [v["idx"] for v in vessels if v["kind"] == "carrier"]
    # carriers: port runs, benign waiting offshore, and the meetings booked below
    meetings = []
    for c in carriers:
        meetings.append([])
    pairs = []
    pool = [v["idx"] for v in fish if v["idx"] not in illicit]
    if demo:
        pairs.append((names[DEMO["pair"][0]], names[DEMO["pair"][1]]))
        pool = [i for i in pool if i not in pairs[0]]
    r.shuffle(pool)
    while len(pairs) < 6:
        pairs.append((pool.pop(), pool.pop()))
    followers = {b: a for a, b in pairs}

    # each illicit boat gets one or two timed events before the region loads: a dark incursion or a meeting with a carrier
    events, t_meet = {}, {}
    for i in sorted(illicit):
        times = sorted(r.choice(np.arange(144, LOAD_SLOT - 230, 12), int(r.integers(1, 3)), replace=False).tolist())
        evs = []
        for te in times:
            if r.random() < 0.5:
                evs.append(("incursion", int(te)))
                continue
            c = int(r.choice([x for x in carriers if not (demo and x == names[DEMO["carrier"]])]))
            if any(abs(te - s0) < 220 for s0, _, _ in meetings[carriers.index(c)]):
                evs.append(("incursion", int(te)))
                continue
            dur = int(r.integers(18, 42))
            ang = r.uniform(0, 2 * math.pi)
            pt = (float(np.clip(MPA["x"] + (MPA["r"] + r.uniform(15, 50)) * math.cos(ang), 90, 380)), float(np.clip(MPA["y"] + (MPA["r"] + r.uniform(15, 50)) * math.sin(ang), 20, 280)))
            meetings[carriers.index(c)].append((int(te), dur, pt))
            t_meet[(i, int(te))] = (c, int(te), dur, pt)
            evs.append(("transship", int(te)))
        events[i] = evs
    # honest, licensed boats that transship with a declared authorisation and an observer, AIS on throughout
    honest = [v["idx"] for v in fish if v["idx"] not in illicit and v["licensed"] and v["idx"] not in followers and v["idx"] not in {a for a, _ in pairs}
              and not (demo and v["name"] in (DEMO["benign"],))]
    authorisations = []
    for i in r.choice(honest, 4, replace=False):
        i = int(i)
        for _ in range(20):
            te = int(r.choice(np.arange(144, LOAD_SLOT - 230, 12)))
            c = int(r.choice([x for x in carriers if not (demo and x == names[DEMO["carrier"]])]))
            if all(abs(te - s0) >= 220 for s0, _, _ in meetings[carriers.index(c)]):
                break
        dur = int(r.integers(18, 42))
        ang = r.uniform(0, 2 * math.pi)
        pt = (float(np.clip(MPA["x"] + (MPA["r"] + r.uniform(15, 50)) * math.cos(ang), 90, 380)), float(np.clip(MPA["y"] + (MPA["r"] + r.uniform(15, 50)) * math.sin(ang), 20, 280)))
        meetings[carriers.index(c)].append((te, dur, pt))
        t_meet[(i, te)] = (c, te, dur, pt)
        events[i] = [("authorised", te)]
        authorisations.append({"vessel": i, "carrier": c, "day": te // 144})
    if demo:
        c = names[DEMO["carrier"]]
        meetings[carriers.index(c)].append((13 * 144 + 13 * 6, 33, (292.0, 196.0)))      # day 13 13:00 to 18:30
        t_meet[(names[DEMO["name"]], 0)] = (c, 13 * 144 + 13 * 6, 33, (292.0, 196.0))
        events[names[DEMO["name"]]] = [("demo", 12 * 144)]

    for c in carriers:
        tr = Track(r, PORTS[r.integers(2)])
        for start, dur, pt in sorted(meetings[carriers.index(c)]):
            while tr.t < start - 200:
                # fill time before the meeting: port calls and idle waits offshore
                if r.random() < 0.5:
                    tr.go(PORTS[r.integers(2)], 11)
                    tr.wait(min(int(r.integers(40, 120)), max(1, start - 200 - tr.t)))
                else:
                    tr.go(ground_point(r, int(r.integers(1, 4))), 10)
                    tr.wait(min(int(r.integers(12, 40)), max(1, start - 200 - tr.t)), 0.6)
            tr.go(pt, 11)
            tr.wait_until(start)
            tr.wait(dur, 0.4)
        while tr.t < SLOTS:
            if r.random() < 0.5:
                tr.go(PORTS[r.integers(2)], 11)
                tr.wait(r.integers(40, 120))
            else:
                tr.go(ground_point(r, int(r.integers(0, 4))), 10)
                tr.wait(r.integers(12, 40), 0.6)            # benign: a carrier idling offshore, nobody alongside
        tracks[c] = tr

    for v in vessels:
        i = v["idx"]
        if v["kind"] == "cargo":
            a = (r.uniform(120, 400), 0.0)
            tr = Track(r, a)
            tr.leave(r.integers(0, 100), a)
            while tr.t < SLOTS:
                b = (r.uniform(120, 400), H_KM if a[1] == 0 else 0.0)
                tr.go(b, r.uniform(13, 17))
                a = (r.uniform(120, 400), b[1])
                tr.leave(r.integers(30, 200), a)
            tracks[i] = tr
        if v["kind"] != "fishing" or i in followers:
            continue
        home = PORTS[r.integers(2)]
        tr = Track(r, home)
        tr.wait(r.integers(0, 60))
        todo = list(events.get(i, []))
        while tr.t < SLOTS:
            if demo and i == names[DEMO["benign"]] and tr.t > 11 * 144:
                tr.go(GROUNDS[3][:2], 10)                              # fishes the poorly covered east through the demo window
                tr.fish(GROUNDS[3][:2], GROUNDS[3][2], SLOTS)
                break
            g = int(r.choice(4, p=[0.35, 0.25, 0.25, 0.15]))
            nxt = todo[0][1] if todo else SLOTS * 2
            if todo and tr.t >= nxt - 150:
                kind, te = todo.pop(0)
                if kind == "demo":
                    demo_itinerary(tr, r, truth, i, t_meet[(i, 0)], tracks)
                    break
                if kind == "incursion":
                    incursion(tr, r, truth, i)
                else:
                    meet(tr, r, truth, i, t_meet[(i, te)], tracks, g, authorised=kind == "authorised")
            else:
                budget = nxt - 150 - tr.t
                if budget < 120:
                    tr.wait(max(1, min(budget, 60)))
                    continue
                tr.go(ground_point(r, g), r.uniform(8, 11))
                tr.fish(GROUNDS[g][:2], GROUNDS[g][2], min(int(r.integers(60, 260)), max(10, nxt - 150 - tr.t - 70)))
            tr.go(home, r.uniform(8, 11))
            tr.wait(min(int(r.integers(30, 150)), max(1, (todo[0][1] - 150 - tr.t) if todo else 999)))
        tracks[i] = tr
    for lead, fol in pairs:
        tr = Track(r, tracks[lead].pos[0])
        tr.follow(tracks[lead], SLOTS, np.array([r.uniform(0.4, 1.0), r.uniform(-0.6, 0.6)]))
        tracks[fol] = tr

    # AIS as received: dark on purpose, transponder outages, then the reception lottery
    ais = []
    for v in vessels:
        i = v["idx"]
        tr = tracks[i]
        outage = np.zeros(SLOTS, bool)
        for d in range(DAYS):
            if r.random() < (0.045 if v["kind"] != "cargo" else 0.01):
                s0 = d * 144 + int(r.integers(0, 144))
                n = int(r.integers(18, 96))
                outage[s0:s0 + n] = True
                truth["outages"].append((i, s0, min(SLOTS, s0 + n)))
        if demo and v["name"] in (DEMO["benign"], DEMO["carrier"], DEMO["name"]):
            outage[:] = False                       # the demo's silences are the scripted ones
        rec = (~tr.dark) & (~outage) & (~tr.away) & (r.random(SLOTS) < reception(tr.pos[:, 0], tr.pos[:, 1]))
        for s in np.flatnonzero(rec):
            ais.append((i, int(s), float(tr.pos[s, 0]), float(tr.pos[s, 1]), float(tr.speed[s])))
    ais.sort(key=lambda a: (a[1], a[0]))

    # SAR passes: twice a day, a 160 km swath, mostly tasked over the reserve
    sar = []
    for d in range(DAYS):
        for hh in SAR_HOURS:
            s = d * 144 + hh * 6
            cx = float(np.clip(MPA["x"] + r.normal(0, 40), 80, 320))
            if demo and d == 13:
                cx = 230.0 if hh == 6 else 280.0          # the demo's last day: tasked over the reserve, then over the carrier
            box = (cx - 80, cx + 80)
            for v in vessels:
                p = tracks[v["idx"]].pos[s]
                if not (box[0] <= p[0] <= box[1]) or dist_port(p[0], p[1]) < 4 or tracks[v["idx"]].away[s]:
                    continue
                pdet = 0.92 if v["length"] >= 40 else 0.7 if v["length"] >= 20 else 0.4
                if demo and d == 13 and v["name"] == DEMO["name"]:
                    pdet = 1.0                              # the demo's story needs both sightings; stated on the page
                if r.random() < pdet:
                    sar.append({"slot": s, "x": float(p[0] + r.normal(0, 0.3)), "y": float(p[1] + r.normal(0, 0.3)), "length": float(v["length"] * r.lognormal(0, 0.25)), "truth": v["idx"]})
            for _ in range(r.poisson(3)):
                sar.append({"slot": s, "x": float(r.uniform(*box)), "y": float(r.uniform(5, H_KM - 5)), "length": float(r.lognormal(math.log(14), 0.4)), "truth": None})
    return {"seed": seed, "authorisations": authorisations, "vessels": vessels, "tracks": tracks, "ais": ais, "sar": sar, "truth": truth, "names": names, "pairs": pairs}


def first_dark(tr, since):
    k = np.flatnonzero(tr.dark[since:tr.t])
    return since + int(k[0]) if len(k) else tr.t


def incursion(tr, r, truth, i):
    """AIS off some kilometres before the reserve, fish inside, AIS back on after leaving."""
    ang = r.uniform(0, 2 * math.pi)
    tr.go(mpa_edge(r, ang, r.uniform(8, 45)), 10)
    t_off = tr.t
    tr.is_dark = True
    inside = (MPA["x"] + 0.5 * MPA["r"] * math.cos(ang + r.normal(0, 0.4)), MPA["y"] + 0.5 * MPA["r"] * math.sin(ang + r.normal(0, 0.4)))
    tr.go(inside, 8)
    t_in = tr.t
    tr.fish(inside, 8, r.integers(36, 120), avoid_mpa=False)
    truth["incursion"].append((i, t_in, tr.t))
    tr.go(mpa_edge(r, ang + r.normal(0, 0.6), r.uniform(8, 40)), 9)
    tr.is_dark = False
    truth["dark"].append((i, t_off, tr.t, "incursion"))


def meet(tr, r, truth, i, booking, tracks, g, authorised=False):
    """Cross to a carrier and lie alongside; six in ten unauthorised boats go dark for it."""
    c, start, dur, pt = booking
    dark = not authorised and r.random() < 0.6
    trip = float(np.linalg.norm(np.array(pt) - tr.p))
    tr.wait(max(0, start - tr.t - int(trip / (9 * KM_PER_SLOT_PER_KN)) - 4))
    t0 = tr.t
    tr.go(pt, 9, dark_after_km=max(0.0, trip - r.uniform(15, 30)) if dark else None)
    tr.wait_until(start)
    a = tr.t
    tr.follow(tracks[c], start + dur, np.array([0.12, 0.08]))
    if tr.t - a >= 6:
        truth["transship"].append((i, c, a, tr.t, pt, authorised))
    away = ground_point(r, g)
    if dark:
        tr.go(away, 9, light_after_km=r.uniform(10, 25))
        tr.is_dark = False
        truth["dark"].append((i, first_dark(tr, t0), tr.t, "transship"))
    else:
        tr.go(away, 9)


def demo_itinerary(tr, r, truth, i, meet, tracks):
    """Southern Star 7, the last two days: approaches the reserve from the west, goes dark eight kilometres out, fishes inside
    through the night (a SAR pass at 06:00 sees it), crosses to the carrier Polar Crown, lies alongside for five hours, and
    comes back on air heading home. Meeting: day 13 13:00 to 18:30."""
    c, start, dur, pt = meet
    tr.go((150.0, 150.0), 10)
    tr.fish((150.0, 150.0), 10, 12 * 144 + 18 * 6 - tr.t)            # fishes lawfully west of the reserve until day 12 18:00
    edge = mpa_edge(r, math.pi, 8.0)
    tr.go(edge, 10)
    t_off = tr.t
    tr.is_dark = True
    tr.go((222.0, 152.0), 8)
    t_in = tr.t
    tr.fish((222.0, 152.0), 6, 13 * 144 + 7 * 6 - tr.t, avoid_mpa=False)   # until day 13 07:00
    truth["incursion"].append((i, t_in, tr.t))
    tr.go(pt, 9)
    tr.wait_until(start)
    from_carrier = np.array([0.12, 0.08])
    # alongside: follow the carrier's (already rendered) position
    tr.follow(tracks[c], start + dur, from_carrier)
    truth["transship"].append((i, c, start, start + dur, pt, False))
    tr.go((260.0, 230.0), 9, light_after_km=6)
    tr.is_dark = False
    truth["dark"].append((i, t_off, tr.t, "incursion and transship"))
    tr.go(PORTS[1], 10)
    tr.done()
