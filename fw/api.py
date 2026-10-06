"""Public API (modular monolith). Blueprint services map to: AIS ingest (load, poll, checked batches), satellite catalog (SAR
detections), track fusion (SAR against AIS), behaviour model (gap classifier), rendezvous detection, license registry, risk
scoring and case management. Every detector reads what is in the database, never the generator; the generator's truth is
attached to responses only under `simulation_truth`.

    uvicorn fw.api:app          python -m core.jobs fw.api      # the worker
"""
import datetime
import hashlib
import json
import math
import uuid

import numpy as np
from fastapi import Depends, Header
from fastapi.encoders import jsonable_encoder
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from core import audit, db, jobs
from core.app import Ctx, Problem, create_app, run

from . import engine as E, world as W

READ = {"sea:read", "jobs:read"}
WORK = READ | {"tracks:write", "risk:score", "cases:open"}
PERMISSIONS = {"viewer": READ, "analyst": WORK, "supervisor": WORK | {"region:load", "cases:escalate", "audit:read"}}
app = create_app("fisheries-watch", PERMISSIONS)
auth = app.state.auth
IdemKey = Header(None, alias="Idempotency-Key")
DAY0 = datetime.datetime(2026, 9, 22, tzinfo=datetime.timezone.utc)
MAX_KN = 35.0                     # a fix implying more than this from the previous one is flagged as a possible spoof


def ts(slot):
    return DAY0 + datetime.timedelta(minutes=W.SLOT_MIN * int(slot))


def slot_of(t):
    return int(round((t - DAY0).total_seconds() / 60 / W.SLOT_MIN))


def hhmm(slot):
    return ts(slot).strftime("%a %H:%M")


def worker_ctx(job):
    return Ctx(job["tenant_id"], uuid.UUID(job["payload"]["actor_id"]), "worker", "system")


def vid(t, seq):
    return uuid.uuid5(t, f"vessel-{seq}")


def did(t, k):
    return uuid.uuid5(t, f"sar-{k}")


def region(c):
    g = c.execute("SELECT * FROM region").fetchone()
    if not g:
        raise Problem(409, "no_region", "load a region first: POST /v1/region:load")
    return g


def world_from_db(c, t):
    """The monitoring centre's view: registry, received AIS (spoof-flagged fixes left out), SAR detections, declarations."""
    vs = c.execute("""SELECT v.vessel_id, v.seq, v.mmsi, v.kind, v.length_m, v.gear, i.name, i.flag, i.owner, l.licensed, l.prior_violations
                        FROM vessel v JOIN identity i USING (vessel_id) JOIN license l USING (vessel_id) ORDER BY v.seq""").fetchall()
    vessels = [{"idx": r["seq"], "id": r["vessel_id"], "mmsi": r["mmsi"], "kind": r["kind"], "length": float(r["length_m"]), "gear": r["gear"], "name": r["name"], "flag": r["flag"],
                "owner": r["owner"], "licensed": r["licensed"], "prior_violations": r["prior_violations"]} for r in vs]
    seq = {r["vessel_id"]: r["seq"] for r in vs}
    ais = [(seq[r["vessel_id"]], slot_of(r["ts"]), r["x_km"], r["y_km"], r["sog_kn"]) for r in c.execute("SELECT vessel_id, ts, x_km, y_km, sog_kn FROM track_point WHERE NOT ('possible_spoof' = ANY(flags))")]
    ais.sort(key=lambda a: (a[1], a[0]))
    sar = [{"id": r["detection_id"], "slot": slot_of(r["ts"]), "x": r["x_km"], "y": r["y_km"], "length": r["length_m"], "truth": None} for r in c.execute("SELECT * FROM satellite_detection ORDER BY ts")]
    auths = [{"vessel": seq[r["vessel_id"]], "carrier": seq[r["carrier_id"]], "day": (r["day"] - DAY0.date()).days} for r in c.execute("SELECT * FROM transship_authorisation")]
    return {"vessels": vessels, "ais": ais, "sar": sar, "authorisations": auths, "by_seq": {v["idx"]: v for v in vessels}}


def load_slots(c, t, w, s0, s1):
    """Write AIS and SAR between two slots from the generator's feed."""
    rows = [(vid(t, i), ts(s), x, y, sp, [], t) for i, s, x, y, sp in w["ais"] if s0 <= s < s1]
    db.load(c, "track_point", ["vessel_id", "ts", "x_km", "y_km", "sog_kn", "flags", "tenant_id"], rows)
    sar = [(did(t, k), t, ts(d["slot"]), d["x"], d["y"], d["length"]) for k, d in enumerate(w["sar"]) if s0 <= d["slot"] < s1]
    db.load(c, "satellite_detection", ["detection_id", "tenant_id", "ts", "x_km", "y_km", "length_m"], sar)
    return len(rows), len(sar)


def detect(c, t, g):
    """Run every detector on what the database holds at the region's clock; replace the stored events."""
    m = E.models()
    wd = world_from_db(c, t)
    clock = slot_of(g["clock"])
    A = E.analyse(wd, clock, m["gap"])
    V = wd["by_seq"]
    c.execute("DELETE FROM behavior_event")
    c.execute("DELETE FROM rendezvous")
    ev = []
    for gp in A["gaps"]:
        ev.append((uuid.uuid5(t, f"gap-{gp['vessel']}-{gp['s0']}"), t, V[gp["vessel"]]["id"], "ais_gap", ts(gp["s0"]), None if gp["open"] else ts(gp["s1"]), gp["x0"], gp["y0"], gp["p"],
                   Jsonb({"hours": round(gp["hours"], 1), "open": gp["open"], "silence_probability_log10": round(gp["logp"], 1), "reserve_km": round(gp["mpa_km"], 1), "km_moved": round(gp["km"], 1),
                          "end": [round(gp["x1"], 1), round(gp["y1"], 1)], "features": dict(zip(E.GAP_FEATURES, [round(x, 3) for x in gp["f"]]))}), E.VERSION))
    for d in A["sar"]:
        if d["dark"]:
            cands = [V[i]["id"] for i in d["candidates"]]
            ev.append((uuid.uuid5(t, f"dark-{d['id']}"), t, cands[0] if cands else None, "dark_target", ts(d["slot"]), None, d["x"], d["y"], 1.0 if cands else 0.5,
                       Jsonb({"detection_id": str(d["id"]), "length_m": round(d["length"]), "candidates": [str(x) for x in cands], "reserve_km": round(float(E.dist_to_mpa(np.array([[d["x"], d["y"]]]))[0]), 1)}), E.VERSION))
    db.load(c, "behavior_event", ["event_id", "tenant_id", "vessel_id", "kind", "start_ts", "end_ts", "x_km", "y_km", "score", "evidence", "model_version"], ev)
    rz = []
    for a, b, s0, s1, x, y in A["encounters"]:
        rz.append((uuid.uuid5(t, f"enc-{a}-{b}-{s0}"), t, V[a]["id"], V[b]["id"], "encounter", ts(s0), ts(s1), x, y, E.declared(wd, a, b, s0), Jsonb({"hours": round((s1 - s0) / 6, 1)})))
    for d in A["dark_transship"]:
        rz.append((uuid.uuid5(t, f"dts-{d['carrier']}-{d['s0']}"), t, V[d["carrier"]]["id"], V[d["partners"][0]]["id"] if d["partners"] else None, "dark_transship_candidate", ts(d["s0"]), ts(d["s1"]), d["x"], d["y"], False,
                   Jsonb({"hours": round((d["s1"] - d["s0"]) / 6, 1), "partners": [str(V[i]["id"]) for i in d["partners"]], "sar_corroborated": d["sar_corroborated"]})))
    db.load(c, "rendezvous", ["event_id", "tenant_id", "carrier_id", "vessel_id", "kind", "start_ts", "end_ts", "x_km", "y_km", "declared", "evidence"], rz)
    return A, wd


# ---------------------------------------------------------------- region and AIS

class LoadIn(BaseModel):
    seed: int = Field(15, ge=0, le=10 ** 6)


@app.post("/v1/region:load", status_code=202, tags=["region"], summary="(+) Load the region at day 12 00:00: the registry and licences, the reserve, twelve days of received AIS and SAR passes; train the gap and risk models on past cases; run every detector")
def load_region(body: LoadIn, ctx: Ctx = Depends(auth("region:load")), idem: str | None = IdemKey):
    def work(c):
        if c.execute("SELECT 1 FROM region").fetchone():
            raise Problem(409, "region_exists")
        return 202, jobs.enqueue(c, ctx, "region.load", body.model_dump())
    return run(ctx, idem, body, work)


@jobs.handler("region.load")
def load_job(c, job):
    t, seed = job["tenant_id"], job["payload"]["seed"]
    w = W.build(seed, demo=True)
    rid = uuid.uuid5(t, "region")
    c.execute("INSERT INTO region (id, tenant_id, name, model_seed, clock) VALUES (%s,%s,'Coral Coast monitoring region',%s,%s)", [rid, t, seed, ts(W.LOAD_SLOT)])
    V = w["vessels"]
    db.load(c, "vessel", ["vessel_id", "tenant_id", "mmsi", "kind", "length_m", "gear", "seq"], [(vid(t, v["idx"]), t, v["mmsi"], v["kind"], round(v["length"], 1), str(v["gear"]), v["idx"]) for v in V])
    db.load(c, "identity", ["vessel_id", "tenant_id", "name", "flag", "owner", "valid_from"], [(vid(t, v["idx"]), t, v["name"], v["flag"], v["owner"], ts(0) - datetime.timedelta(days=400)) for v in V])
    db.load(c, "license", ["vessel_id", "tenant_id", "zone", "licensed", "prior_violations"], [(vid(t, v["idx"]), t, "Coral Coast EEZ", v["licensed"], v["prior_violations"]) for v in V])
    db.load(c, "transship_authorisation", ["vessel_id", "carrier_id", "day", "tenant_id"], [(vid(t, a["vessel"]), vid(t, a["carrier"]), (DAY0 + datetime.timedelta(days=a["day"])).date(), t) for a in w["authorisations"]])
    c.execute("INSERT INTO protected_area (area_id, tenant_id, name, centre_x_km, centre_y_km, radius_km) VALUES (%s,%s,%s,%s,%s,%s)", [uuid.uuid5(t, "mpa"), t, W.MPA["name"], W.MPA["x"], W.MPA["y"], W.MPA["r"]])
    n_ais, n_sar = load_slots(c, t, w, 0, W.LOAD_SLOT)
    g = region(c)
    m = E.models()
    A, wd = detect(c, t, g)
    audit.record(c, worker_ctx(job), "region.loaded", "region", rid, {"vessels": len(V), "ais": n_ais, "sar": n_sar})
    kinds = [v["kind"] for v in V]
    return {"region_id": rid, "area_id": uuid.uuid5(t, "mpa"), "clock": hhmm(W.LOAD_SLOT), "vessels": {k: kinds.count(k) for k in ("fishing", "carrier", "cargo")}, "ais_messages": n_ais, "sar_detections": n_sar,
            "licensed_fishing": sum(1 for v in V if v["kind"] == "fishing" and v["licensed"]), "declared_transshipments": len(w["authorisations"]),
            "coverage": {"near_coast": round(float(A["coverage"][:3].mean()), 2), "offshore": round(float(A["coverage"][5:15].mean()), 2), "east_of_300km": round(float(A["coverage"][15:].mean()), 2)},
            "models": {"version": E.VERSION, "trained_on": f"past cases from {len(E.TRAIN_SEEDS)} other simulated fortnights", "gap_examples": m["gap_examples"], "deliberate_gaps": m["gap_positive"], "vessels": m["risk_examples"], "rule_breakers": m["risk_positive"],
                       "gap_weights": dict(zip(E.GAP_FEATURES, [round(float(x), 2) for x in m["gap"].coef_[0]])), "risk_weights": dict(zip(E.RISK_FEATURES, [round(float(x), 2) for x in m["risk"].coef_[0]]))},
            "simulation_truth": {"what": "the generator's record, never shown to a detector", "rule_breakers": len(w["truth"]["illicit"]), "deliberate_silences": len(w["truth"]["dark"]), "transshipments": sum(1 for x in w["truth"]["transship"] if not x[5]), "transponder_failures": len(w["truth"]["outages"])}}


class PollIn(BaseModel):
    hours: int = Field(48, ge=1, le=48)


@app.post("/v1/tracks:poll", status_code=202, tags=["tracks"], summary="(+) The next hours of AIS and SAR arrive and every detector runs again; stands in for the feeds")
def poll(body: PollIn, ctx: Ctx = Depends(auth("tracks:write")), idem: str | None = IdemKey):
    def work(c):
        g = region(c)
        if slot_of(g["clock"]) + body.hours * 6 > W.SLOTS:
            raise Problem(422, "end_of_data", "the simulated fortnight ends at day 14 00:00")
        return 202, jobs.enqueue(c, ctx, "tracks.poll", body.model_dump())
    return run(ctx, idem, body, work)


@jobs.handler("tracks.poll")
def poll_job(c, job):
    t = job["tenant_id"]
    g = region(c)
    s0 = slot_of(g["clock"])
    s1 = s0 + job["payload"]["hours"] * 6
    w = W.build(g["model_seed"], demo=True)
    n_ais, n_sar = load_slots(c, t, w, s0, s1)
    c.execute("UPDATE region SET clock = %s", [ts(s1)])
    A, _ = detect(c, t, {**g, "clock": ts(s1)})
    audit.record(c, worker_ctx(job), "tracks.polled", "region", g["id"], {"from": hhmm(s0), "to": hhmm(s1), "ais": n_ais, "sar": n_sar})
    return {"clock": hhmm(s1), "ais_messages": n_ais, "sar_detections": n_sar, "gaps_open_or_closed": len(A["gaps"])}


class Fix(BaseModel):
    mmsi: int
    ts: datetime.datetime
    x_km: float
    y_km: float
    sog_kn: float = Field(ge=0, le=60)


class IngestIn(BaseModel):
    fixes: list[Fix] = Field(min_length=1, max_length=5000)


@app.post("/v1/tracks/ingest", status_code=201, tags=["tracks"], summary="AIS fixes from a partner feed: each is checked (known MMSI, a timezone, not in the future, inside the region, not a duplicate) and a fix that implies an impossible speed from the vessel's last one is kept but flagged as a possible spoof")
def ingest(body: IngestIn, ctx: Ctx = Depends(auth("tracks:write")), idem: str | None = IdemKey):
    def work(c):
        g = region(c)
        by_mmsi = {r["mmsi"]: r["vessel_id"] for r in c.execute("SELECT mmsi, vessel_id FROM vessel")}
        accepted, flagged, rejected = 0, [], []
        for f in body.fixes:
            why = None
            v = by_mmsi.get(f.mmsi)
            if v is None:
                why = "MMSI not in the registry"
            elif f.ts.tzinfo is None:
                why = "timestamp needs a timezone"
            elif f.ts > g["clock"]:
                why = "timestamp is in the future"
            elif not (0 <= f.x_km <= W.W_KM and 0 <= f.y_km <= W.H_KM):
                why = "position outside the region"
            elif c.execute("SELECT 1 FROM track_point WHERE vessel_id = %s AND ts = %s", [v, f.ts]).fetchone():
                why = "duplicate fix"
            if why:
                rejected.append({"mmsi": f.mmsi, "why": why})
                continue
            prev = c.execute("SELECT ts, x_km, y_km FROM track_point WHERE vessel_id = %s AND ts < %s ORDER BY ts DESC LIMIT 1", [v, f.ts]).fetchone()
            flags = []
            if prev:
                hrs = max((f.ts - prev["ts"]).total_seconds() / 3600, 1 / 60)
                kn = math.hypot(f.x_km - prev["x_km"], f.y_km - prev["y_km"]) / 1.852 / hrs
                if kn > MAX_KN:
                    flags = ["possible_spoof"]
                    flagged.append({"mmsi": f.mmsi, "implied_knots": round(kn), "why": "implies an impossible speed from the last fix; kept, flagged, left out of the detectors"})
            c.execute("INSERT INTO track_point (vessel_id, ts, x_km, y_km, sog_kn, flags, tenant_id) VALUES (%s,%s,%s,%s,%s,%s,%s)", [v, f.ts, f.x_km, f.y_km, f.sog_kn, flags, ctx.tenant_id])
            accepted += 1
        audit.record(c, ctx, "tracks.ingested", "region", g["id"], {"accepted": accepted, "flagged": len(flagged), "rejected": len(rejected)})
        return 201, {"accepted": accepted, "flagged": flagged, "rejected": rejected}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- what the detectors found

_cache = {}


def cached_view(c, t, g):
    """Distance-rule meetings and the coverage map depend only on the fixes held; recompute when the clock or the count changes."""
    key = (t, g["clock"], c.execute("SELECT count(*) AS n FROM track_point").fetchone()["n"])
    if key not in _cache:
        wd = world_from_db(c, t)
        clock = slot_of(g["clock"])
        rx = E.by_vessel(wd["ais"], len(wd["vessels"]), clock)
        P, S = E.positions(rx, len(wd["vessels"]))
        _cache.clear()
        _cache[key] = {"wd": wd, "P": P, "S": S, "cov": E.coverage(rx, clock), "dist": {}}
    return _cache[key]


def names(c):
    return {r["vessel_id"]: r for r in c.execute("SELECT v.vessel_id, v.kind, v.length_m, i.name, i.flag, i.owner, l.licensed FROM vessel v JOIN identity i USING (vessel_id) JOIN license l USING (vessel_id)")}


def truth_for(g):
    w = W.build(g["model_seed"], demo=True)
    return w, {v["idx"]: v for v in w["vessels"]}


@app.get("/v1/events/suspicious", tags=["events"], summary="The detectors' findings in a window: AIS silences scored by the gap model (with the six-hour rule's count beside them), SAR targets with no AIS and the silent boats that could have been them, carrier encounters and loitering carriers with a possible dark partner (with the distance-only rule's count beside them)")
def suspicious(hours: int = 48, top: int = 8, ctx: Ctx = Depends(auth("sea:read"))):
    with db.tx(ctx.tenant_id) as c:
        g = region(c)
        since = g["clock"] - datetime.timedelta(hours=hours)
        N = names(c)
        gaps = c.execute("SELECT * FROM behavior_event WHERE kind = 'ais_gap' AND coalesce(end_ts, %s) > %s ORDER BY score DESC", [g["clock"], since]).fetchall()
        dark = c.execute("SELECT * FROM behavior_event WHERE kind = 'dark_target' AND start_ts > %s ORDER BY start_ts", [since]).fetchall()
        rz = c.execute("SELECT * FROM rendezvous WHERE end_ts > %s ORDER BY kind, start_ts", [since]).fetchall()
        view = cached_view(c, ctx.tenant_id, g)
    wd = view["wd"]
    clock = slot_of(g["clock"])
    if hours not in view["dist"]:
        view["dist"][hours] = E.encounters(view["P"], view["S"], wd["vessels"], clock, rule="distance", since=max(0, clock - hours * 6))
    dist_rule = view["dist"][hours]
    w, tv = truth_for(g)
    deliberate = E.label_gaps([{"vessel": next(v["idx"] for v in wd["vessels"] if v["id"] == e["vessel_id"]), "s0": slot_of(e["start_ts"]), "s1": slot_of(e["end_ts"] or g["clock"])} for e in gaps], w["truth"])
    by_id = {v["id"]: v for v in wd["vessels"]}
    sar_truth = {str(did(ctx.tenant_id, k)): d["truth"] for k, d in enumerate(w["sar"])}
    ts_truth = [(x[0], x[1], x[2], x[3]) for x in w["truth"]["transship"] if not x[5]]
    gap_rows = [{"event_id": e["event_id"], "vessel_id": e["vessel_id"], "vessel": N[e["vessel_id"]]["name"], "kind": N[e["vessel_id"]]["kind"], "from": hhmm(slot_of(e["start_ts"])), "hours": e["evidence"]["hours"], "open": e["evidence"]["open"],
                 "score": round(e["score"], 3), "silence_probability_log10": e["evidence"]["silence_probability_log10"], "reserve_km": e["evidence"]["reserve_km"], "km_moved": e["evidence"]["km_moved"],
                 "simulation_truth": "deliberate" if d["deliberate"] else "not deliberate"} for e, d in zip(gaps, deliberate)]
    return jsonable_encoder({"clock": hhmm(clock), "window_hours": hours,
                             "gaps": {"total": len(gaps), "flagged_by_model": sum(1 for e in gaps if e["score"] > 0.5), "flagged_by_six_hour_rule": sum(1 for e in gaps if e["evidence"]["hours"] >= E.BASELINE_GAP_H),
                                      "deliberate_in_truth": sum(d["deliberate"] for d in deliberate), "top": gap_rows[:top]},
                             "dark_targets": [{"event_id": e["event_id"], "at": hhmm(slot_of(e["start_ts"])), "x_km": round(e["x_km"], 1), "y_km": round(e["y_km"], 1), "length_m": e["evidence"]["length_m"], "reserve_km": e["evidence"]["reserve_km"],
                                               "candidates": [N[uuid.UUID(x)]["name"] for x in e["evidence"]["candidates"]], "vessel_id": e["vessel_id"],
                                               "simulation_truth": (tv[sar_truth[e["evidence"]["detection_id"]]]["name"] if sar_truth.get(e["evidence"]["detection_id"]) is not None else "wave clutter")} for e in dark],
                             "rendezvous": [{"kind": r["kind"], "carrier": N[r["carrier_id"]]["name"], "vessel": N[r["vessel_id"]]["name"] if r["vessel_id"] else None, "from": hhmm(slot_of(r["start_ts"])), "hours": r["evidence"]["hours"], "declared": r["declared"],
                                             "partners": [N[uuid.UUID(x)]["name"] for x in r["evidence"].get("partners", [])], "sar_corroborated": r["evidence"].get("sar_corroborated"),
                                             "simulation_truth": next((tv[x[0]]["name"] for x in ts_truth if tv[x[1]]["name"] == N[r["carrier_id"]]["name"] and x[2] <= slot_of(r["end_ts"]) and x[3] >= slot_of(r["start_ts"])), "no transshipment")} for r in rz],
                             "distance_rule": {"what": "any two vessels within 1 km for an hour, outside port", "events": len(dist_rule), "pairs": sorted({f"{by_id_seq(wd, a)} & {by_id_seq(wd, b)}" for a, b, *_ in dist_rule})[:8]}})


def by_id_seq(wd, i):
    return wd["by_seq"][i]["name"]


@app.get("/v1/areas/{area_id}/activity", tags=["events"], summary="Inside and around the protected area: AIS presence by vessel, silences that began near it, SAR targets inside with and without AIS, and how well AIS is received there")
def activity(area_id: uuid.UUID, hours: int = 48, ctx: Ctx = Depends(auth("sea:read"))):
    with db.tx(ctx.tenant_id) as c:
        g = region(c)
        a = c.execute("SELECT * FROM protected_area WHERE area_id = %s", [area_id]).fetchone()
        if not a:
            raise Problem(404, "area_not_found")
        since = g["clock"] - datetime.timedelta(hours=hours)
        cx, cy, rr = float(a["centre_x_km"]), float(a["centre_y_km"]), float(a["radius_km"])
        inside = c.execute("""SELECT i.name, v.kind, count(*) AS fixes, count(*) FILTER (WHERE p.sog_kn < 5) AS slow FROM track_point p JOIN vessel v USING (vessel_id) JOIN identity i USING (vessel_id)
                               WHERE p.ts > %s AND (p.x_km - %s)^2 + (p.y_km - %s)^2 < %s^2 AND NOT ('possible_spoof' = ANY(p.flags)) GROUP BY 1, 2 ORDER BY 4 DESC, 3 DESC""", [since, cx, cy, rr]).fetchall()
        det = c.execute("SELECT count(*) AS n FROM satellite_detection WHERE ts > %s AND (x_km - %s)^2 + (y_km - %s)^2 < %s^2", [since, cx, cy, rr]).fetchone()["n"]
        dark_in = c.execute("SELECT count(*) AS n FROM behavior_event WHERE kind = 'dark_target' AND start_ts > %s AND (evidence->>'reserve_km')::float < 0", [since]).fetchone()["n"]
        near = c.execute("SELECT count(*) AS n, count(*) FILTER (WHERE score > 0.5) AS flagged FROM behavior_event WHERE kind = 'ais_gap' AND coalesce(end_ts, %s) > %s AND (evidence->>'reserve_km')::float < 20", [g["clock"], since]).fetchone()
        cov = cached_view(c, ctx.tenant_id, g)["cov"]
    ci = (int(cx // E.CELL), int(cy // E.CELL))
    fishing = [r for r in inside if r["kind"] == "fishing"]
    return jsonable_encoder({"area": a["name"], "window_hours": hours, "radius_km": rr, "vessels_seen_inside": len(inside), "fishing_boats_seen_inside": len(fishing),
                             "fishing_speed_fixes_inside": sum(r["slow"] for r in fishing), "transit_fixes_inside": sum(r["fixes"] - r["slow"] for r in inside),
                             "fixes_inside": [{"vessel": r["name"], "kind": r["kind"], "fixes": r["fixes"], "under_5_knots": r["slow"]} for r in inside[:12]],
                             "sar_detections_inside": det, "dark_targets_inside": dark_in, "gaps_starting_within_20km": near["n"], "of_which_flagged": near["flagged"],
                             "ais_reception": {"inside": round(float(cov[ci[0] - 1:ci[0] + 2, ci[1] - 1:ci[1] + 2].mean()), 2), "near_coast": round(float(cov[:3].mean()), 2), "east_of_300km": round(float(cov[15:].mean()), 2)}})


# ---------------------------------------------------------------- vessels, risk, cases

@app.get("/v1/vessels/{vessel_id}/timeline", tags=["vessels"], summary="One vessel: identity, licence, its fixes (every half hour), silences with their scores, SAR targets attributed to it, meetings, and its latest risk score")
def timeline(vessel_id: uuid.UUID, hours: int = 72, ctx: Ctx = Depends(auth("sea:read"))):
    with db.tx(ctx.tenant_id) as c:
        g = region(c)
        N = names(c)
        if vessel_id not in N:
            raise Problem(404, "vessel_not_found")
        since = g["clock"] - datetime.timedelta(hours=hours)
        lic = c.execute("SELECT * FROM license WHERE vessel_id = %s", [vessel_id]).fetchone()
        fixes = c.execute("SELECT ts, x_km, y_km, sog_kn, flags FROM track_point WHERE vessel_id = %s AND ts > %s ORDER BY ts", [vessel_id, since]).fetchall()
        gaps = c.execute("SELECT * FROM behavior_event WHERE vessel_id = %s AND kind = 'ais_gap' AND coalesce(end_ts, %s) > %s ORDER BY start_ts", [vessel_id, g["clock"], since]).fetchall()
        sightings = c.execute("SELECT * FROM behavior_event WHERE kind = 'dark_target' AND start_ts > %s AND evidence->'candidates' ? %s", [since, str(vessel_id)]).fetchall()
        meets = c.execute("SELECT * FROM rendezvous WHERE end_ts > %s AND (vessel_id = %s OR evidence->'partners' ? %s)", [since, vessel_id, str(vessel_id)]).fetchall()
        risk = c.execute("SELECT * FROM risk_score WHERE vessel_id = %s ORDER BY scored_at DESC LIMIT 1", [vessel_id]).fetchone()
        carriers = {r["carrier_id"] for r in meets}
        carrier_fixes = {N[k]["name"]: [[round(r["x_km"], 1), round(r["y_km"], 1)] for r in c.execute("SELECT x_km, y_km FROM track_point WHERE vessel_id = %s AND ts > %s ORDER BY ts", [k, since])][::2] for k in carriers}
    v = N[vessel_id]
    return jsonable_encoder({"vessel_id": vessel_id, "name": v["name"], "kind": v["kind"], "length_m": float(v["length_m"]), "flag": v["flag"], "owner": v["owner"], "licensed": lic["licensed"], "prior_violations": lic["prior_violations"],
                             "fixes": [{"at": hhmm(slot_of(f["ts"])), "slot": slot_of(f["ts"]), "x": round(f["x_km"], 1), "y": round(f["y_km"], 1), "sog": round(f["sog_kn"], 1)} for f in fixes if not f["flags"]],
                             "gaps": [{"from": hhmm(slot_of(e["start_ts"])), "to": hhmm(slot_of(e["end_ts"])) if e["end_ts"] else "still silent", "hours": e["evidence"]["hours"], "score": round(e["score"], 3),
                                       "start": [round(e["x_km"], 1), round(e["y_km"], 1)], "end": e["evidence"]["end"], "features": e["evidence"]["features"]} for e in gaps],
                             "sightings": [{"at": hhmm(slot_of(e["start_ts"])), "x": round(e["x_km"], 1), "y": round(e["y_km"], 1), "length_m": e["evidence"]["length_m"], "top_candidate": e["vessel_id"] == vessel_id, "candidates": len(e["evidence"]["candidates"]), "reserve_km": e["evidence"]["reserve_km"]} for e in sightings],
                             "meetings": [{"kind": r["kind"], "carrier": N[r["carrier_id"]]["name"], "from": hhmm(slot_of(r["start_ts"])), "hours": r["evidence"]["hours"], "x": round(r["x_km"], 1), "y": round(r["y_km"], 1), "declared": r["declared"], "top_partner": r["vessel_id"] == vessel_id, "sar_corroborated": r["evidence"].get("sar_corroborated")} for r in meets],
                             "carrier_tracks": carrier_fixes, "risk": {"score": round(risk["score"], 3), "rank": risk["rank"], "contributions": risk["contributions"]} if risk else None,
                             "reserve": W.MPA})


class ScoreIn(BaseModel):
    top: int = Field(10, ge=1, le=110)


@app.post("/v1/risk/score", status_code=201, tags=["risk"], summary="Score every fishing vessel over the fortnight: a logistic model over its worst silence, dark sightings near the reserve, undeclared carrier encounters, being a loitering carrier's likely dark partner, and the registry; each factor's contribution is returned, with a ranking by silent hours beside it")
def score(body: ScoreIn, ctx: Ctx = Depends(auth("risk:score")), idem: str | None = IdemKey):
    def work(c):
        g = region(c)
        m = E.models()
        wd = world_from_db(c, ctx.tenant_id)
        A = E.analyse(wd, slot_of(g["clock"]), m["gap"])
        R = E.score_risk(wd, A, m)
        now = g["clock"]
        db.load(c, "risk_score", ["vessel_id", "tenant_id", "scored_at", "score", "rank", "contributions", "model_version"],
                [(wd["by_seq"][r["vessel"]]["id"], ctx.tenant_id, now, r["score"], k + 1, Jsonb({f: round(x, 2) for f, x in r["contributions"].items()}), E.VERSION) for k, r in enumerate(R)])
        w, tv = truth_for(g)
        ill = set(w["truth"]["illicit"])
        by_hours = sorted(R, key=lambda r: -r["gap_hours"])
        rows = lambda rs: [{"vessel_id": wd["by_seq"][r["vessel"]]["id"], "vessel": wd["by_seq"][r["vessel"]]["name"], "score": round(r["score"], 3), "silent_hours": round(r["gap_hours"], 1),     # noqa: E731
                            "contributions": {f: round(x, 2) for f, x in sorted(r["contributions"].items(), key=lambda kv: -abs(kv[1]))}, "features": {f: round(float(x), 3) for f, x in r["features"].items()},
                            "simulation_truth": "broke the rules" if r["vessel"] in ill else "lawful"} for r in rs[:body.top]]
        audit.record(c, ctx, "risk.scored", "region", g["id"], {"vessels": len(R), "top": wd["by_seq"][R[0]["vessel"]]["name"]})
        k = body.top
        return 201, {"scored_at": hhmm(slot_of(now)), "vessels": len(R), "model": E.VERSION, "ranking": rows(R), "by_silent_hours": rows(by_hours),
                     "simulation_truth": {"rule_breakers": len(ill), f"found_in_top_{k}": sum(1 for r in R[:k] if r["vessel"] in ill), f"found_in_top_{k}_by_silent_hours": sum(1 for r in by_hours[:k] if r["vessel"] in ill)}}
    return run(ctx, idem, body, work)


class CaseIn(BaseModel):
    vessel_id: uuid.UUID
    summary: str = Field(min_length=10, max_length=2000)


@app.post("/v1/cases", status_code=201, tags=["cases"], summary="Open an investigation case: the vessel's evidence (silences, sightings, meetings, risk contributions, registry) frozen into the case with its SHA-256")
def open_case(body: CaseIn, ctx: Ctx = Depends(auth("cases:open")), idem: str | None = IdemKey):
    def work(c):
        region(c)
        if not c.execute("SELECT 1 FROM vessel WHERE vessel_id = %s", [body.vessel_id]).fetchone():
            raise Problem(404, "vessel_not_found")
        ev = {"gaps": c.execute("SELECT event_id, start_ts, end_ts, score, evidence FROM behavior_event WHERE vessel_id = %s AND kind = 'ais_gap' AND score > 0.2 ORDER BY score DESC LIMIT 5", [body.vessel_id]).fetchall(),
              "sightings": c.execute("SELECT event_id, start_ts, x_km, y_km, evidence FROM behavior_event WHERE kind = 'dark_target' AND evidence->'candidates' ? %s", [str(body.vessel_id)]).fetchall(),
              "meetings": c.execute("SELECT event_id, kind, carrier_id, start_ts, end_ts, evidence FROM rendezvous WHERE vessel_id = %s OR evidence->'partners' ? %s", [body.vessel_id, str(body.vessel_id)]).fetchall(),
              "risk": c.execute("SELECT score, rank, contributions, model_version, scored_at FROM risk_score WHERE vessel_id = %s ORDER BY scored_at DESC LIMIT 1", [body.vessel_id]).fetchone(),
              "registry": c.execute("SELECT i.name, i.flag, i.owner, l.licensed, l.prior_violations FROM identity i JOIN license l USING (vessel_id) WHERE vessel_id = %s", [body.vessel_id]).fetchone()}
        ev = jsonable_encoder(ev)
        digest = hashlib.sha256(json.dumps(ev, sort_keys=True).encode()).hexdigest()
        cid = uuid.uuid7()
        c.execute("INSERT INTO cases (case_id, tenant_id, vessel_id, status, summary, evidence, evidence_sha256, opened_by) VALUES (%s,%s,%s,'open',%s,%s,%s,%s)", [cid, ctx.tenant_id, body.vessel_id, body.summary, Jsonb(ev), digest, ctx.actor_id])
        audit.record(c, ctx, "case.opened", "case", cid, {"vessel": ev["registry"]["name"], "evidence_sha256": digest})
        return 201, {"case_id": cid, "status": "open", "vessel": ev["registry"]["name"], "evidence_sha256": digest, "evidence_counts": {k: len(v) for k, v in ev.items() if isinstance(v, list)}, "risk": ev["risk"]}
    return run(ctx, idem, body, work)


class EscalateIn(BaseModel):
    action: str = Field(pattern="^(request_inspection|notify_flag_state|refer_to_port_state)$")
    note: str = Field(min_length=3, max_length=1000)


@app.post("/v1/cases/{case_id}/escalate", tags=["cases"], summary="(+) Escalate a case to an action outside the centre (an at-sea inspection, the flag state, the port state); needs a supervisor who did not open it")
def escalate(case_id: uuid.UUID, body: EscalateIn, ctx: Ctx = Depends(auth("cases:open")), idem: str | None = IdemKey):
    def work(c):
        k = c.execute("SELECT * FROM cases WHERE case_id = %s", [case_id]).fetchone()
        if not k:
            raise Problem(404, "case_not_found")
        if k["opened_by"] == ctx.actor_id:
            raise Problem(409, "four_eyes", "the analyst who opened the case cannot escalate it")
        if "cases:escalate" not in PERMISSIONS[ctx.role]:
            raise Problem(403, "forbidden", "escalation needs a supervisor")
        if k["status"] != "open":
            raise Problem(409, "case_not_open")
        c.execute("UPDATE cases SET status = 'escalated', escalated_by = %s, escalated_at = now(), action = %s WHERE case_id = %s", [ctx.actor_id, body.action, case_id])
        audit.record(c, ctx, "case.escalated", "case", case_id, {"action": body.action, "note": body.note})
        return 200, {"case_id": case_id, "status": "escalated", "action": body.action, "escalated_by": ctx.actor_name, "evidence_sha256": k["evidence_sha256"]}
    return run(ctx, idem, body, work)
