"""Integration, end-to-end and security tests against a real Postgres."""
import psycopg
import pytest

from core import db, jobs, scenario

from .conftest import bearer

SUP, ANALYST, VIEWER, OTHER = bearer("coast", "supervisor"), bearer("coast", "analyst"), bearer("coast", "viewer"), bearer("other", "supervisor")


def test_nothing_works_before_a_region(client):
    assert client.get("/v1/events/suspicious", headers=VIEWER).json()["code"] == "no_region"
    assert client.post("/v1/region:load", headers=ANALYST, json={}).status_code == 403


@pytest.fixture(scope="module")
def demo(client):
    results, _ = scenario.run(client, drain=jobs.drain)
    return results


def one(sql, params=(), tenant=None):
    with db.tx(tenant) as c:
        return c.execute(sql, params).fetchone()


def test_demo_journey(demo):
    load = demo["load"]["result"]
    assert load["vessels"] == {"fishing": 110, "carrier": 4, "cargo": 8} and one("SELECT count(*) AS n FROM license")["n"] == 122
    assert load["coverage"]["near_coast"] > load["coverage"]["east_of_300km"]
    assert demo["area0"]["fishing_speed_fixes_inside"] < 30 and demo["area"]["fishing_speed_fixes_inside"] == 0
    assert demo["poll"]["result"]["clock"].endswith("00:00") and demo["ingest"]["accepted"] == 2 and len(demo["ingest"]["flagged"]) == 1 and len(demo["ingest"]["rejected"]) == 3
    assert one("SELECT count(*) AS n FROM track_point WHERE 'possible_spoof' = ANY(flags)")["n"] == 1
    s = demo["susp"]
    flagged = [g for g in s["gaps"]["top"] if g["score"] > 0.5]
    assert "Southern Star 7" in [g["vessel"] for g in flagged] and s["gaps"]["flagged_by_model"] < s["gaps"]["flagged_by_six_hour_rule"] / 5
    assert next(g for g in flagged if g["vessel"] == "Southern Star 7")["simulation_truth"] == "deliberate"
    inside = [d for d in s["dark_targets"] if d["reserve_km"] < 0]
    assert any(d["simulation_truth"] == "Southern Star 7" and "Southern Star 7" in d["candidates"] for d in inside)
    assert any(d["simulation_truth"] == "Southern Star 7" and d["candidates"] == ["Southern Star 7"] for d in s["dark_targets"])
    pc = [r for r in s["rendezvous"] if r["carrier"] == "Polar Crown" and r["kind"] == "dark_transship_candidate"]
    assert pc and pc[0]["partners"][0] == "Southern Star 7" and pc[0]["sar_corroborated"] and pc[0]["simulation_truth"] == "Southern Star 7"
    assert "Bay Sister 1 & Bay Sister 2" in s["distance_rule"]["pairs"] and not any(r["vessel"] in ("Bay Sister 1", "Bay Sister 2") for r in s["rendezvous"])
    r = demo["risk"]
    assert r["simulation_truth"]["found_in_top_10"] > r["simulation_truth"]["found_in_top_10_by_silent_hours"] and one("SELECT count(*) AS n FROM risk_score")["n"] == 110
    t = demo["timeline"]
    assert t["name"] == "Southern Star 7" and t["risk"]["rank"] <= 10 and t["gaps"][0]["hours"] > 20 and any(m["carrier"] == "Polar Crown" for m in t["meetings"])
    assert demo["case"]["evidence_counts"]["gaps"] >= 1 and demo["self"]["code"] == "four_eyes" and demo["escalate"]["status"] == "escalated"
    assert one("SELECT status, action FROM cases")["action"] == "request_inspection"
    assert demo["audit"]["chain_valid"] and {"region.loaded", "tracks.polled", "tracks.ingested", "risk.scored", "case.opened", "case.escalated"} <= {e["action"] for e in demo["audit"]["events"]}


def test_validation_and_replay(client, demo):
    assert client.post("/v1/tracks:poll", headers=ANALYST, json={"hours": 1}).json()["code"] == "end_of_data"
    vid = demo["timeline"]["vessel_id"]
    assert client.get("/v1/vessels/00000000-0000-0000-0000-000000000000/timeline", headers=VIEWER).status_code == 404
    assert client.post("/v1/cases", headers=ANALYST, json={"vessel_id": vid, "summary": "short"}).status_code == 422
    h = {**ANALYST, "Idempotency-Key": "case-replay"}
    body = {"vessel_id": vid, "summary": "second look at the same evidence"}
    a, b = client.post("/v1/cases", headers=h, json=body), client.post("/v1/cases", headers=h, json=body)
    assert a.json() == b.json() and b.headers.get("Idempotent-Replay") == "true"
    assert client.post(f"/v1/cases/{a.json()['case_id']}/escalate", headers=VIEWER, json={"action": "request_inspection", "note": "x x"}).status_code == 403
    assert client.post(f"/v1/cases/{demo['case']['case_id']}/escalate", headers=SUP, json={"action": "request_inspection", "note": "again"}).json()["code"] == "case_not_open"


def test_tenant_isolation_and_append_only_audit(client, demo):
    vid = demo["timeline"]["vessel_id"]
    assert client.get(f"/v1/vessels/{vid}/timeline", headers=OTHER).json()["code"] == "no_region"
    with db.tx() as c:
        other = c.execute("SELECT DISTINCT tenant_id FROM api_tokens WHERE actor_name LIKE '%%Northern Isles%%'").fetchone()["tenant_id"]
    assert one("SELECT count(*) AS n FROM track_point", tenant=other)["n"] == 0 and one("SELECT count(*) AS n FROM cases", tenant=other)["n"] == 0
    with pytest.raises(psycopg.errors.RaiseException), db.tx() as c:
        c.execute("UPDATE audit_events SET action = 'edited'")
