-- Domain model. Names follow the blueprint's schema (its `case` table is `cases`: CASE is reserved); additions are marked (+).
-- Positions are x/y kilometres in the synthetic region (x east from the coast), not PostGIS.

CREATE TABLE region (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants UNIQUE, name text NOT NULL, model_seed int NOT NULL, clock timestamptz NOT NULL, created_at timestamptz NOT NULL DEFAULT now());   -- (+) the monitored region and its clock
SELECT enable_tenant_rls('region');

CREATE TABLE vessel (vessel_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, mmsi bigint NOT NULL, kind text NOT NULL CHECK (kind IN ('fishing', 'carrier', 'cargo')), length_m numeric NOT NULL, gear text NOT NULL, seq int NOT NULL, UNIQUE (tenant_id, mmsi));   -- (+) seq
SELECT enable_tenant_rls('vessel');

CREATE TABLE identity (vessel_id uuid NOT NULL REFERENCES vessel, tenant_id uuid NOT NULL REFERENCES tenants, name text NOT NULL, flag text NOT NULL, owner text NOT NULL, valid_from timestamptz NOT NULL, PRIMARY KEY (vessel_id, valid_from));
SELECT enable_tenant_rls('identity');

CREATE TABLE license (vessel_id uuid PRIMARY KEY REFERENCES vessel, tenant_id uuid NOT NULL REFERENCES tenants, zone text NOT NULL, licensed boolean NOT NULL, prior_violations int NOT NULL DEFAULT 0);   -- (+) prior violations on record
SELECT enable_tenant_rls('license');

CREATE TABLE transship_authorisation (vessel_id uuid NOT NULL REFERENCES vessel, carrier_id uuid NOT NULL REFERENCES vessel, day date NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (vessel_id, carrier_id, day));   -- (+) declared transshipments with an observer
SELECT enable_tenant_rls('transship_authorisation');

CREATE TABLE protected_area (area_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, name text NOT NULL, centre_x_km numeric NOT NULL, centre_y_km numeric NOT NULL, radius_km numeric NOT NULL);
SELECT enable_tenant_rls('protected_area');

CREATE TABLE track_point (vessel_id uuid NOT NULL REFERENCES vessel, ts timestamptz NOT NULL, x_km double precision NOT NULL, y_km double precision NOT NULL, sog_kn real NOT NULL, flags text[] NOT NULL DEFAULT '{}', tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (vessel_id, ts));
SELECT enable_tenant_rls('track_point');
CREATE INDEX track_point_ts ON track_point (tenant_id, ts);

CREATE TABLE satellite_detection (detection_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, ts timestamptz NOT NULL, x_km double precision NOT NULL, y_km double precision NOT NULL, length_m real NOT NULL, sensor text NOT NULL DEFAULT 'SAR');
SELECT enable_tenant_rls('satellite_detection');

CREATE TABLE behavior_event (event_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, vessel_id uuid REFERENCES vessel, kind text NOT NULL CHECK (kind IN ('ais_gap', 'dark_target')), start_ts timestamptz NOT NULL, end_ts timestamptz, x_km double precision NOT NULL, y_km double precision NOT NULL, score double precision NOT NULL, evidence jsonb NOT NULL, model_version text NOT NULL);
SELECT enable_tenant_rls('behavior_event');
CREATE INDEX behavior_event_score ON behavior_event (tenant_id, kind, score DESC);

CREATE TABLE rendezvous (event_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, carrier_id uuid NOT NULL REFERENCES vessel, vessel_id uuid REFERENCES vessel, kind text NOT NULL CHECK (kind IN ('encounter', 'dark_transship_candidate')), start_ts timestamptz NOT NULL, end_ts timestamptz NOT NULL, x_km double precision NOT NULL, y_km double precision NOT NULL, declared boolean NOT NULL DEFAULT false, evidence jsonb NOT NULL);
SELECT enable_tenant_rls('rendezvous');

CREATE TABLE risk_score (vessel_id uuid NOT NULL REFERENCES vessel, tenant_id uuid NOT NULL REFERENCES tenants, scored_at timestamptz NOT NULL, score double precision NOT NULL, rank int NOT NULL, contributions jsonb NOT NULL, model_version text NOT NULL, PRIMARY KEY (vessel_id, scored_at));
SELECT enable_tenant_rls('risk_score');

CREATE TABLE cases (case_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, vessel_id uuid NOT NULL REFERENCES vessel, status text NOT NULL CHECK (status IN ('open', 'escalated', 'closed')), summary text NOT NULL, evidence jsonb NOT NULL, evidence_sha256 text NOT NULL, opened_by uuid NOT NULL, escalated_by uuid, escalated_at timestamptz, action text, created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('cases');
