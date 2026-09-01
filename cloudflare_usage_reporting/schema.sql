CREATE TABLE IF NOT EXISTS usage_events (
  event_id TEXT PRIMARY KEY,
  app_instance_id TEXT NOT NULL,
  event_kind TEXT NOT NULL CHECK (event_kind IN ('enrollment', 'recording')),
  recordings_delta INTEGER NOT NULL CHECK (recordings_delta >= 0),
  seconds_delta INTEGER NOT NULL CHECK (seconds_delta >= 0),
  occurred_at TEXT NOT NULL,
  app_version TEXT NOT NULL,
  received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

DROP INDEX IF EXISTS usage_events_occurred_at;

CREATE INDEX IF NOT EXISTS usage_events_received_at
  ON usage_events (received_at);
