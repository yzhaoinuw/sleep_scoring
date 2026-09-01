import assert from "node:assert/strict";
import test from "node:test";

import worker from "../src/index.js";

const VALID_EVENT = {
  event_id: "66b2d854-7e10-4e9b-8a49-fc1c9fc184c1",
  app_instance_id: "b2c73834-8c48-4eb4-a5bd-5b78e5ec79f4",
  event_kind: "recording",
  recordings_delta: 1,
  seconds_delta: 3600,
  occurred_at: "2026-08-31T16:00:00+00:00",
  app_version: "v0.17.1",
};

function makeEnv() {
  const inserts = [];
  const queryLog = [];
  return {
    inserts,
    queryLog,
    ADMIN_TOKEN: "test-token",
    USAGE_EVENT_ROUTE_LIMITER: { limit: async () => ({ success: true }) },
    USAGE_EVENT_APP_LIMITER: { limit: async () => ({ success: true }) },
    DB: {
      prepare(query) {
        queryLog.push(query);
        return {
          bind(...values) {
            return {
              async run() {
                inserts.push(values);
              },
              async all() {
                return { results: [] };
              },
            };
          },
        };
      },
    },
  };
}

test("accepts a UUID-backed usage event", async () => {
  const env = makeEnv();
  const response = await worker.fetch(
    new Request("https://usage.example/v1/usage-events", {
      method: "POST",
      body: JSON.stringify(VALID_EVENT),
    }),
    env,
  );

  assert.equal(response.status, 201);
  assert.equal(env.inserts.length, 1);
});

test("rejects malformed event identifiers before writing", async () => {
  const env = makeEnv();
  const response = await worker.fetch(
    new Request("https://usage.example/v1/usage-events", {
      method: "POST",
      body: JSON.stringify({ ...VALID_EVENT, app_instance_id: "not-a-uuid" }),
    }),
    env,
  );

  assert.equal(response.status, 400);
  assert.equal(env.inserts.length, 0);
});

test("groups administrator summaries by Worker receipt time", async () => {
  const env = makeEnv();
  const response = await worker.fetch(
    new Request("https://usage.example/admin/usage-summary?group=week", {
      headers: { authorization: "Bearer test-token" },
    }),
    env,
  );

  assert.equal(response.status, 200);
  assert.match(env.queryLog[0], /strftime\('%Y-%W', received_at\)/);
  assert.match(env.queryLog[0], /datetime\(received_at\)/);
});
