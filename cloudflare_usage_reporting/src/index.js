const EVENT_FIELDS = [
  "event_id",
  "app_instance_id",
  "event_kind",
  "recordings_delta",
  "seconds_delta",
  "occurred_at",
  "app_version",
];
const UUID_V4_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const ISO_8601_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})$/;
const MAX_APP_VERSION_LENGTH = 64;

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function authorized(request, env) {
  return env.ADMIN_TOKEN && request.headers.get("authorization") === `Bearer ${env.ADMIN_TOKEN}`;
}

function validEvent(event) {
  if (!event || typeof event !== "object" || !EVENT_FIELDS.every((key) => key in event)) {
    return false;
  }
  return (
    typeof event.event_id === "string" &&
    UUID_V4_RE.test(event.event_id) &&
    typeof event.app_instance_id === "string" &&
    UUID_V4_RE.test(event.app_instance_id) &&
    ["enrollment", "recording"].includes(event.event_kind) &&
    Number.isInteger(event.recordings_delta) &&
    event.recordings_delta >= 0 &&
    Number.isInteger(event.seconds_delta) &&
    event.seconds_delta >= 0 &&
    typeof event.occurred_at === "string" &&
    ISO_8601_RE.test(event.occurred_at) &&
    !Number.isNaN(Date.parse(event.occurred_at)) &&
    typeof event.app_version === "string" &&
    event.app_version.length <= MAX_APP_VERSION_LENGTH
  );
}

function dateRange(url) {
  const from = url.searchParams.get("from") || "1970-01-01T00:00:00+00:00";
  const to = url.searchParams.get("to") || "9999-12-31T23:59:59+00:00";
  return { from, to };
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/healthz") {
      return json({ status: "ok" });
    }

    if (request.method === "POST" && url.pathname === "/v1/usage-events") {
      const routeLimit = await env.USAGE_EVENT_ROUTE_LIMITER.limit({
        key: "v1/usage-events",
      });
      if (!routeLimit.success) {
        return json({ error: "usage reporting is temporarily rate limited" }, 429);
      }

      let event;
      try {
        event = await request.json();
      } catch {
        return json({ error: "invalid JSON" }, 400);
      }
      if (!validEvent(event)) {
        return json({ error: "invalid usage event" }, 400);
      }

      const appLimit = await env.USAGE_EVENT_APP_LIMITER.limit({
        key: event.app_instance_id,
      });
      if (!appLimit.success) {
        return json({ error: "usage reporting is temporarily rate limited" }, 429);
      }

      await env.DB.prepare(
        `INSERT OR IGNORE INTO usage_events
          (event_id, app_instance_id, event_kind, recordings_delta, seconds_delta, occurred_at, app_version)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
      )
        .bind(
          event.event_id,
          event.app_instance_id,
          event.event_kind,
          event.recordings_delta,
          event.seconds_delta,
          event.occurred_at,
          event.app_version,
        )
        .run();
      return json({ accepted: true }, 201);
    }

    if (request.method === "GET" && url.pathname === "/admin/usage-summary") {
      if (!authorized(request, env)) {
        return json({ error: "unauthorized" }, 401);
      }
      const { from, to } = dateRange(url);
      const group = url.searchParams.get("group") || "day";
      const bucket = group === "week" ? "strftime('%Y-%W', received_at)" : "substr(received_at, 1, 10)";
      if (!new Set(["day", "week"]).has(group)) {
        return json({ error: "group must be day or week" }, 400);
      }

      const result = await env.DB.prepare(
        `SELECT ${bucket} AS period,
                SUM(recordings_delta) AS recordings_scored,
                SUM(seconds_delta) AS seconds_scored,
                ROUND(SUM(seconds_delta) / 3600.0, 1) AS hours_scored,
                COUNT(DISTINCT app_instance_id) AS reporting_app_copies
           FROM usage_events
          WHERE datetime(received_at) >= datetime(?) AND datetime(received_at) < datetime(?)
          GROUP BY period
          ORDER BY period`,
      )
        .bind(from, to)
        .all();
      return json({ from, to, group, rows: result.results });
    }

    return json({ error: "not found" }, 404);
  },
};
