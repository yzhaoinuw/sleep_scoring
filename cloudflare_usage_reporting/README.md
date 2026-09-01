# Sleep Scoring Usage Reporting Service

This Cloudflare Worker receives the app's explicitly opted-in aggregate usage
events. It stores no recording names, paths, signal data, annotations, animal
identifiers, or local deduplication fingerprints.

## Deploy

1. Create a Cloudflare D1 database named `sleep-scoring-usage` and copy its ID
   into `wrangler.toml`.
2. Run `npx wrangler d1 execute sleep-scoring-usage --remote --file=schema.sql`.
3. Set an administrator secret: `npx wrangler secret put ADMIN_TOKEN`.
4. Deploy with `npx wrangler deploy`.
5. Set `USAGE_REPORT_URL` in `app_src/config.py` to the deployed
   `/v1/usage-events` URL. Users opt in only by setting
   `ENABLE_USAGE_REPORTING = True` in that same file, then restarting the app.
   The opt-in flag is preserved by compatible source-only updates; keep the
   endpoint itself release-managed.

Run the Worker checks with `node --test` from this directory.

The public write route is protected by two Cloudflare Worker rate-limit
bindings: 600 requests per minute for the route and 20 requests per minute for
each opaque app-copy ID. The limits are local to each Cloudflare location and
eventually consistent, so they reduce abuse rather than provide accounting.
The app deliberately does not embed a shared secret: anything distributed with
the app can be extracted.

The Worker deliberately has no public read endpoint. To inspect usage, call
the administrator endpoint with the secret token:

```text
GET /admin/usage-summary?from=2026-09-01T00:00:00+00:00&to=2026-10-01T00:00:00+00:00&group=day
Authorization: Bearer <ADMIN_TOKEN>
```

Use `group=week` for weekly totals. Summary periods use the Worker receipt
time, not a client clock. Events are idempotent by `event_id`, so an app may
retry a failed upload safely. An enrollment event reports the local total
already present when a user first opts in; it contributes to lifetime totals
but does not reconstruct historical completion dates.

`GET /healthz` is the only unauthenticated read route. It returns
`{"status":"ok"}` and is safe for deployment monitoring.

## Live deployment

The current service endpoint is
`https://sleep-scoring-usage-reporting.brainflowzzz.workers.dev/v1/usage-events`.
Its administrator summaries remain protected by the `ADMIN_TOKEN` secret.
