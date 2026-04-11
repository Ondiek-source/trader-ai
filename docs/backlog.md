# BACKLOG: Dashboard — "Scripts & Diagnostics" Tab

Priority: Medium
Description:
  Add a new tab to the FastAPI dashboard with:

- Pre-built log filter buttons (signals, health, errors, training)
- One-click copy-to-clipboard for common az commands
- Live filtered log stream (websocket, not grep)
- Resource usage chart (memory/CPU over time from health events)
- Table view with columns: timestamp, event, pair, detail
- Filter by pair, event type, time range
- Auto-scroll toggle

Technical:

- WebSocket proxy to az container logs (or read from blob log handler)
- Store health snapshots in a rolling in-memory list for chart
- Use existing BlobLogHandler as data source (already persists logs to blob)

change confidence_threshold back to .65 afterwards
