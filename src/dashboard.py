"""
dashboard.py — Lightweight HTTP status dashboard for Trader AI.

Exposes two endpoints on port 8080:
  GET /         → HTML dashboard (auto-refreshes every 10s)
  GET /status   → JSON status payload (polled by the dashboard)
  GET /health   → Simple health check (used by Docker HEALTHCHECK)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any

logger = logging.getLogger(__name__)

# ── HTML Dashboard ─────────────────────────────────────────────────────────────

DASHBOARD_HTML: str = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Trader AI — Live Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0d1117; color: #c9d1d9; min-height: 100vh; padding: 24px;
    }
    h1 { color: #58a6ff; font-size: 1.4rem; margin-bottom: 4px; }
    .subtitle { color: #8b949e; font-size: 0.85rem; margin-bottom: 24px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
    .card {
      background: #161b22; border: 1px solid #30363d; border-radius: 8px;
      padding: 16px; display: flex; flex-direction: column; gap: 6px;
    }
    .card-label { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .card-value { font-size: 1.6rem; font-weight: 700; color: #c9d1d9; }
    .card-value.green { color: #3fb950; }
    .card-value.red { color: #f85149; }
    .card-value.yellow { color: #d29922; }
    .card-value.blue { color: #58a6ff; }
    .card-value.gray { color: #8b949e; }
    .section-title { color: #8b949e; font-size: 0.8rem; text-transform: uppercase;
      letter-spacing: 0.08em; margin-bottom: 10px; margin-top: 8px; }
    .bar-wrap { background: #21262d; border-radius: 4px; height: 8px; overflow: hidden; margin-top: 4px; }
    .bar { height: 100%; border-radius: 4px; background: #58a6ff; transition: width 0.5s; }
    .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
      margin-right: 6px; vertical-align: middle; }
    .dot-green { background: #3fb950; box-shadow: 0 0 6px #3fb950; }
    .dot-red { background: #f85149; }
    .dot-yellow { background: #d29922; animation: pulse 1.5s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
    .log { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
      padding: 16px; font-family: monospace; font-size: 0.78rem; color: #8b949e;
      max-height: 180px; overflow-y: auto; }
    .log-entry { padding: 2px 0; border-bottom: 1px solid #21262d; }
    .log-entry.win { color: #3fb950; }
    .log-entry.loss { color: #f85149; }
    .log-entry.draw { color: #d29922; }
    .log-entry.signal { color: #58a6ff; }
    .log-entry.info { color: #8b949e; }
    .footer { text-align: center; color: #484f58; font-size: 0.75rem; margin-top: 24px; }
    #last-update { color: #484f58; font-size: 0.75rem; }
  </style>
</head>
<body>
  <h1>Trader AI &mdash; Live Dashboard</h1>
  <div class="subtitle">Auto-refreshes every 10 seconds &nbsp;|&nbsp; <span id="last-update">Loading...</span></div>

  <div class="section-title">Session</div>
  <div class="grid">
    <div class="card">
      <div class="card-label">Status</div>
      <div class="card-value" id="status-text">—</div>
    </div>
    <div class="card">
      <div class="card-label">Wins Today</div>
      <div class="card-value green" id="wins">—</div>
    </div>
    <div class="card">
      <div class="card-label">Losses Today</div>
      <div class="card-value red" id="losses">—</div>
    </div>
    <div class="card">
      <div class="card-label">Draws Today</div>
      <div class="card-value gray" id="draws">—</div>
    </div>
    <div class="card">
      <div class="card-label">Net Profit</div>
      <div class="card-value" id="profit">—</div>
    </div>
    <div class="card">
      <div class="card-label">Signals Fired</div>
      <div class="card-value blue" id="signals">—</div>
    </div>
    <div class="card">
      <div class="card-label">Elapsed</div>
      <div class="card-value" id="elapsed">—</div>
    </div>
  </div>

  <div class="section-title">Engine</div>
  <div class="grid">
    <div class="card">
      <div class="card-label">Confidence Threshold</div>
      <div class="card-value yellow" id="threshold">—</div>
      <div class="bar-wrap"><div class="bar" id="threshold-bar" style="width:0%"></div></div>
    </div>
    <div class="card">
      <div class="card-label">Martingale Streak</div>
      <div class="card-value" id="streak">—</div>
    </div>
    <div class="card">
      <div class="card-label">Pending Signals</div>
      <div class="card-value" id="pending">—</div>
    </div>
    <div class="card">
      <div class="card-label">Practice Mode</div>
      <div class="card-value" id="practice">—</div>
    </div>
  </div>

  <div class="section-title">Connections</div>
  <div class="grid">
    <div class="card">
      <div class="card-label">Price Stream</div>
      <div class="card-value" id="stream-status">—</div>
    </div>
    <div class="card">
      <div class="card-label">Quotex Account</div>
      <div class="card-value" id="quotex-status">—</div>
    </div>
    <div class="card">
      <div class="card-label">Quotex Balance</div>
      <div class="card-value green" id="quotex-balance">—</div>
    </div>
    <div class="card">
      <div class="card-label">Ticks Received</div>
      <div class="card-value" id="ticks">—</div>
    </div>
  </div>

  <div class="section-title">Recent Activity</div>
  <div class="log" id="activity-log">
    <div class="log-entry info">Waiting for data...</div>
  </div>

  <div class="footer">Trader AI &copy; 2024 &mdash; All signals use key: Ondiek</div>

<script>
  const activity = [];

  function classifyEvent(text) {
    const lower = (text || '').toLowerCase();
    if (lower.includes('signal') || lower.includes('fired')) return 'signal';
    if (lower.includes('win')) return 'win';
    if (lower.includes('loss')) return 'loss';
    if (lower.includes('draw')) return 'draw';
    return 'info';
  }

  function renderLog() {
    const el = document.getElementById('activity-log');
    if (activity.length === 0) {
      el.innerHTML = '<div class="log-entry info">Waiting for data...</div>';
      return;
    }
    el.innerHTML = activity.map(e =>
      '<div class="log-entry ' + e.cls + '">[' + e.time + '] ' + e.text + '</div>'
    ).join('');
  }

  async function refresh() {
    try {
      const r = await fetch('/status');
      const d = await r.json();

      // Session
      const session = d.session || {};
      const active = session.is_active;
      const stopped = d.stopped;
      document.getElementById('status-text').innerHTML =
        stopped ? '<span class="status-dot dot-red"></span>STOPPED' :
        active  ? '<span class="status-dot dot-green"></span>ACTIVE' :
                  '<span class="status-dot dot-yellow"></span>IDLE';

      document.getElementById('wins').textContent = session.wins ?? '—';
      document.getElementById('losses').textContent = session.losses ?? '—';
      document.getElementById('draws').textContent = session.draws ?? 0;

      const profit = session.net_profit ?? 0;
      const profitEl = document.getElementById('profit');
      profitEl.textContent = '$' + profit.toFixed(2);
      profitEl.className = 'card-value ' + (profit >= 0 ? 'green' : 'red');

      document.getElementById('signals').textContent = session.signals_fired ?? '—';
      document.getElementById('elapsed').textContent =
        session.elapsed_minutes ? Math.round(session.elapsed_minutes) + ' min' : '—';

      // Engine
      const threshold = d.confidence_threshold ?? 0;
      document.getElementById('threshold').textContent = (threshold * 100).toFixed(0) + '%';
      document.getElementById('threshold-bar').style.width = (threshold * 100) + '%';

      const streak = d.martingale_streak ?? 0;
      const streakEl = document.getElementById('streak');
      streakEl.textContent = streak;
      streakEl.className = 'card-value ' + (streak === 0 ? 'green' : streak >= 3 ? 'red' : 'yellow');

      document.getElementById('pending').textContent = d.pending_signals ?? 0;
      document.getElementById('practice').innerHTML =
        d.practice_mode
          ? '<span style="color:#d29922">PRACTICE</span>'
          : '<span style="color:#f85149">LIVE</span>';

      // Connections
      const stream = d.stream || {};
      document.getElementById('stream-status').innerHTML =
        stream.connected
          ? '<span class="status-dot dot-green"></span>Connected'
          : '<span class="status-dot dot-red"></span>Disconnected';
      document.getElementById('ticks').textContent =
        stream.ticks_received ? stream.ticks_received.toLocaleString() : '—';

      const quotex = d.quotex || {};
      document.getElementById('quotex-status').innerHTML =
        quotex.connected
          ? '<span class="status-dot dot-green"></span>Connected'
          : '<span class="status-dot dot-yellow"></span>Not connected';
      document.getElementById('quotex-balance').textContent =
        quotex.balance ? '$' + quotex.balance.toFixed(2) : '—';

      // Activity log — append every unique event
      const currentEvent = d.last_event || '';
      if (currentEvent) {
        const lastEntry = activity.length > 0 ? activity[0] : null;
        const now = new Date().toLocaleTimeString();
        const cls = classifyEvent(currentEvent);
        if (!lastEntry || lastEntry.text !== currentEvent) {
          activity.unshift({ time: now, text: currentEvent, cls });
          if (activity.length > 20) activity.pop();
        }
      }
      renderLog();

      const now = new Date().toLocaleTimeString();
      document.getElementById('last-update').textContent =
        'Last updated: ' + now;

    } catch(e) {
      document.getElementById('last-update').textContent = 'Connection error — retrying...';
    }
  }

  refresh();
  setInterval(refresh, 10000);
</script>
</body>
</html>"""


# ── Status store (updated by main.py) ─────────────────────────────────────────


class StatusStore:
    """
    Thread-safe store for live system status.

    Updated by the main trading loop via :meth:`update`.  Read by the
    HTTP handler on every ``/status`` request.

    The orchestrator **must** call ``status_store.update({"last_event": ...})``
    after every significant event for the activity log to populate.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {
            "stopped": False,
            "martingale_streak": 0,
            "confidence_threshold": 0.65,
            "pending_signals": 0,
            "practice_mode": True,
            "session": {},
            "stream": {"connected": False, "ticks_received": 0},
            "quotex": {"connected": False, "balance": 0.0},
            "last_event": "",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    def update(self, data: dict[str, Any]) -> None:
        """
        Merge *data* into the current status.

        Args:
            data: Partial status dict.  Keys not present are left unchanged.
        """
        self._data.update(data)

    def get(self) -> dict[str, Any]:
        """
        Return a snapshot of the current status with a ``server_time`` field.

        Returns:
            Copy of the status dict safe for JSON serialisation.
        """
        return {**self._data, "server_time": datetime.now(timezone.utc).isoformat()}


# Singleton — imported by main.py
status_store: StatusStore = StatusStore()


# ── HTTP handler ───────────────────────────────────────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    """Serves the dashboard HTML, JSON status, and health check."""

    def do_GET(self) -> None:  # noqa: N802
        """
        Route GET requests to the appropriate handler.

        ``/`` and ``/index.html`` return the dashboard HTML.
        ``/status`` returns the live JSON status.
        ``/health`` returns a plain-text ``"ok"``.
        """
        if self.path in ("/", "/index.html"):
            self._respond(200, "text/html", DASHBOARD_HTML.encode())
        elif self.path == "/status":
            payload: bytes = json.dumps(status_store.get()).encode()
            self._respond(200, "application/json", payload)
        elif self.path == "/health":
            self._respond(200, "text/plain", b"ok")
        else:
            self._respond(404, "text/plain", b"not found")

    def _respond(self, code: int, content_type: str, body: bytes) -> None:
        """
        Send an HTTP response with CORS headers.

        Args:
            code: HTTP status code.
            content_type: ``Content-Type`` header value.
            body: Response body bytes.
        """
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Suppress default HTTP server log output to stderr."""
        pass


# ── Async runner ───────────────────────────────────────────────────────────────


async def run_dashboard(port: int = 8080) -> None:
    """
    Run the HTTP dashboard in a daemon background thread.

    The thread is daemonised so it dies automatically when the process
    exits.  The coroutine sleeps forever to keep the task alive.

    Args:
        port: TCP port to bind (default 8080).
    """
    server: HTTPServer = HTTPServer(("0.0.0.0", port), _Handler)
    thread: Thread = Thread(target=server.serve_forever, daemon=True, name="dashboard")
    thread.start()
    logger.info(
        {
            "event": "dashboard_started",
            "port": port,
            "url": f"http://0.0.0.0:{port}",
        }
    )
    # Keep coroutine alive — daemon thread handles the actual serving
    while True:
        await asyncio.sleep(60)
