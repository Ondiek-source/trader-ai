"""
core/dashboard.py — Lightweight HTTP status dashboard for Trader AI.

Exposes three endpoints:
    GET /         → HTML dashboard (auto-refreshes every 10s)
    GET /status   → JSON status payload combining live state + journal data
    GET /health   → Structured health check for Docker / load balancers

Data sources:
    StatusStore  — push-based live state from pipeline.py / live.py
    Journal      — pull-based trade history and session stats on /status
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

from core.config import get_settings

logger = logging.getLogger(__name__)

# ── HTML Dashboard ─────────────────────────────────────────────────────────────

DASHBOARD_HTML: str = """\
<!DOCTYPE html>
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
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px; margin-bottom: 24px;
    }
    .card {
      background: #161b22; border: 1px solid #30363d; border-radius: 8px;
      padding: 16px; display: flex; flex-direction: column; gap: 6px;
    }
    .card-label {
      color: #8b949e; font-size: 0.75rem; text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .card-value { font-size: 1.6rem; font-weight: 700; color: #c9d1d9; }
    .card-value.green { color: #3fb950; }
    .card-value.red { color: #f85149; }
    .card-value.yellow { color: #d29922; }
    .card-value.blue { color: #58a6ff; }
    .card-value.gray { color: #8b949e; }
    .section-title {
      color: #8b949e; font-size: 0.8rem; text-transform: uppercase;
      letter-spacing: 0.08em; margin-bottom: 10px; margin-top: 8px;
    }
    .bar-wrap {
      background: #21262d; border-radius: 4px; height: 8px;
      overflow: hidden; margin-top: 4px;
    }
    .bar { height: 100%; border-radius: 4px; background: #58a6ff; transition: width 0.5s; }
    .bar.red { background: #f85149; }
    .bar.yellow { background: #d29922; }
    .status-dot {
      display: inline-block; width: 8px; height: 8px; border-radius: 50%;
      margin-right: 6px; vertical-align: middle;
    }
    .dot-green { background: #3fb950; box-shadow: 0 0 6px #3fb950; }
    .dot-red { background: #f85149; box-shadow: 0 0 6px #f85149; }
    .dot-yellow { background: #d29922; animation: pulse 1.5s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
    .log {
      background: #161b22; border: 1px solid #30363d; border-radius: 8px;
      padding: 16px; font-family: monospace; font-size: 0.78rem;
      color: #8b949e; max-height: 240px; overflow-y: auto;
    }
    .log-entry { padding: 3px 0; border-bottom: 1px solid #21262d; }
    .log-entry.win { color: #3fb950; }
    .log-entry.loss { color: #f85149; }
    .log-entry.draw { color: #d29922; }
    .log-entry.signal { color: #58a6ff; }
    .log-entry.kill { color: #f85149; font-weight: bold; }
    .log-entry.info { color: #8b949e; }
    .table-wrap {
      background: #161b22; border: 1px solid #30363d; border-radius: 8px;
      padding: 16px; overflow-x: auto;
    }
    table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
    th { color: #8b949e; text-align: left; padding: 6px 10px;
         border-bottom: 1px solid #30363d; font-weight: 600; }
    td { padding: 6px 10px; border-bottom: 1px solid #21262d; }
    .footer { text-align: center; color: #484f58; font-size: 0.75rem; margin-top: 24px; }
    #last-update { color: #484f58; font-size: 0.75rem; }
    .kill-banner {
      background: #f8514922; border: 1px solid #f85149; border-radius: 8px;
      padding: 16px; margin-bottom: 16px; color: #f85149; font-weight: bold;
      text-align: center; display: none;
    }
  </style>
</head>
<body>
  <h1>Trader AI &mdash; Live Dashboard</h1>
  <div class="subtitle">
    Auto-refreshes every 10s &nbsp;|&nbsp;
    <span id="last-update">Loading...</span>
  </div>

  <div class="kill-banner" id="kill-banner">
    KILL SWITCH ACTIVE &mdash; Max martingale streak reached. Trading halted.
  </div>

  <div class="section-title">Session</div>
  <div class="grid">
    <div class="card">
      <div class="card-label">Status</div>
      <div class="card-value" id="status-text">&mdash;</div>
    </div>
    <div class="card">
      <div class="card-label">Wins</div>
      <div class="card-value green" id="wins">&mdash;</div>
    </div>
    <div class="card">
      <div class="card-label">Losses</div>
      <div class="card-value red" id="losses">&mdash;</div>
    </div>
    <div class="card">
      <div class="card-label">Win Rate</div>
      <div class="card-value" id="winrate">&mdash;</div>
    </div>
    <div class="card">
      <div class="card-label">Net Profit</div>
      <div class="card-value" id="profit">&mdash;</div>
    </div>
    <div class="card">
      <div class="card-label">Signals Fired</div>
      <div class="card-value blue" id="signals">&mdash;</div>
    </div>
    <div class="card">
      <div class="card-label">Elapsed</div>
      <div class="card-value" id="elapsed">&mdash;</div>
    </div>
  </div>

  <div class="section-title">Engine</div>
  <div class="grid">
    <div class="card">
      <div class="card-label">Base Threshold</div>
      <div class="card-value gray" id="base-threshold">&mdash;</div>
    </div>
    <div class="card">
      <div class="card-label">Effective Threshold</div>
      <div class="card-value yellow" id="threshold">&mdash;</div>
      <div class="bar-wrap"><div class="bar" id="threshold-bar" style="width:0%"></div></div>
    </div>
    <div class="card">
      <div class="card-label">Martingale Streak</div>
      <div class="card-value" id="streak">&mdash;</div>
    </div>
    <div class="card">
      <div class="card-label">Max Streak</div>
      <div class="card-value gray" id="max-streak">&mdash;</div>
    </div>
    <div class="card">
      <div class="card-label">Pending Signals</div>
      <div class="card-value" id="pending">&mdash;</div>
    </div>
    <div class="card">
      <div class="card-label">Mode</div>
      <div class="card-value" id="practice">&mdash;</div>
    </div>
  </div>

  <div class="section-title">Connections</div>
  <div class="grid">
    <div class="card">
      <div class="card-label">Price Stream</div>
      <div class="card-value" id="stream-status">&mdash;</div>
    </div>
    <div class="card">
      <div class="card-label">Quotex Account</div>
      <div class="card-value" id="quotex-status">&mdash;</div>
    </div>
    <div class="card">
      <div class="card-label">Quotex Balance</div>
      <div class="card-value green" id="quotex-balance">&mdash;</div>
    </div>
    <div class="card">
      <div class="card-label">Ticks Received</div>
      <div class="card-value" id="ticks">&mdash;</div>
    </div>
  </div>

  <div class="section-title">Recent Trades</div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr><th>Time</th><th>Symbol</th><th>Side</th><th>Result</th><th>P&amp;L</th></tr>
      </thead>
      <tbody id="trades-body">
        <tr><td colspan="5" style="color:#484f58">No trades yet</td></tr>
      </tbody>
    </table>
  </div>

  <div class="section-title" style="margin-top:16px">Activity Log</div>
  <div class="log" id="activity-log">
    <div class="log-entry info">Waiting for data...</div>
  </div>

  <div class="footer">Trader AI Dashboard</div>

<script>
  const activity = [];

  function classifyEvent(text) {
    const lower = (text || '').toLowerCase();
    if (lower.includes('kill') || lower.includes('halted')) return 'kill';
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
    el.innerHTML = activity.map(function(e) {
      return '<div class="log-entry ' + e.cls + '">[' + e.time + '] ' +
             e.text.replace(/</g, '&lt;') + '</div>';
    }).join('');
  }

  function renderTrades(trades) {
    var tbody = document.getElementById('trades-body');
    if (!trades || trades.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="color:#484f58">No trades yet</td></tr>';
      return;
    }
    tbody.innerHTML = trades.slice(0, 10).map(function(t) {
      var resultCls = t.result === 'win' ? 'green' : t.result === 'loss' ? 'red' : 'yellow';
      var pnl = t.pnl != null ? '$' + Number(t.pnl).toFixed(2) : '&mdash;';
      var time = t.time || '&mdash;';
      return '<tr>' +
        '<td>' + time + '</td>' +
        '<td>' + (t.symbol || '&mdash;') + '</td>' +
        '<td>' + (t.side || '&mdash;') + '</td>' +
        '<td style="color:' + resultCls + '">' + (t.result || '?') + '</td>' +
        '<td style="color:' + resultCls + '">' + pnl + '</td>' +
        '</tr>';
    }).join('');
  }

  async function refresh() {
    try {
      var r = await fetch('/status');
      var d = await r.json();

      // Kill switch
      var killActive = d.kill_switch_active || false;
      document.getElementById('kill-banner').style.display = killActive ? 'block' : 'none';

      // Session
      var session = d.session || {};
      var active = session.is_active;
      var stopped = d.stopped;
      document.getElementById('status-text').innerHTML =
        killActive ? '<span class="status-dot dot-red"></span>KILLED' :
        stopped ? '<span class="status-dot dot-red"></span>STOPPED' :
        active  ? '<span class="status-dot dot-green"></span>ACTIVE' :
                  '<span class="status-dot dot-yellow"></span>IDLE';

      document.getElementById('wins').textContent = session.wins ?? '—';
      document.getElementById('losses').textContent = session.losses ?? '—';

      var wins = session.wins || 0;
      var losses = session.losses || 0;
      var total = wins + losses;
      var winrate = total > 0 ? (wins / total * 100).toFixed(1) + '%' : '—';
      var wrEl = document.getElementById('winrate');
      wrEl.textContent = winrate;
      wrEl.className = 'card-value ' + (total > 0 ? (wins / total >= 0.5 ? 'green' : 'red') : '');

      var profit = session.net_profit ?? 0;
      var profitEl = document.getElementById('profit');
      profitEl.textContent = '$' + profit.toFixed(2);
      profitEl.className = 'card-value ' + (profit >= 0 ? 'green' : 'red');

      document.getElementById('signals').textContent = session.signals_fired ?? '—';
      document.getElementById('elapsed').textContent =
        session.elapsed_minutes ? Math.round(session.elapsed_minutes) + ' min' : '—';

      // Engine
      var baseT = d.base_confidence_threshold ?? 0;
      document.getElementById('base-threshold').textContent = (baseT * 100).toFixed(0) + '%';

      var effT = d.confidence_threshold ?? 0;
      document.getElementById('threshold').textContent = (effT * 100).toFixed(0) + '%';
      var barEl = document.getElementById('threshold-bar');
      barEl.style.width = (effT * 100) + '%';
      barEl.className = 'bar' + (effT > 0.85 ? ' red' : effT > 0.75 ? ' yellow' : '');

      var streak = d.martingale_streak ?? 0;
      var maxStreak = d.martingale_max_streak ?? 3;
      var streakEl = document.getElementById('streak');
      streakEl.textContent = streak + ' / ' + maxStreak;
      streakEl.className = 'card-value ' +
        (streak === 0 ? 'green' : streak >= maxStreak ? 'red' : 'yellow');

      document.getElementById('max-streak').textContent = maxStreak;
      document.getElementById('pending').textContent = d.pending_signals ?? 0;
      document.getElementById('practice').innerHTML =
        d.practice_mode
          ? '<span style="color:#d29922">PRACTICE</span>'
          : '<span style="color:#f85149">LIVE</span>';

      // Connections
      var stream = d.stream || {};
      document.getElementById('stream-status').innerHTML =
        stream.connected
          ? '<span class="status-dot dot-green"></span>Connected'
          : '<span class="status-dot dot-red"></span>Disconnected';
      document.getElementById('ticks').textContent =
        stream.ticks_received ? stream.ticks_received.toLocaleString() : '—';

      var quotex = d.quotex || {};
      document.getElementById('quotex-status').innerHTML =
        quotex.connected
          ? '<span class="status-dot dot-green"></span>Connected'
          : '<span class="status-dot dot-yellow"></span>Not connected';
      document.getElementById('quotex-balance').textContent =
        quotex.balance != null ? '$' + Number(quotex.balance).toFixed(2) : '—';

      // Recent trades
      renderTrades(d.recent_trades);

      // Activity log
      var currentEvent = d.last_event || '';
      if (currentEvent) {
        var lastEntry = activity.length > 0 ? activity[0] : null;
        var now = new Date().toLocaleTimeString();
        var cls = classifyEvent(currentEvent);
        if (!lastEntry || lastEntry.text !== currentEvent) {
          activity.unshift({ time: now, text: currentEvent, cls: cls });
          if (activity.length > 50) activity.pop();
        }
      }
      renderLog();

      document.getElementById('last-update').textContent =
        'Last updated: ' + new Date().toLocaleTimeString();

    } catch(e) {
      document.getElementById('last-update').textContent = 'Connection error — retrying...';
    }
  }

  refresh();
  setInterval(refresh, 10000);
</script>
</body>
</html>
"""


# ── Status Store ───────────────────────────────────────────────────────────────


class StatusStore:
    """
    Thread-safe store for live system status.

    Updated by pipeline.py / live.py via :meth:`update`.
    Read by the HTTP handler on every ``/status`` request.

    A threading.Lock guards all reads and writes because the HTTP
    server runs in a daemon thread while updates arrive from the
    asyncio event loop thread.
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._data: dict[str, Any] = {
            # Session state
            "stopped": False,
            "kill_switch_active": False,
            "session": {},
            # Engine state
            "base_confidence_threshold": None,
            "confidence_threshold": None,
            "martingale_streak": 0,
            "martingale_max_streak": 3,
            "pending_signals": 0,
            "practice_mode": True,
            # Connections
            "stream": {"connected": False, "ticks_received": 0},
            "quotex": {"connected": False, "balance": 0.0},
            # Activity
            "last_event": "",
            "recent_trades": [],
            # Metadata
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    def update(self, data: dict[str, Any]) -> None:
        """Merge *data* into the current status."""
        with self._lock:
            self._data.update(data)

    def get(self) -> dict[str, Any]:
        """Return a deep-copy snapshot with server timestamp."""
        with self._lock:
            snapshot = copy.deepcopy(self._data)
            snapshot["server_time"] = datetime.now(timezone.utc).isoformat()
            return snapshot

    @property
    def _lock_obj(self) -> threading.Lock:
        """Expose the lock for direct status_store._lock access (health_task)."""
        return self._lock


# Singleton — imported by pipeline.py
status_store: StatusStore = StatusStore()


# ── Journal reader ─────────────────────────────────────────────────────────────


def _load_recent_trades(journal_dir: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Read the most recent trade entries from the journal Parquet file.

    Returns an empty list if the journal doesn't exist yet or is unreadable.
    The dashboard calls this on every /status request so the trade table
    stays current without push-based updates from live.py.

    Args:
        journal_dir: Path to the journal data directory.
        limit: Maximum number of recent trades to return.

    Returns:
        List of dicts with keys: time, symbol, side, result, pnl.
    """
    try:
        import pandas as pd

        trades_path = Path(journal_dir) / "trades.parquet"
        if not trades_path.exists():
            return []

        df = pd.read_parquet(trades_path)
        if df.empty:
            return []

        df = df.tail(limit).iloc[::-1]  # most recent first

        trades: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            result_val = float(row.get("result", 0))
            trades.append({
                "time": str(row.get("timestamp", ""))[:19],
                "symbol": str(row.get("symbol", "")),
                "side": str(row.get("side", "")),
                "result": "win" if result_val > 0 else "loss" if result_val < 0 else "draw",
                "pnl": result_val,
            })
        return trades

    except Exception:
        return []


# ── HTTP Handler ───────────────────────────────────────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    """Serves the dashboard HTML, JSON status, and health check."""

    # Class-level config — set once before first request.
    _journal_dir: str = ""

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._respond(200, "text/html", DASHBOARD_HTML.encode())
        elif self.path == "/status":
            snapshot = status_store.get()
            # Enrich with journal trade history
            snapshot["recent_trades"] = _load_recent_trades(
                self._journal_dir, limit=10
            )
            payload: bytes = json.dumps(snapshot, default=str).encode()
            self._respond(200, "application/json", payload)
        elif self.path == "/health":
            health = {
                "status": "ok",
                "kill_switch": status_store.get().get("kill_switch_active", False),
                "server_time": datetime.now(timezone.utc).isoformat(),
            }
            self._respond(200, "application/json", json.dumps(health).encode())
        else:
            self._respond(404, "text/plain", b"not found")

    def _respond(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Suppress default HTTP server log output."""
        pass


# ── Async runner ───────────────────────────────────────────────────────────────


async def run_dashboard(port: int = 8080) -> None:
    """
    Run the HTTP dashboard in a daemon background thread.

    Configures the journal directory from settings so the /status
    endpoint can pull recent trades from the journal Parquet file.

    Args:
        port: TCP port to bind (default 8080).
    """
    settings = get_settings()

    # Resolve journal directory — journal.py writes trades.parquet
    # alongside storage data. Use the same data_dir root.
    journal_dir: str = str(
        Path(settings.data_dir).resolve()
    )
    _Handler._journal_dir = journal_dir

    server: HTTPServer = HTTPServer(("0.0.0.0", port), _Handler)
    thread: Thread = Thread(
        target=server.serve_forever, daemon=True, name="dashboard"
    )
    thread.start()

    logger.info(
        {
            "event": "dashboard_started",
            "port": port,
            "url": f"http://0.0.0.0:{port}",
            "journal_dir": journal_dir,
        }
    )

    # Keep coroutine alive — daemon thread handles serving
    while True:
        await asyncio.sleep(60)
