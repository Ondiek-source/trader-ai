"""
core/dashboard.py — Lightweight HTTP status dashboard for Trader AI.

Exposes three endpoints:
    GET /         → HTML dashboard (auto-refreshes every 10s)
    GET /status   → JSON status payload combining live state + journal data
    GET /health   → Structured health check for Docker / load balancers

Data sources:
    StatusStore  — push-based; recent_trades pushed by Reporter
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import threading
from datetime import datetime, timezone
from core.config import local_now
from http.server import BaseHTTPRequestHandler, HTTPServer
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
<title>Trader AI</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Sora:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0e0f11;
    --surface: #161719;
    --surface2: #1d1f22;
    --border: rgba(255,255,255,0.07);
    --border2: rgba(255,255,255,0.12);
    --text: #f0f0ee;
    --muted: #6b6d73;
    --accent: #c8f135;
    --accent2: #5af0b4;
    --red: #ff5a5a;
    --amber: #f0a840;
    --blue: #5aa8ff;
    --font-sans: 'Sora', sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    font-size: 13px;
    min-height: 100vh;
    padding: 24px;
    line-height: 1.5;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
  }
  .logo { display: flex; align-items: center; gap: 10px; }
  .logo-mark {
    width: 28px; height: 28px;
    background: var(--accent);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
  }
  .logo-mark svg { width: 14px; height: 14px; }
  .logo-text { font-weight: 600; font-size: 15px; letter-spacing: -0.02em; color: var(--text); }
  .logo-text span { color: var(--accent); }
  .header-right { display: flex; align-items: center; gap: 12px; }
  .badge-version {
    font-family: var(--font-mono); font-size: 11px; color: var(--muted);
    background: var(--surface2); border: 1px solid var(--border);
    padding: 3px 8px; border-radius: 4px; letter-spacing: 0.02em;
  }
  .badge-live {
    font-family: var(--font-mono); font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--accent); display: flex; align-items: center; gap: 5px;
  }
  .live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent); animation: blink 1.8s ease-in-out infinite;
  }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

  .kill-banner {
    display: none; align-items: center; gap: 10px;
    background: rgba(255,90,90,0.08); border: 1px solid rgba(255,90,90,0.3);
    border-radius: 8px; padding: 12px 16px; margin-bottom: 20px;
    font-family: var(--font-mono); font-size: 11px; color: var(--red); letter-spacing: 0.03em;
  }
  .kill-banner.active { display: flex; }

  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 12px; }
  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 12px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
  @media (max-width: 900px) {
    .grid-4 { grid-template-columns: repeat(2, 1fr); }
    .grid-3 { grid-template-columns: repeat(2, 1fr); }
  }
  @media (max-width: 540px) {
    .grid-4, .grid-3, .grid-2 { grid-template-columns: 1fr; }
  }

  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px;
    transition: border-color 0.2s; position: relative; overflow: hidden;
  }
  .card:hover { border-color: var(--border2); }
  .card-label {
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--muted); margin-bottom: 10px; font-weight: 400;
  }
  .card-value {
    font-size: 28px; font-weight: 300; letter-spacing: -0.03em;
    line-height: 1; margin-bottom: 6px;
  }
  .card-value.mono { font-family: var(--font-mono); font-size: 22px; font-weight: 400; }
  .card-sub { font-size: 11px; color: var(--muted); font-family: var(--font-mono); }
  .green { color: var(--accent2); }
  .red { color: var(--red); }
  .amber { color: var(--amber); }
  .blue { color: var(--blue); }
  .lime { color: var(--accent); }

  .bar-track { height: 2px; background: var(--border); border-radius: 2px; margin-top: 12px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 2px; transition: width 0.6s ease; }
  .bar-fill.green { background: var(--accent2); }
  .bar-fill.amber { background: var(--amber); }

  .status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: var(--font-mono); font-size: 11px;
    letter-spacing: 0.06em; text-transform: uppercase;
    padding: 4px 10px; border-radius: 20px;
  }
  .status-pill.active { background: rgba(90,240,180,0.1); color: var(--accent2); border: 1px solid rgba(90,240,180,0.2); }
  .status-pill.idle { background: rgba(240,168,64,0.1); color: var(--amber); border: 1px solid rgba(240,168,64,0.2); }
  .status-pill.killed { background: rgba(255,90,90,0.1); color: var(--red); border: 1px solid rgba(255,90,90,0.2); }
  .status-dot-sm { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }

  .section-header {
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em;
    color: var(--muted); margin: 20px 0 10px; font-weight: 400;
    display: flex; align-items: center; gap: 8px;
  }
  .section-header::after { content: ''; flex: 1; height: 1px; background: var(--border); }

  .engine-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px;
  }
  .engine-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 14px; }
  .engine-metric-val { font-family: var(--font-mono); font-size: 18px; font-weight: 500; line-height: 1; margin-bottom: 4px; }
  .engine-metric-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }

  .conn-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  .conn-item { background: var(--surface2); border-radius: 8px; padding: 12px 10px; text-align: center; }
  .conn-status { font-family: var(--font-mono); font-size: 11px; margin-bottom: 4px; font-weight: 500; }
  .conn-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }

  .table-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
  table { width: 100%; border-collapse: collapse; }
  thead th {
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--muted); font-weight: 400; padding: 12px 16px;
    text-align: left; background: var(--surface2); border-bottom: 1px solid var(--border);
  }
  tbody td { padding: 10px 16px; border-bottom: 1px solid var(--border); font-size: 12px; font-family: var(--font-mono); }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover { background: var(--surface2); }
  .result-win { color: var(--accent2); font-weight: 500; }
  .result-loss { color: var(--red); font-weight: 500; }
  .result-draw { color: var(--amber); font-weight: 500; }
  .side-call { color: var(--blue); }
  .side-put { color: var(--amber); }

  .log-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
  .log-inner {
    font-family: var(--font-mono); font-size: 11px;
    max-height: 240px; overflow-y: auto; padding: 4px 0;
  }
  .log-inner::-webkit-scrollbar { width: 4px; }
  .log-inner::-webkit-scrollbar-track { background: transparent; }
  .log-inner::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
  .log-row {
    display: flex; gap: 14px; padding: 7px 16px;
    border-bottom: 1px solid var(--border); align-items: baseline;
    transition: background 0.1s;
  }
  .log-row:last-child { border-bottom: none; }
  .log-row:hover { background: var(--surface2); }
  .log-ts { color: var(--muted); min-width: 72px; font-size: 10px; }
  .log-msg { flex: 1; line-height: 1.4; }
  .log-row.win .log-msg { color: var(--accent2); }
  .log-row.loss .log-msg { color: var(--red); }
  .log-row.signal .log-msg { color: var(--blue); }
  .log-row.kill .log-msg { color: var(--red); font-weight: 500; }
  .log-row.info .log-msg { color: #888; }

  .footer {
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--border);
    font-size: 11px; color: var(--muted); font-family: var(--font-mono);
  }
</style>
</head>
<body>

<div class="header">
  <div class="logo">
    <div class="logo-mark">
      <svg viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M2 10L7 4L12 10" stroke="#0e0f11" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M4 10L7 7L10 10" stroke="#0e0f11" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    <span class="logo-text">Trader<span>AI</span></span>
  </div>
  <div class="header-right">
    <div class="badge-live"><span class="live-dot"></span>Live</div>
    <div class="badge-version" id="version">v3.1.0</div>
  </div>
</div>

<div class="kill-banner" id="kill-banner">
  <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M7 1L13 12H1L7 1Z" stroke="currentColor" stroke-width="1.2"/><path d="M7 5.5V8" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/><circle cx="7" cy="10" r="0.6" fill="currentColor"/></svg>
  KILL SWITCH ACTIVE — Max martingale streak reached. Trading halted.
</div>

<div class="grid-4">
  <div class="card">
    <div class="card-label">Status</div>
    <div id="status-pill" style="margin-bottom:8px"><span class="status-pill idle"><span class="status-dot-sm"></span>Idle</span></div>
    <div class="card-sub" id="session-time">—</div>
  </div>
  <div class="card">
    <div class="card-label">Net P&amp;L</div>
    <div class="card-value mono" id="profit" style="color:var(--accent)">—</div>
    <div class="card-sub">today</div>
  </div>
  <div class="card">
    <div class="card-label">Win Rate</div>
    <div class="card-value" id="winrate">—</div>
    <div class="bar-track"><div class="bar-fill green" id="winrate-bar" style="width:0%"></div></div>
  </div>
  <div class="card">
    <div class="card-label">Trades Executed</div>
    <div class="card-value" id="signals">—</div>
    <div class="card-sub" id="signal-rate">—</div>
  </div>
</div>

<div class="grid-4">
  <div class="card">
    <div class="card-label">Wins</div>
    <div class="card-value green" id="wins">—</div>
    <div class="card-sub">closed</div>
  </div>
  <div class="card">
    <div class="card-label">Losses</div>
    <div class="card-value red" id="losses">—</div>
    <div class="card-sub">closed</div>
  </div>
  <div class="card">
    <div class="card-label">Draws</div>
    <div class="card-value amber" id="draws">—</div>
    <div class="card-sub">break-even</div>
  </div>
  <div class="card">
    <div class="card-label">Session</div>
    <div class="card-value mono blue" id="elapsed">—</div>
    <div class="card-sub">elapsed</div>
  </div>
</div>

<div class="grid-2">
  <div class="engine-card">
    <div class="card-label" style="margin-bottom:14px">Engine config</div>
    <div class="engine-row">
      <div class="engine-metric"><div class="engine-metric-val" id="base-threshold">—</div><div class="engine-metric-label">Base</div></div>
      <div class="engine-metric"><div class="engine-metric-val amber" id="threshold">—</div><div class="engine-metric-label">Effective</div></div>
      <div class="engine-metric"><div class="engine-metric-val" id="streak">—</div><div class="engine-metric-label">Streak</div></div>
      <div class="engine-metric"><div class="engine-metric-val blue" id="pending">—</div><div class="engine-metric-label">Pending</div></div>
    </div>
    <div class="bar-track"><div class="bar-fill amber" id="threshold-bar" style="width:0%"></div></div>
    <div style="margin-top:10px;font-size:11px;color:var(--muted);font-family:var(--font-mono)">Martingale max: <span id="max-streak" style="color:var(--text)">—</span> &nbsp;·&nbsp; Mode: <span id="practice-mode" style="color:var(--amber)">—</span></div>
  </div>

  <div class="engine-card">
    <div class="card-label" style="margin-bottom:14px">Connections</div>
    <div class="conn-grid">
      <div class="conn-item"><div class="conn-status" id="stream-status">—</div><div class="conn-label">Stream</div></div>
      <div class="conn-item"><div class="conn-status" id="quotex-status">—</div><div class="conn-label">Quotex</div></div>
      <div class="conn-item"><div class="conn-status green" id="quotex-balance">—</div><div class="conn-label">Balance</div></div>
      <div class="conn-item"><div class="conn-status blue" id="ticks">—</div><div class="conn-label">Ticks</div></div>
    </div>
  </div>
</div>

<div class="section-header">Recent Trades</div>
<div class="table-card">
  <table>
    <thead>
      <tr><th>Time</th><th>Symbol</th><th>Side</th><th>Result</th><th>P&amp;L</th><th>Duration</th></tr>
    </thead>
    <tbody id="trades-body">
      <tr><td colspan="6" style="color:var(--muted);text-align:center;padding:20px;font-size:11px">No trades yet</td></tr>
    </tbody>
  </table>
</div>

<div class="section-header">Activity Log</div>
<div class="log-card">
  <div class="log-inner" id="activity-log">
    <div class="log-row info"><span class="log-ts">—</span><span class="log-msg">Waiting for data…</span></div>
  </div>
</div>

<div class="footer">
  <span id="ts">—</span>
</div>

<script>
  const activity = [];
  let lastEventHash = '';

  function esc(t) {
    return (t||'').replace(/[&<>]/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]||m));
  }

  function evtType(t) {
    const l = (t||'').toLowerCase();
    if (l.includes('kill')||l.includes('halted')) return 'kill';
    if (l.includes('signal')||l.includes('fired')) return 'signal';
    if (l.includes('win')) return 'win';
    if (l.includes('loss')) return 'loss';
    return 'info';
  }

  function renderLog() {
    const el = document.getElementById('activity-log');
    if (!activity.length) {
      el.innerHTML = '<div class="log-row info"><span class="log-ts">—</span><span class="log-msg">Waiting for data…</span></div>';
      return;
    }
    el.innerHTML = activity.slice(0,40).map(e =>
      `<div class="log-row ${e.type}"><span class="log-ts">${e.time}</span><span class="log-msg">${esc(e.text)}</span></div>`
    ).join('');
  }

  function renderTrades(trades) {
    const tb = document.getElementById('trades-body');
    if (!trades || !trades.length) {
      tb.innerHTML = '<tr><td colspan="6" style="color:var(--muted);text-align:center;padding:20px;font-size:11px">No trades yet</td></tr>';
      return;
    }
    tb.innerHTML = trades.slice(0,10).map(t => {
      const rc = t.result === 'win' ? 'result-win' : t.result === 'loss' ? 'result-loss' : 'result-draw';
      const sc = (t.side||'').toLowerCase() === 'call' ? 'side-call' : 'side-put';
      const pnl = t.pnl != null ? (t.pnl >= 0 ? '+$' : '-$') + Math.abs(t.pnl).toFixed(2) : '—';
      const pnlColor = t.pnl >= 0 ? 'var(--accent2)' : 'var(--red)';
      return `<tr>
        <td style="color:var(--muted)">${t.time||'—'}</td>
        <td>${t.symbol||'—'}</td>
        <td class="${sc}">${(t.side||'—').toUpperCase()}</td>
        <td class="${rc}">${(t.result||'?').toUpperCase()}</td>
        <td style="color:${pnlColor}">${pnl}</td>
        <td style="color:var(--muted)">${t.duration||'—'}</td>
      </tr>`;
    }).join('');
  }

  async function refresh() {
    try {
      const r = await fetch('/status');
      const d = await r.json();

      const killActive = d.kill_switch_active || false;
      document.getElementById('kill-banner').classList.toggle('active', killActive);

      const session = d.session || {};
      const active = session.is_active;
      const stopped = d.stopped;
      let pillClass, pillLabel;
      if (killActive) { pillClass = 'killed'; pillLabel = 'Killed'; }
      else if (stopped) { pillClass = 'killed'; pillLabel = 'Stopped'; }
      else if (active) { pillClass = 'active'; pillLabel = 'Active'; }
      else { pillClass = 'idle'; pillLabel = 'Idle'; }
      document.getElementById('status-pill').innerHTML =
        `<span class="status-pill ${pillClass}"><span class="status-dot-sm"></span>${pillLabel}</span>`;

      if (d.started_at) {
        document.getElementById('session-time').textContent = 'Started ' + new Date(d.started_at).toLocaleTimeString();
      }

      const wins = session.wins || 0;
      const losses = session.losses || 0;
      const draws = session.draws || 0;
      const total = wins + losses;
      const wr = total > 0 ? (wins / total * 100).toFixed(1) : '0.0';

      document.getElementById('wins').textContent = wins;
      document.getElementById('losses').textContent = losses;
      document.getElementById('draws').textContent = draws;
      document.getElementById('winrate').textContent = wr + '%';
      document.getElementById('winrate-bar').style.width = wr + '%';

      const profit = session.net_profit ?? 0;
      const profitEl = document.getElementById('profit');
      profitEl.textContent = (profit >= 0 ? '+$' : '-$') + Math.abs(profit).toFixed(2);
      profitEl.style.color = profit >= 0 ? 'var(--accent)' : 'var(--red)';

      const signals = session.signals_fired ?? 0;
      document.getElementById('signals').textContent = signals;
      document.getElementById('signal-rate').textContent = total > 0 ? (signals/total).toFixed(1) + ' per trade' : '—';

      const elapsed = session.elapsed_minutes ? Math.round(session.elapsed_minutes) : 0;
      const h = Math.floor(elapsed / 60), m = elapsed % 60;
      document.getElementById('elapsed').textContent = h + 'h ' + String(m).padStart(2,'0') + 'm';

      const baseT = d.base_confidence_threshold ?? 0;
      document.getElementById('base-threshold').textContent = Math.round(baseT * 100) + '%';
      const effT = d.confidence_threshold ?? 0;
      document.getElementById('threshold').textContent = Math.round(effT * 100) + '%';
      document.getElementById('threshold-bar').style.width = (effT * 100) + '%';

      const streak = d.martingale_streak ?? 0;
      const maxStreak = d.martingale_max_streak ?? 3;
      document.getElementById('streak').textContent = streak + '/' + maxStreak;
      document.getElementById('max-streak').textContent = maxStreak;
      document.getElementById('pending').textContent = d.pending_signals ?? 0;
      document.getElementById('practice-mode').textContent = d.practice_mode ? 'Practice' : 'Live';
      document.getElementById('practice-mode').style.color = d.practice_mode ? 'var(--amber)' : 'var(--red)';

      const stream = d.stream || {};
      document.getElementById('stream-status').textContent = stream.connected ? 'Online' : 'Offline';
      document.getElementById('stream-status').style.color = stream.connected ? 'var(--accent2)' : 'var(--red)';
      document.getElementById('ticks').textContent = stream.ticks_received ? Number(stream.ticks_received).toLocaleString() : '—';

      const quotex = d.quotex || {};
      document.getElementById('quotex-status').textContent = quotex.connected ? 'Online' : 'Offline';
      document.getElementById('quotex-status').style.color = quotex.connected ? 'var(--accent2)' : 'var(--amber)';
      document.getElementById('quotex-balance').textContent = quotex.balance != null ? '$' + Number(quotex.balance).toFixed(2) : '—';

      renderTrades(d.recent_trades);

      const recentEvents = d.recent_events || [];
      if (recentEvents.length > 0) {
        activity.length = 0;
        for (const ev of recentEvents.slice(0, 40)) {
            activity.push({
                time: new Date(ev.timestamp).toLocaleTimeString(),
                text: ev.message,
                type: ev.type
            });
        }
        renderLog();
    }

      document.getElementById('ts').textContent = 'Updated ' + new Date().toLocaleTimeString();
    } catch(e) {
      document.getElementById('ts').textContent = 'Connection error…';
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
            "recent_events": [],
            # Metadata
            "started_at": local_now().isoformat(),
        }

    def update(self, data: dict[str, Any]) -> None:
        """Merge *data* into the current status."""
        with self._lock:
            self._data.update(data)

    def get(self) -> dict[str, Any]:
        """Return a deep-copy snapshot with server timestamp."""
        with self._lock:
            snapshot = copy.deepcopy(self._data)
            snapshot["server_time"] = local_now().isoformat()
            return snapshot

    def add_event(self, message: str, event_type: str = "info") -> None:
        """
        Push an event to the dashboard activity log.

        Args:
            message: Human-readable event description
            event_type: "info", "win", "loss", "signal", "kill"
        """
        with self._lock:
            events = self._data.get("recent_events", [])
            events.insert(
                0,
                {
                    "timestamp": local_now().isoformat(),
                    "message": message,
                    "type": event_type,
                },
            )
            # Keep last 100 events
            self._data["recent_events"] = events[:100]
            self._data["last_event"] = message

    @property
    def _lock_obj(self) -> threading.Lock:
        """Expose the lock for direct status_store._lock access (health_task)."""
        return self._lock


# Singleton — imported by pipeline.py
status_store: StatusStore = StatusStore()


# ── HTTP Handler ───────────────────────────────────────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    """Serves the dashboard HTML, JSON status, and health check."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._respond(200, "text/html", DASHBOARD_HTML.encode())
        elif self.path == "/status":
            snapshot = status_store.get()
            # recent_trades is now pushed by Reporter.push_dashboard()
            payload: bytes = json.dumps(snapshot, default=str).encode()
            self._respond(200, "application/json", payload)
        elif self.path == "/health":
            health = {
                "status": "ok",
                "kill_switch": status_store.get().get("kill_switch_active", False),
                "server_time": local_now().isoformat(),
            }
            self._respond(200, "application/json", json.dumps(health).encode())
        else:
            self._respond(404, "text/plain", b"not found")

    def _respond(self, code: int, content_type: str, body: bytes) -> None:
        try:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

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

    server: HTTPServer = HTTPServer(("0.0.0.0", port), _Handler)
    thread: Thread = Thread(target=server.serve_forever, daemon=True, name="dashboard")
    thread.start()

    logger.info(
        {
            "event": "DASHBOARD_STARTED",
            "port": port,
            "url": f"http://0.0.0.0:{port}",
        }
    )

    # Keep coroutine alive — daemon thread handles serving
    while True:
        await asyncio.sleep(60)
