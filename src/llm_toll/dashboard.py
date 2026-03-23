"""Web dashboard for viewing LLM cost and usage statistics."""

from __future__ import annotations

import http.server
import json
from typing import Any
from urllib.parse import parse_qs, urlparse

from llm_toll.store import BaseStore

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>llm-toll Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f1117; color: #e0e0e0; padding: 24px;
  }
  h1 { font-size: 1.6rem; margin-bottom: 24px; color: #fff; }
  h2 { font-size: 1.1rem; margin-bottom: 12px; color: #c0c0c0; }
  .cards {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px; margin-bottom: 32px;
  }
  .card {
    background: #1a1d27; border-radius: 10px; padding: 20px;
    border: 1px solid #2a2d37;
  }
  .card .label { font-size: 0.85rem; color: #888; margin-bottom: 4px; }
  .card .value { font-size: 1.8rem; font-weight: 700; color: #4fc3f7; }
  .chart-container {
    background: #1a1d27; border-radius: 10px; padding: 20px;
    border: 1px solid #2a2d37; margin-bottom: 32px; max-height: 400px;
  }
  table {
    width: 100%; border-collapse: collapse; background: #1a1d27;
    border-radius: 10px; overflow: hidden; margin-bottom: 32px;
    border: 1px solid #2a2d37;
  }
  th, td { padding: 10px 14px; text-align: left; }
  th { background: #22252f; color: #aaa; font-size: 0.8rem; text-transform: uppercase; }
  td { border-top: 1px solid #2a2d37; font-size: 0.9rem; }
  tr:hover td { background: #22252f; }
  .right { text-align: right; }
  .refresh-note { font-size: 0.75rem; color: #555; margin-top: 16px; }
</style>
</head>
<body>
<h1>llm-toll Dashboard</h1>

<div class="cards">
  <div class="card"><div class="label">Total Cost</div><div class="value" id="total-cost">--</div></div>
  <div class="card"><div class="label">Total Calls</div><div class="value" id="total-calls">--</div></div>
  <div class="card"><div class="label">Projects</div><div class="value" id="project-count">--</div></div>
  <div class="card"><div class="label">Models</div><div class="value" id="model-count">--</div></div>
</div>

<h2>Daily Cost Trend (Last 30 Days)</h2>
<div class="chart-container"><canvas id="trend-chart"></canvas></div>

<h2>Projects</h2>
<table id="projects-table">
  <thead><tr>
    <th>Project</th><th class="right">Calls</th>
    <th class="right">Input Tokens</th><th class="right">Output Tokens</th>
    <th class="right">Total Cost</th>
  </tr></thead>
  <tbody></tbody>
</table>

<h2>Models</h2>
<table id="models-table">
  <thead><tr>
    <th>Model</th><th class="right">Calls</th>
    <th class="right">Input Tokens</th><th class="right">Output Tokens</th>
    <th class="right">Total Cost</th>
  </tr></thead>
  <tbody></tbody>
</table>

<div class="refresh-note">Auto-refreshes every 30 seconds.</div>

<script>
let trendChart = null;

function fmt(n) { return n == null ? '0' : Number(n).toLocaleString(); }
function fmtCost(n) { return '$' + Number(n || 0).toFixed(4); }

async function fetchJSON(url) {
  const r = await fetch(url);
  return r.json();
}

async function refresh() {
  const [summary, trends, projects, models] = await Promise.all([
    fetchJSON('/api/summary'),
    fetchJSON('/api/trends'),
    fetchJSON('/api/projects'),
    fetchJSON('/api/models'),
  ]);

  document.getElementById('total-cost').textContent = fmtCost(summary.total_cost);
  document.getElementById('total-calls').textContent = fmt(summary.total_calls);
  document.getElementById('project-count').textContent = fmt(summary.project_count);
  document.getElementById('model-count').textContent = fmt(summary.model_count);

  // Chart
  const labels = trends.map(t => t.date);
  const data = trends.map(t => t.daily_cost);
  if (trendChart) trendChart.destroy();
  trendChart = new Chart(document.getElementById('trend-chart'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Daily Cost ($)',
        data,
        borderColor: '#4fc3f7',
        backgroundColor: 'rgba(79,195,247,0.1)',
        fill: true, tension: 0.3,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: '#888' }, grid: { color: '#2a2d37' } },
        y: { ticks: { color: '#888', callback: v => '$' + v.toFixed(4) }, grid: { color: '#2a2d37' } }
      },
      plugins: { legend: { labels: { color: '#ccc' } } }
    }
  });

  // Projects table
  const pBody = document.querySelector('#projects-table tbody');
  pBody.innerHTML = projects.map(p =>
    `<tr><td>${p.project}</td><td class="right">${fmt(p.call_count)}</td>` +
    `<td class="right">${fmt(p.total_input_tokens)}</td>` +
    `<td class="right">${fmt(p.total_output_tokens)}</td>` +
    `<td class="right">${fmtCost(p.total_cost)}</td></tr>`
  ).join('');

  // Models table
  const mBody = document.querySelector('#models-table tbody');
  mBody.innerHTML = models.map(m =>
    `<tr><td>${m.model}</td><td class="right">${fmt(m.call_count)}</td>` +
    `<td class="right">${fmt(m.total_input_tokens)}</td>` +
    `<td class="right">${fmt(m.total_output_tokens)}</td>` +
    `<td class="right">${fmtCost(m.total_cost)}</td></tr>`
  ).join('');
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>
"""


def _json_default(obj: object) -> str:
    """Fallback serializer for JSON encoding."""
    return str(obj)


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler serving the llm-toll web dashboard."""

    store: BaseStore  # set dynamically via subclass

    def do_GET(self) -> None:
        """Dispatch GET requests to the appropriate handler."""
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._send_html(_DASHBOARD_HTML)
        elif path == "/api/summary":
            self._handle_summary()
        elif path == "/api/trends":
            days = int(params.get("days", ["30"])[0])
            self._send_json(self.store.get_daily_cost_trends(days=days))
        elif path == "/api/projects":
            self._send_json(self.store.get_all_project_summaries())
        elif path == "/api/models":
            self._send_json(self.store.get_model_summaries())
        elif path == "/api/budgets":
            self._send_json(self.store.get_budget_utilization())
        elif path == "/api/logs":
            project = params.get("project", [None])[0]
            limit = int(params.get("limit", ["1000"])[0])
            self._send_json(self.store.get_usage_logs_filtered(project=project, limit=limit))
        else:
            self.send_error(404, "Not Found")

    def _handle_summary(self) -> None:
        projects = self.store.get_all_project_summaries()
        models = self.store.get_model_summaries()
        total_cost = sum(p.get("total_cost", 0) for p in projects)
        total_calls = sum(p.get("call_count", 0) for p in projects)
        self._send_json(
            {
                "total_cost": total_cost,
                "total_calls": total_calls,
                "project_count": len(projects),
                "model_count": len(models),
            }
        )

    def _send_json(self, data: Any) -> None:
        body = json.dumps(data, default=_json_default).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default request logging."""


def serve_dashboard(store: BaseStore, port: int = 8050) -> None:
    """Start the web dashboard server.

    Parameters
    ----------
    store:
        The usage store backend to query.
    port:
        TCP port to listen on (default 8050).
    """
    handler = type("Handler", (DashboardHandler,), {"store": store})
    server = http.server.HTTPServer(("127.0.0.1", port), handler)
    print(f"Dashboard running at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
