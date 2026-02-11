async function loadSnapshot() {
  const candidates = ['./data/latest_snapshot.json', '/data/latest_snapshot.json'];
  for (const url of candidates) {
    const res = await fetch(url, { cache: 'no-store' });
    if (res.ok) return res.json();
  }
  throw new Error('Snapshot not found. Run scripts/build_dashboard_snapshot.py first.');
}

function format(v) {
  if (typeof v === 'boolean') return v ? 'YES' : 'NO';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(4);
  return String(v ?? '');
}

function renderTable(id, rows, preferredCols = []) {
  const table = document.getElementById(id);
  table.innerHTML = '';
  if (!rows || rows.length === 0) {
    table.innerHTML = '<tr><td>No data available</td></tr>';
    return;
  }
  const cols = preferredCols.length ? preferredCols : Object.keys(rows[0]);
  const thead = document.createElement('thead');
  const hr = document.createElement('tr');
  cols.forEach((c) => {
    const th = document.createElement('th');
    th.textContent = c;
    hr.appendChild(th);
  });
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  rows.forEach((r) => {
    const tr = document.createElement('tr');
    cols.forEach((c) => {
      const td = document.createElement('td');
      td.textContent = format(r[c]);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

function renderCards(data) {
  const node = document.getElementById('metric-cards');
  const metrics = data.metrics || {};
  const mode = data.data_mode || 'unknown';
  const cardKeys = [
    ['model_oos_precision', 'Model OOS Precision'],
    ['walk_forward_precision', 'WF Precision'],
    ['walk_forward_sharpe', 'WF Sharpe'],
    ['model_ready', 'Model Ready'],
    ['win_rate', 'Win Rate'],
  ];
  node.innerHTML = cardKeys
    .map(([k, label]) => {
      const v = metrics[k];
      const display = k === 'model_ready' ? (v ? '<span class="badge-ok">YES</span>' : '<span class="badge-no">NO</span>') : format(v);
      return `<div class=\"card\"><div class=\"label\">${label}</div><div class=\"value\">${display}</div><div class=\"label\">mode: ${mode}</div><div class=\"label\">generated: ${data.generated_at_utc || 'n/a'}</div></div>`;
    })
    .join('');
}

function renderSafeguards(data) {
  const node = document.getElementById('safeguards');
  const s = data.operational_safeguards || {};
  const stale = (s.stale_alerts || []).length ? (s.stale_alerts || []).join(' | ') : 'none';
  const lat = s.fetch_latency_ms || {};
  const latText = Object.keys(lat).length ? Object.entries(lat).map(([k,v]) => `${k}:${v}ms`).join(', ') : 'n/a';
  node.innerHTML = `
    <div class="safeguard-row"><strong>Market Open (ET):</strong> ${format(s.market_open_et)}</div>
    <div class="safeguard-row"><strong>Failover Mode:</strong> ${format(s.failover_mode)}</div>
    <div class="safeguard-row warn"><strong>Stale Alerts:</strong> ${stale}</div>
    <div class="safeguard-row"><strong>Fetch Latency:</strong> ${latText}</div>
    <div class=\"safeguard-row\"><strong>Ready Tickers:</strong> ${(s.data_ready_tickers || []).join(', ') || 'none'}</div>
    <div class=\"safeguard-row warn\"><strong>Dropped Tickers:</strong> ${Object.keys(s.dropped_tickers || {}).length ? JSON.stringify(s.dropped_tickers) : 'none'}</div>
  `;
}

(async () => {
  try {
    const data = await loadSnapshot();
    renderCards(data);
    renderSafeguards(data);

    renderTable('top-table', data.top_ranked_opportunities || [], [
      'ticker', 'pattern', 'entry', 'stop', 'target', 'risk_reward', 'ml_probability', 'liquidity_20d_usd', 'opportunity_score', 'timestamp'
    ]);

    renderTable('active-table', data.active_setups || [], [
      'ticker', 'timeframe', 'pattern', 'entry', 'stop', 'target', 'risk_reward', 'quantity', 'ml_probability', 'timestamp'
    ]);
    renderTable('watch-table', data.watchlist || [], ['ticker', 'timeframe', 'state', 'timestamp']);

    renderTable('backtest-table', [
      {
        win_rate: data.metrics?.win_rate,
        avg_r: data.metrics?.avg_r,
        max_drawdown: data.metrics?.max_drawdown,
        sharpe: data.metrics?.sharpe,
      },
    ]);
  } catch (err) {
    document.body.innerHTML = `<main class="container"><div class="panel"><h2>Dashboard data not ready</h2><p>${err.message}</p></div></main>`;
  }
})();
