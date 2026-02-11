async function loadSnapshot() {
  const res = await fetch('/data/latest_snapshot.json', { cache: 'no-store' });
  if (!res.ok) throw new Error('Snapshot not found. Run scripts/build_dashboard_snapshot.py first.');
  return res.json();
}

function format(v) {
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

function renderCards(metrics) {
  const node = document.getElementById('metric-cards');
  const cardKeys = [
    ['model_oos_precision', 'Model OOS Precision'],
    ['walk_forward_precision', 'WF Precision'],
    ['walk_forward_sharpe', 'WF Sharpe'],
    ['model_ready', 'Model Ready'],
  ];
  node.innerHTML = cardKeys
    .map(([k, label]) => {
      const v = metrics[k];
      const display = k === 'model_ready' ? (v ? '<span class="badge-ok">YES</span>' : '<span class="badge-no">NO</span>') : format(v);
      return `<div class="card"><div class="label">${label}</div><div class="value">${display}</div></div>`;
    })
    .join('');
}

(async () => {
  try {
    const data = await loadSnapshot();
    renderCards(data.metrics || {});

    renderTable('active-table', data.active_setups || [], [
      'ticker',
      'timeframe',
      'pattern',
      'entry',
      'stop',
      'target',
      'risk_reward',
      'quantity',
      'ml_probability',
      'timestamp',
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
