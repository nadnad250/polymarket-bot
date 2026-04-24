// Polymarket BTC Bot — Dashboard static (GitHub Pages)
// Lit data/*.json générés par les GitHub Actions et met à jour le UI.

const DATA_URL = "data/";
const REFRESH_MS = 30000;

const fmt = {
  usd: (v) => "$" + Number(v).toLocaleString("fr-FR", { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
  pct: (v) => (v > 0 ? "+" : "") + Number(v).toFixed(2) + "%",
  pctNoSign: (v) => Number(v).toFixed(2) + "%",
  num: (v) => Number(v).toLocaleString("fr-FR"),
  time: (ts) => new Date(ts).toLocaleTimeString("fr-FR"),
};

async function fetchJSON(path) {
  try {
    const r = await fetch(DATA_URL + path + "?t=" + Date.now());
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

function setSignedValue(id, v, unit = "%") {
  const el = document.getElementById(id);
  if (!el) return;
  if (v == null || isNaN(v)) {
    el.textContent = "—";
    el.className = "value";
    return;
  }
  el.textContent = (v > 0 ? "+" : "") + Number(v).toFixed(2) + unit;
  el.classList.remove("pos", "neg");
  if (v > 0) el.classList.add("pos");
  else if (v < 0) el.classList.add("neg");
}

function renderLatest(latest) {
  if (!latest || !latest.btc_price) return;
  document.getElementById("btc-price").textContent = fmt.usd(latest.btc_price);
  document.getElementById("btc-sub").textContent = "Binance spot · mis à jour " + fmt.time(latest.ts);
  document.getElementById("poly-yes").textContent = (latest.poly_yes * 100).toFixed(1) + "%";
  document.getElementById("poly-event").textContent = latest.poly_question || "—";
  document.getElementById("spread").textContent = (latest.spread_bps || 0).toFixed(1) + " bps";
  const imb = latest.ob_imb || 0;
  const imbEl = document.getElementById("obimb");
  imbEl.textContent = (imb > 0 ? "+" : "") + imb.toFixed(3);
  imbEl.className = "big-price " + (imb > 0 ? "pos" : imb < 0 ? "neg" : "");

  document.getElementById("last-update").textContent = "Dernière update : " + fmt.time(latest.ts);
}

function renderTrades(data) {
  if (!data || !data.metrics) return;
  const m = data.metrics;

  document.getElementById("capital").textContent = fmt.usd(m.capital || 1000);
  document.getElementById("capital-sub").textContent = "initial $1000";

  const roiEl = document.getElementById("roi");
  const roi = m.roi_pct || 0;
  roiEl.textContent = (roi > 0 ? "+" : "") + roi.toFixed(2) + "%";
  roiEl.className = "value " + (roi > 0 ? "pos" : roi < 0 ? "neg" : "");

  document.getElementById("trades").textContent = m.total_trades || 0;
  document.getElementById("trades-sub").textContent = (data.trades?.filter(t => t.outcome == null).length || 0) + " ouverts";

  document.getElementById("wr").textContent = m.win_rate != null ? (m.win_rate * 100).toFixed(1) + "%" : "—";
  document.getElementById("wr-sub").textContent = m.total_trades ? `sur ${m.total_trades} trades` : "—";

  const pnlEl = document.getElementById("pnl");
  pnlEl.textContent = (m.total_pnl >= 0 ? "+" : "") + fmt.usd(m.total_pnl || 0).replace("$", "$");
  pnlEl.className = "value " + (m.total_pnl > 0 ? "pos" : m.total_pnl < 0 ? "neg" : "");
  document.getElementById("pnl-sub").textContent = "moy/trade " + (m.avg_pnl != null ? fmt.usd(m.avg_pnl) : "—");

  document.getElementById("bestworst").innerHTML =
    `<span class="pos">${fmt.usd(m.best_trade || 0)}</span> / <span class="neg">${fmt.usd(m.worst_trade || 0)}</span>`;

  // Trades table
  const tbody = document.querySelector("#trades-tbl tbody");
  tbody.innerHTML = "";
  (data.trades || []).slice(0, 50).forEach(t => {
    const tr = document.createElement("tr");
    const outcomeBadge = t.outcome == null ? '<span class="tag">—</span>'
      : t.outcome ? '<span class="tag tag-win">WIN</span>'
      : '<span class="tag tag-loss">LOSS</span>';
    const pnlClass = t.pnl > 0 ? "pos" : t.pnl < 0 ? "neg" : "";
    tr.innerHTML = `
      <td>${fmt.time(t.ts)}</td>
      <td><span class="tag tag-${t.side.toLowerCase()}">${t.side}</span></td>
      <td>${t.entry.toFixed(3)}</td>
      <td>$${t.size.toFixed(2)}</td>
      <td>${t.btc_entry.toLocaleString("fr-FR")} → ${t.btc_exit ? t.btc_exit.toLocaleString("fr-FR") : "—"}</td>
      <td>${(t.momentum * 100).toFixed(3)}%</td>
      <td>${t.imbalance.toFixed(2)}</td>
      <td>${outcomeBadge}</td>
      <td class="${pnlClass}">${t.pnl >= 0 ? "+" : ""}$${t.pnl.toFixed(2)}</td>
    `;
    tbody.appendChild(tr);
  });

  // Equity curve
  if (data.equity && data.equity.length) {
    renderEquity(data.equity);
  }
}

const plotlyLayout = {
  plot_bgcolor: "#0a0d14",
  paper_bgcolor: "transparent",
  font: { color: "#e4e8f0", family: "Inter" },
  margin: { t: 20, r: 16, b: 40, l: 60 },
  xaxis: { gridcolor: "#232a3a", linecolor: "#232a3a" },
  yaxis: { gridcolor: "#232a3a", linecolor: "#232a3a" },
  hovermode: "x unified",
};

function renderTicks(ticks) {
  if (!ticks || !ticks.length) return;
  const ts = ticks.map(t => new Date(t.ts));
  const btc = ticks.map(t => t.btc);
  const yes = ticks.map(t => t.yes * 100);

  Plotly.react("chart-btc", [{
    x: ts, y: btc, type: "scatter", mode: "lines",
    line: { color: "#f7931a", width: 2 },
    fill: "tozeroy",
    fillcolor: "rgba(247, 147, 26, 0.08)",
    name: "BTC",
    hovertemplate: "<b>$%{y:,.2f}</b><extra></extra>",
  }], { ...plotlyLayout, yaxis: { ...plotlyLayout.yaxis, tickformat: "$,.0f" } }, { displayModeBar: false, responsive: true });

  Plotly.react("chart-poly", [{
    x: ts, y: yes, type: "scatter", mode: "lines",
    line: { color: "#58a6ff", width: 2 },
    fill: "tozeroy",
    fillcolor: "rgba(88, 166, 255, 0.08)",
    name: "YES %",
    hovertemplate: "<b>%{y:.1f}%</b><extra></extra>",
  }], { ...plotlyLayout, yaxis: { ...plotlyLayout.yaxis, range: [0, 100], ticksuffix: "%" } }, { displayModeBar: false, responsive: true });
}

function renderEquity(equity) {
  const ts = equity.map(e => new Date(e.ts));
  const eq = equity.map(e => e.equity);
  const last = eq[eq.length - 1] || 1000;
  const color = last >= 1000 ? "#3fb950" : "#f85149";
  const fillcolor = last >= 1000 ? "rgba(63, 185, 80, 0.1)" : "rgba(248, 81, 73, 0.1)";

  Plotly.react("chart-equity", [{
    x: ts, y: eq, type: "scatter", mode: "lines",
    line: { color, width: 2.5 },
    fill: "tozeroy",
    fillcolor,
    name: "Equity",
    hovertemplate: "<b>$%{y:,.2f}</b><extra></extra>",
  }], {
    ...plotlyLayout,
    yaxis: { ...plotlyLayout.yaxis, tickformat: "$,.0f" },
    shapes: [{
      type: "line", xref: "paper", x0: 0, x1: 1,
      y0: 1000, y1: 1000,
      line: { color: "#7a8598", width: 1, dash: "dash" },
    }],
  }, { displayModeBar: false, responsive: true });
}

function renderModel(metrics) {
  if (!metrics || Object.keys(metrics).length === 0) {
    document.getElementById("m-date").textContent = "jamais entraîné";
    return;
  }
  const m = metrics.metrics || metrics;
  document.getElementById("m-acc").textContent = m.accuracy != null ? (m.accuracy * 100).toFixed(2) + "%" : "—";
  document.getElementById("m-auc").textContent = m.auc != null ? m.auc.toFixed(3) : "—";
  document.getElementById("m-brier").textContent = m.brier != null ? m.brier.toFixed(4) : "—";
  document.getElementById("m-logloss").textContent = m.logloss != null ? m.logloss.toFixed(4) : "—";
  document.getElementById("m-train").textContent = m.n_train != null ? fmt.num(m.n_train) : "—";
  document.getElementById("m-date").textContent = metrics.trained_at
    ? new Date(metrics.trained_at).toLocaleString("fr-FR")
    : "—";
}

async function refresh() {
  const [latest, ticks, trades, model] = await Promise.all([
    fetchJSON("latest.json"),
    fetchJSON("ticks.json"),
    fetchJSON("trades.json"),
    fetchJSON("metrics.json"),
  ]);
  if (latest) renderLatest(latest);
  if (ticks) renderTicks(ticks);
  if (trades) renderTrades(trades);
  if (model) renderModel(model);
}

// Auto-detect GitHub repo URL for footer
(function detectRepo() {
  const host = location.hostname;
  if (host.endsWith("github.io")) {
    const user = host.split(".")[0];
    const repo = location.pathname.split("/")[1] || "polymarket-bot";
    document.getElementById("repo-link").href = `https://github.com/${user}/${repo}`;
  }
})();

refresh();
setInterval(refresh, REFRESH_MS);
