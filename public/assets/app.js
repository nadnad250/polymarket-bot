// Polymarket BTC Bot — Dashboard static (GitHub Pages)
// Lit data/*.json générés par les GitHub Actions et met à jour le UI.
// v2: bypass cache CDN GitHub Pages via raw.githubusercontent.com + freshness UI.

const REPO_OWNER = "nadnad250";
const REPO_NAME = "polymarket-bot";
const REPO_BRANCH = "main";
const RAW_BASE = `https://raw.githubusercontent.com/${REPO_OWNER}/${REPO_NAME}/${REPO_BRANCH}/public/data/`;
const FALLBACK_BASE = "data/";
const REFRESH_MS = 15000;
const BOT_RUN_INTERVAL_MS = 10 * 60 * 1000; // 10 min (cron throttlé par GitHub)

const fmt = {
  usd: (v) => "$" + Number(v).toLocaleString("fr-FR", { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
  pct: (v) => (v > 0 ? "+" : "") + Number(v).toFixed(2) + "%",
  pctNoSign: (v) => Number(v).toFixed(2) + "%",
  num: (v) => Number(v).toLocaleString("fr-FR"),
  time: (ts) => new Date(ts).toLocaleTimeString("fr-FR"),
  duration: (ms) => {
    if (ms < 0) ms = 0;
    const totalSec = Math.floor(ms / 1000);
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    if (m >= 60) {
      const h = Math.floor(m / 60);
      const mm = m % 60;
      return `${h}h ${mm}m`;
    }
    return `${m}m ${s.toString().padStart(2, "0")}s`;
  },
  ago: (ms) => {
    if (ms < 60_000) return Math.max(0, Math.floor(ms / 1000)) + "s ago";
    if (ms < 3_600_000) return Math.floor(ms / 60_000) + " min ago";
    return Math.floor(ms / 3_600_000) + "h ago";
  },
};

// --- State ---
const state = {
  lastTradesCount: null,
  lastTickTs: null,
  lastUpdatedAt: null,
  countdownTimer: null,
};

// --- Fetch with raw.githubusercontent fallback ---
async function fetchJSON(filename, { force = false } = {}) {
  const cacheBuster = "?t=" + Date.now() + (force ? "&f=1" : "");
  const urls = [RAW_BASE + filename + cacheBuster, FALLBACK_BASE + filename + cacheBuster];
  for (const url of urls) {
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) continue;
      return await r.json();
    } catch {
      // try next
    }
  }
  return null;
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

// --- Freshness indicator ---
function getDataTimestamp(latest) {
  if (!latest) return null;
  const candidate = latest.updated_at || latest.ts;
  if (!candidate) return null;
  const d = new Date(candidate);
  return isNaN(d.getTime()) ? null : d.getTime();
}

function renderFreshness(latest) {
  const el = document.getElementById("last-update");
  if (!el) return;
  const ts = getDataTimestamp(latest);
  if (!ts) {
    el.textContent = "Data indisponible";
    el.style.color = "#f85149";
    el.style.fontWeight = "600";
    return;
  }
  const ageMs = Date.now() - ts;
  const ageStr = fmt.ago(ageMs);
  let label, color;
  if (ageMs < 2 * 60 * 1000) {
    label = "Data fraîche";
    color = "#3fb950";
  } else if (ageMs < 15 * 60 * 1000) {
    label = "Data récente";
    color = "#d29922";
  } else {
    label = "Data ancienne";
    color = "#f85149";
  }
  el.innerHTML = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:6px;vertical-align:middle;box-shadow:0 0 8px ${color}"></span><strong style="color:${color}">${label}</strong> · ${ageStr} <span id="next-run-countdown" style="color:#7a8598;margin-left:8px;font-size:0.9em"></span>`;
  el.style.color = color;
  el.style.fontWeight = "600";
}

function updateCountdown() {
  const target = document.getElementById("next-run-countdown");
  if (!target) return;
  if (!state.lastUpdatedAt) {
    target.textContent = "";
    return;
  }
  const expectedNext = Math.max(state.lastUpdatedAt + BOT_RUN_INTERVAL_MS, Date.now() + 60_000);
  const remaining = expectedNext - Date.now();
  target.textContent = `· Prochain run estimé dans ${fmt.duration(remaining)}`;
}

// --- Notifications visuelles ---
function flashElement(el, color = "#3fb950", durationMs = 2000) {
  if (!el) return;
  const original = el.style.boxShadow;
  const originalBorder = el.style.border;
  el.style.transition = "box-shadow 0.3s ease, border-color 0.3s ease";
  el.style.boxShadow = `0 0 0 2px ${color}, 0 0 24px ${color}`;
  el.style.border = `1px solid ${color}`;
  setTimeout(() => {
    el.style.boxShadow = original;
    el.style.border = originalBorder;
  }, durationMs);
}

// --- Force refresh button ---
function ensureForceRefreshButton() {
  if (document.getElementById("force-refresh-btn")) return;
  const anchor = document.getElementById("last-update");
  if (!anchor || !anchor.parentNode) return;
  const btn = document.createElement("button");
  btn.id = "force-refresh-btn";
  btn.type = "button";
  btn.textContent = "↻ Force refresh";
  btn.title = "Refetch immédiat (bypass cache)";
  btn.style.cssText = [
    "margin-left:10px",
    "padding:4px 10px",
    "font-size:0.85em",
    "font-weight:600",
    "background:#1f6feb",
    "color:#fff",
    "border:none",
    "border-radius:6px",
    "cursor:pointer",
    "transition:background 0.2s, transform 0.1s",
    "vertical-align:middle",
  ].join(";");
  btn.addEventListener("mouseenter", () => (btn.style.background = "#388bfd"));
  btn.addEventListener("mouseleave", () => (btn.style.background = "#1f6feb"));
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "↻ Refresh…";
    btn.style.opacity = "0.7";
    try {
      await refresh({ force: true });
    } finally {
      btn.disabled = false;
      btn.textContent = "↻ Force refresh";
      btn.style.opacity = "1";
    }
  });
  anchor.parentNode.insertBefore(btn, anchor.nextSibling);
}

// --- Renderers ---
function renderLatest(latest) {
  if (!latest || !latest.btc_price) return;
  const priceEl = document.getElementById("btc-price");
  const priceCard = priceEl ? priceEl.closest(".card, .stat-card, .metric-card") || priceEl.parentElement : null;

  if (state.lastTickTs && latest.ts && latest.ts !== state.lastTickTs) {
    flashElement(priceCard, "#f7931a", 1500);
  }
  state.lastTickTs = latest.ts;

  if (priceEl) priceEl.textContent = fmt.usd(latest.btc_price);
  const subEl = document.getElementById("btc-sub");
  if (subEl) subEl.textContent = "Binance spot · mis à jour " + fmt.time(latest.ts);
  const polyYes = document.getElementById("poly-yes");
  if (polyYes) polyYes.textContent = (latest.poly_yes * 100).toFixed(1) + "%";
  const polyEvent = document.getElementById("poly-event");
  if (polyEvent) polyEvent.textContent = latest.poly_question || "—";
  const spread = document.getElementById("spread");
  if (spread) spread.textContent = (latest.spread_bps || 0).toFixed(1) + " bps";
  const imb = latest.ob_imb || 0;
  const imbEl = document.getElementById("obimb");
  if (imbEl) {
    imbEl.textContent = (imb > 0 ? "+" : "") + imb.toFixed(3);
    imbEl.className = "big-price " + (imb > 0 ? "pos" : imb < 0 ? "neg" : "");
  }

  state.lastUpdatedAt = getDataTimestamp(latest);
  renderFreshness(latest);
  ensureForceRefreshButton();
  updateCountdown();
}

function renderTrades(data) {
  if (!data || !data.metrics) return;
  const m = data.metrics;

  const cap = document.getElementById("capital");
  if (cap) cap.textContent = fmt.usd(m.capital || 1000);
  const capSub = document.getElementById("capital-sub");
  if (capSub) capSub.textContent = "initial $1000";

  const roiEl = document.getElementById("roi");
  if (roiEl) {
    const roi = m.roi_pct || 0;
    roiEl.textContent = (roi > 0 ? "+" : "") + roi.toFixed(2) + "%";
    roiEl.className = "value " + (roi > 0 ? "pos" : roi < 0 ? "neg" : "");
  }

  const tradesEl = document.getElementById("trades");
  if (tradesEl) tradesEl.textContent = m.total_trades || 0;
  const tradesSub = document.getElementById("trades-sub");
  if (tradesSub) tradesSub.textContent = (data.trades?.filter(t => t.outcome == null).length || 0) + " ouverts";

  const wr = document.getElementById("wr");
  if (wr) wr.textContent = m.win_rate != null ? (m.win_rate * 100).toFixed(1) + "%" : "—";
  const wrSub = document.getElementById("wr-sub");
  if (wrSub) wrSub.textContent = m.total_trades ? `sur ${m.total_trades} trades` : "—";

  const pnlEl = document.getElementById("pnl");
  if (pnlEl) {
    pnlEl.textContent = (m.total_pnl >= 0 ? "+" : "") + fmt.usd(m.total_pnl || 0).replace("$", "$");
    pnlEl.className = "value " + (m.total_pnl > 0 ? "pos" : m.total_pnl < 0 ? "neg" : "");
  }
  const pnlSub = document.getElementById("pnl-sub");
  if (pnlSub) pnlSub.textContent = "moy/trade " + (m.avg_pnl != null ? fmt.usd(m.avg_pnl) : "—");

  const bw = document.getElementById("bestworst");
  if (bw) {
    bw.innerHTML =
      `<span class="pos">${fmt.usd(m.best_trade || 0)}</span> / <span class="neg">${fmt.usd(m.worst_trade || 0)}</span>`;
  }

  // Trades table + flash on new trade
  const tbl = document.getElementById("trades-tbl");
  const tbody = tbl ? tbl.querySelector("tbody") : null;
  const currentCount = (data.trades || []).length;
  if (state.lastTradesCount !== null && currentCount > state.lastTradesCount) {
    flashElement(tbl, "#3fb950", 2000);
  }
  state.lastTradesCount = currentCount;

  if (tbody) {
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
  }

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
  if (typeof Plotly === "undefined") return;
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
  if (typeof Plotly === "undefined") return;
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
  const dateEl = document.getElementById("m-date");
  if (!metrics || Object.keys(metrics).length === 0) {
    if (dateEl) dateEl.textContent = "jamais entraîné";
    return;
  }
  const m = metrics.metrics || metrics;
  const setText = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  setText("m-acc", m.accuracy != null ? (m.accuracy * 100).toFixed(2) + "%" : "—");
  setText("m-auc", m.auc != null ? m.auc.toFixed(3) : "—");
  setText("m-brier", m.brier != null ? m.brier.toFixed(4) : "—");
  setText("m-logloss", m.logloss != null ? m.logloss.toFixed(4) : "—");
  setText("m-train", m.n_train != null ? fmt.num(m.n_train) : "—");
  if (dateEl) {
    dateEl.textContent = metrics.trained_at
      ? new Date(metrics.trained_at).toLocaleString("fr-FR")
      : "—";
  }
}

// --- Main refresh loop ---
async function refresh({ force = false } = {}) {
  const [latest, ticks, trades, model] = await Promise.all([
    fetchJSON("latest.json", { force }),
    fetchJSON("ticks.json", { force }),
    fetchJSON("trades.json", { force }),
    fetchJSON("metrics.json", { force }),
  ]);
  if (latest) renderLatest(latest);
  if (ticks) renderTicks(ticks);
  if (trades) renderTrades(trades);
  if (model) renderModel(model);
  // Si pas de latest, on affiche quand même un état stale
  if (!latest) renderFreshness(null);
}

// Auto-detect GitHub repo URL for footer
(function detectRepo() {
  const host = location.hostname;
  if (host.endsWith("github.io")) {
    const user = host.split(".")[0];
    const repo = location.pathname.split("/")[1] || REPO_NAME;
    const link = document.getElementById("repo-link");
    if (link) link.href = `https://github.com/${user}/${repo}`;
  }
})();

// Boot
refresh();
setInterval(refresh, REFRESH_MS);
state.countdownTimer = setInterval(updateCountdown, 1000);
