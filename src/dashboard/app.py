"""Dashboard FastAPI — visualisation temps réel.

Affiche :
- Prix BTC live (Binance)
- Probabilité Polymarket (YES)
- Prédiction modèle (si chargé)
- Edge détecté + P&L simulateur
- Graphes interactifs (Plotly)

Usage:
    python -m src.dashboard.app
    → http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from config import DASHBOARD_HOST, DASHBOARD_PORT, DB_PATH

app = FastAPI(title="Polymarket Bot Dashboard")

HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Polymarket Bot — Live</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0b0e14; color: #e1e4e8; padding: 20px; }
  h1 { font-size: 24px; margin-bottom: 20px; color: #58a6ff; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .label { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
  .value { font-size: 28px; font-weight: 600; margin-top: 8px; }
  .pos { color: #3fb950; }
  .neg { color: #f85149; }
  .chart { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  .footer { font-size: 12px; color: #6e7681; margin-top: 20px; text-align: center; }
</style>
</head>
<body>
  <h1>⚡ Polymarket Bot — Live Dashboard</h1>
  <div class="grid">
    <div class="card"><div class="label">BTC Price</div><div class="value" id="btc">—</div></div>
    <div class="card"><div class="label">Polymarket YES</div><div class="value" id="yes">—</div></div>
    <div class="card"><div class="label">Model P(up)</div><div class="value" id="prob">—</div></div>
    <div class="card"><div class="label">Edge</div><div class="value" id="edge">—</div></div>
  </div>
  <div class="grid">
    <div class="card"><div class="label">Bankroll</div><div class="value" id="cash">$1000</div></div>
    <div class="card"><div class="label">Trades</div><div class="value" id="trades">0</div></div>
    <div class="card"><div class="label">Win Rate</div><div class="value" id="wr">—</div></div>
    <div class="card"><div class="label">ROI</div><div class="value" id="roi">0%</div></div>
  </div>
  <div class="chart"><div id="chart-price" style="height:360px"></div></div>
  <div class="chart"><div id="chart-prob" style="height:280px"></div></div>
  <div class="footer">Refresh: 5s · Source: Binance + Polymarket · Phase A (edge detection)</div>

<script>
async function refresh() {
  try {
    const r = await fetch('/api/snapshot');
    const d = await r.json();
    if (d.latest) {
      document.getElementById('btc').textContent = '$' + d.latest.btc_price.toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2});
      document.getElementById('yes').textContent = (d.latest.poly_yes * 100).toFixed(1) + '%';
      document.getElementById('prob').textContent = d.latest.model_prob != null ? (d.latest.model_prob * 100).toFixed(1) + '%' : '—';
      const edge = d.latest.edge;
      const edgeEl = document.getElementById('edge');
      edgeEl.textContent = edge != null ? (edge > 0 ? '+' : '') + (edge * 100).toFixed(1) + '%' : '—';
      edgeEl.className = 'value ' + (edge > 0 ? 'pos' : edge < 0 ? 'neg' : '');
    }
    const s = d.sim || {};
    document.getElementById('cash').textContent = '$' + (s.cash ?? 1000).toFixed(2);
    document.getElementById('trades').textContent = s.total_trades ?? 0;
    document.getElementById('wr').textContent = s.win_rate != null ? (s.win_rate * 100).toFixed(1) + '%' : '—';
    const roi = s.roi_pct ?? 0;
    const roiEl = document.getElementById('roi');
    roiEl.textContent = (roi > 0 ? '+' : '') + roi.toFixed(2) + '%';
    roiEl.className = 'value ' + (roi > 0 ? 'pos' : roi < 0 ? 'neg' : '');

    if (d.history && d.history.length) {
      const h = d.history;
      const ts = h.map(x => new Date(x.ts));
      Plotly.react('chart-price', [
        {x: ts, y: h.map(x => x.btc_price), name: 'BTC', line: {color: '#f7931a'}}
      ], {
        title: 'BTC Price (dernières 30 min)',
        plot_bgcolor: '#0b0e14', paper_bgcolor: '#161b22',
        font: {color: '#e1e4e8'}, margin: {t: 40, r: 20, b: 40, l: 60},
        xaxis: {gridcolor: '#30363d'}, yaxis: {gridcolor: '#30363d'}
      }, {displayModeBar: false});
      Plotly.react('chart-prob', [
        {x: ts, y: h.map(x => x.poly_yes * 100), name: 'Polymarket YES', line: {color: '#58a6ff'}},
        {x: ts, y: h.map(x => x.model_prob != null ? x.model_prob * 100 : null), name: 'Modèle P(up)', line: {color: '#3fb950'}},
      ], {
        title: 'Probabilités (%) — Marché vs Modèle',
        plot_bgcolor: '#0b0e14', paper_bgcolor: '#161b22',
        font: {color: '#e1e4e8'}, margin: {t: 40, r: 20, b: 40, l: 60},
        xaxis: {gridcolor: '#30363d'}, yaxis: {gridcolor: '#30363d', range: [0, 100]}
      }, {displayModeBar: false});
    }
  } catch(e) { console.error(e); }
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


def _load_recent_ticks(minutes: int = 30) -> pd.DataFrame:
    if not Path(DB_PATH).exists():
        return pd.DataFrame()
    with closing(sqlite3.connect(str(DB_PATH))) as conn:
        since = pd.Timestamp.now(tz="UTC").timestamp() * 1000 - minutes * 60 * 1000
        df = pd.read_sql(
            "SELECT * FROM ticks WHERE ts > ? ORDER BY ts ASC",
            conn,
            params=(since,),
        )
    return df


def _try_load_model():
    path = Path("data/model_lgbm.pkl")
    if not path.exists():
        return None
    try:
        from src.models.lgbm import load_model
        return load_model(path)
    except Exception:
        return None


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/api/snapshot")
def snapshot():
    df = _load_recent_ticks(30)
    if df.empty:
        return JSONResponse({"latest": None, "history": [], "sim": {}})

    model = _try_load_model()
    model_probs = None
    if model is not None and len(df) > 10:
        try:
            from src.models.features import build_features, FEATURE_COLS
            feats = build_features(df)
            X = feats[FEATURE_COLS].fillna(0)
            model_probs = model.predict_proba(X)[:, 1]
        except Exception:
            model_probs = None

    history = []
    for i, row in df.iterrows():
        prob = float(model_probs[i]) if model_probs is not None else None
        history.append({
            "ts": int(row["ts"]),
            "btc_price": float(row["btc_price"]),
            "poly_yes": float(row["poly_yes"]) if row["poly_yes"] else 0.5,
            "model_prob": prob,
        })

    latest_row = df.iloc[-1]
    latest_prob = float(model_probs[-1]) if model_probs is not None else None
    poly_yes = float(latest_row["poly_yes"]) if latest_row["poly_yes"] else 0.5
    edge = (latest_prob - poly_yes) if latest_prob is not None else None

    latest = {
        "ts": int(latest_row["ts"]),
        "btc_price": float(latest_row["btc_price"]),
        "poly_yes": poly_yes,
        "model_prob": latest_prob,
        "edge": edge,
    }

    # Charge résumé sim si dispo
    sim_path = Path("data/sim_summary.json")
    sim = {}
    if sim_path.exists():
        try:
            sim = json.loads(sim_path.read_text())
        except Exception:
            sim = {}

    return JSONResponse({"latest": latest, "history": history, "sim": sim})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=DASHBOARD_HOST, port=DASHBOARD_PORT)
