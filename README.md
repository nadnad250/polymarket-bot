# ⚡ Polymarket BTC Bot

> **Bot ML ultra avancé pour prédiction BTC Up/Down 5min sur Polymarket.**
> Ensemble LightGBM + XGBoost + LSTM · Kelly fractionnaire · Paper trading $1000.
> Auto-entraîné & déployé via GitHub Actions. Dashboard live sur GitHub Pages.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Auto-trained](https://img.shields.io/badge/auto--trained-every%206h-orange)
![Dashboard](https://img.shields.io/badge/dashboard-GitHub%20Pages-blueviolet)

## 🎯 Ce que fait le bot

1. **Collecte en continu** (toutes les 5 min) via GitHub Actions :
   - Prix BTC/USDT spot + orderbook (Binance REST, gratuit)
   - Probabilité Polymarket YES/NO (Gamma + CLOB API, gratuit)
2. **Entraîne un ensemble ML** toutes les 6h sur la data accumulée :
   - **LightGBM** (gradient boosting, baseline robuste)
   - **XGBoost** (boosting complémentaire)
   - **LSTM PyTorch** (séquentiel, capte la dynamique)
   - Pondération optimisée sur Brier score
3. **Paper trade $1000** — ouvre un trade par event 5min si edge ≥ 3%
4. **Dashboard live** auto-déployé → voir perf en direct sur GitHub Pages

## 🧠 Stratégie

Le marché Polymarket coté `P_YES` est comparé à la prédiction du modèle `P(up)` :

```
edge_YES = P(up) - P_YES
edge_NO  = (1 - P(up)) - (1 - P_YES)
```

Si `max(edge) ≥ 3%` → ouverture position via **Kelly fractionnaire** (max 2% du capital).
Frais modélisés : gas Polygon ~$0.05/tx + spread 2% + slippage 0.5%.

## 📊 Dashboard

**Live :** https://`<user>`.github.io/polymarket-bot/

Affiche :
- Prix BTC temps réel + probabilité Polymarket
- Courbe de capital (equity curve)
- Table des derniers trades avec outcome
- Métriques modèle (accuracy, AUC, Brier, LogLoss)
- Orderbook imbalance + spread bid/ask

## 🚀 Installation

### Local (dev / backtest)

```bash
git clone https://github.com/<user>/polymarket-bot
cd polymarket-bot

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
cp .env.example .env           # optionnel, défauts OK
```

### Lancer en local

```bash
# Collecte continue (feed la DB)
python -m src.fetchers.collector

# Dashboard local (autre terminal)
python -m src.dashboard.app
# → http://127.0.0.1:8000

# Simulateur live baseline (3e terminal)
python -m src.simulator.live_loop

# Entraîner le modèle (après >500 ticks collectés)
python scripts/ci_train.py

# Backtest walk-forward
python -c "from src.simulator.backtest import backtest; from src.fetchers.collector import init_db; import pandas as pd, sqlite3; df = pd.read_sql('SELECT * FROM ticks', sqlite3.connect('data/bot.db')); print(backtest(df).metrics)"
```

## ☁️ Déploiement GitHub Actions

Le bot tourne **100% gratuit sur GitHub**. Pour activer :

### 1. Fork / créer le repo

```bash
gh repo create polymarket-bot --public --source=. --remote=origin --push
```

### 2. Activer GitHub Pages

Repo → Settings → Pages → Source : **GitHub Actions**

### 3. Autoriser les Actions à écrire

Repo → Settings → Actions → General :
- Workflow permissions : **Read and write**
- ✅ Allow GitHub Actions to create and approve pull requests

### 4. C'est parti

Les 3 workflows se lancent automatiquement :

| Workflow | Fréquence | Rôle |
|---|---|---|
| `collect.yml` | toutes les 5 min | 1 tick + inférence + simulate, commit la DB |
| `train.yml` | toutes les 6h | re-entraîne l'ensemble ML, commit le modèle |
| `pages.yml` | sur push `public/**` | redéploie le dashboard |

Le dashboard sera en ligne sur `https://<user>.github.io/polymarket-bot/` sous ~2 min.

## 📁 Architecture

```
polymarket-bot/
├── src/
│   ├── fetchers/         # APIs Binance + Polymarket
│   ├── models/
│   │   ├── features.py   # Feature engineering (18 features)
│   │   ├── lgbm.py       # LightGBM baseline
│   │   ├── lstm.py       # PyTorch LSTM séquentiel
│   │   └── ensemble.py   # Ensemble pondéré des 3
│   ├── simulator/
│   │   ├── paper.py      # Kelly fractionnaire
│   │   ├── fees.py       # Modèle frais Polymarket réalistes
│   │   ├── backtest.py   # Walk-forward backtester
│   │   └── live_loop.py  # Simulateur baseline (momentum)
│   └── dashboard/
│       └── app.py        # FastAPI local
├── public/               # Dashboard static (GitHub Pages)
│   ├── index.html
│   ├── assets/
│   └── data/             # JSON générés par CI
├── scripts/
│   ├── ci_cycle.py       # Cycle 5 min pour CI
│   ├── ci_train.py       # Training pour CI
│   └── export_dashboard_data.py
├── notebooks/
│   └── 01_edge_detection.ipynb
└── .github/workflows/    # CI/CD
```

## 🔬 Features utilisées (modèle)

| Feature | Description |
|---|---|
| `ret_30s/60s/180s/300s` | Returns BTC multi-horizons |
| `vol_60s/300s` | Volatilité réalisée |
| `rsi_14` | RSI 14 périodes |
| `mom_1m/5m` | Momentum absolu |
| `ob_imb` | Orderbook imbalance Binance 10 niveaux |
| `ob_imb_avg_30s` | Imbalance moyenne 30s |
| `spread_pct` | Spread bid/ask normalisé |
| `poly_yes` | Probabilité implicite du marché |
| `poly_edge_vs_5050` | Écart vs 50/50 |
| `poly_vs_momentum` | Mispricing marché vs momentum |

## ⚠️ Avertissements

- Éducation et recherche uniquement. **Pas un conseil financier.**
- 5 min BTC up/down est proche du bruit statistique (edge typique 1-3%).
- Frais + spread Polymarket bouffent ~2% d'edge → backtester critique.
- Paper trading obligatoire pendant ≥ 2 semaines avant argent réel.
- Jamais plus de 2% du capital par trade.
- `--no-verify` et skip de hooks interdits.

## 📜 Licence

MIT
