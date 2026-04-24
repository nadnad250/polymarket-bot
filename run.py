"""Script orchestrateur — lance collector + dashboard en parallèle.

Usage:
    python run.py           # dashboard seul
    python run.py --collect # collector + dashboard
    python run.py --train   # entraîne le modèle depuis data existante
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time

from config import DASHBOARD_HOST, DASHBOARD_PORT


def start_collector():
    from src.fetchers.collector import run
    run()


def start_dashboard():
    import uvicorn
    from src.dashboard.app import app
    uvicorn.run(app, host=DASHBOARD_HOST, port=DASHBOARD_PORT, log_level="warning")


def train_model():
    import sqlite3
    import pandas as pd
    from config import DB_PATH
    from src.models.features import build_features
    from src.models.lgbm import save_model, train

    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("SELECT * FROM ticks ORDER BY ts", conn)
    conn.close()

    if len(df) < 100:
        print(f"[!] Pas assez de ticks ({len(df)}). Lance d'abord `python run.py --collect`.")
        sys.exit(1)

    print(f"[+] Entraînement sur {len(df):,} ticks...")
    feats = build_features(df)
    result = train(feats)
    print(f"\n[✓] Métriques: {result.metrics}")
    print(f"\nTop features:\n{result.feature_importance.head(10)}")

    save_model(result.model, "data/model_lgbm.pkl")
    print("\n[✓] Modèle sauvegardé : data/model_lgbm.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", help="Lance aussi le collector")
    parser.add_argument("--train", action="store_true", help="Entraîne le modèle LGBM")
    args = parser.parse_args()

    if args.train:
        train_model()
        return

    procs = []
    if args.collect:
        p = mp.Process(target=start_collector, daemon=True)
        p.start()
        procs.append(p)
        print(f"[+] Collector PID {p.pid} démarré")

    print(f"[+] Dashboard : http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    try:
        start_dashboard()
    except KeyboardInterrupt:
        print("\n[+] Arrêt...")
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
