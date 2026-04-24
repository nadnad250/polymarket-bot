"""Collecteur principal — aligne prix Polymarket + Binance toutes les N secondes.

Stocke dans SQLite. Gère le renouvellement auto des marchés BTC Up/Down 5min.

Usage:
    python -m src.fetchers.collector
"""
from __future__ import annotations

import signal
import sqlite3
import time
from contextlib import closing
from datetime import datetime

from config import DB_PATH
from src.fetchers.btc import BTCFetcher
from src.fetchers.polymarket import PolymarketClient

POLL_INTERVAL_SEC = 5
EVENT_REFRESH_SEC = 60   # re-chercher un event actif toutes les 60s


def init_db(path: str = str(DB_PATH)) -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ticks (
                ts           INTEGER PRIMARY KEY,
                btc_price    REAL NOT NULL,
                btc_bid      REAL,
                btc_ask      REAL,
                btc_ob_imb   REAL,
                poly_market  TEXT,
                poly_yes     REAL,
                poly_no      REAL,
                poly_volume  REAL,
                poly_question TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ticks_market ON ticks(poly_market);
            """
        )
        conn.commit()


def insert_tick(conn: sqlite3.Connection, row: tuple) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO ticks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        row,
    )
    conn.commit()


def run(poll_sec: int = POLL_INTERVAL_SEC) -> None:
    init_db()
    binance = BTCFetcher()
    poly = PolymarketClient()
    stop = {"flag": False}

    def handler(signum, frame):
        stop["flag"] = True
        print("\n[collector] arrêt demandé...")

    signal.signal(signal.SIGINT, handler)

    current_event = None
    last_event_refresh = 0.0

    try:
        with closing(sqlite3.connect(str(DB_PATH))) as conn:
            while not stop["flag"]:
                now = time.time()

                # Re-cherche l'event actif si expiré ou jamais trouvé
                if current_event is None or (now - last_event_refresh) > EVENT_REFRESH_SEC:
                    try:
                        new_event = poly.find_btc_updown_event()
                        if new_event:
                            if current_event is None or new_event.get("slug") != current_event.get("slug"):
                                print(f"[collector] event actif: {new_event.get('title')}")
                                print(f"            slug={new_event.get('slug')}")
                            current_event = new_event
                            last_event_refresh = now
                    except Exception as e:
                        print(f"[collector] recherche event échouée: {e}")

                if current_event is None:
                    print("[collector] aucun event BTC, retry dans 10s...")
                    time.sleep(10)
                    continue

                try:
                    btc = binance.get_book_ticker()
                    imb = binance.orderbook_imbalance(levels=10)
                    snap = poly.snapshot_event(current_event)
                    if snap is None:
                        print("[collector] snapshot Polymarket indisponible, skip.")
                        time.sleep(poll_sec)
                        continue

                    row = (
                        int(time.time() * 1000),
                        btc.price,
                        btc.bid,
                        btc.ask,
                        imb,
                        snap.event_slug,
                        snap.yes_price,
                        snap.no_price,
                        snap.volume_24h,
                        snap.question[:100],
                    )
                    insert_tick(conn, row)
                    print(
                        f"[{datetime.now():%H:%M:%S}] "
                        f"BTC ${btc.price:,.2f} imb={imb:+.2f} | "
                        f"YES={snap.yes_price:.3f} NO={snap.no_price:.3f} "
                        f"| {snap.event_slug[:40]}"
                    )
                except Exception as e:
                    print(f"[collector] erreur tick: {e}")
                time.sleep(poll_sec)
    finally:
        binance.close()
        poly.close()


if __name__ == "__main__":
    run()
