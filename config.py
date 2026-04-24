"""Configuration globale du bot Polymarket."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Polymarket endpoints
POLYMARKET_CLOB_API = "https://clob.polymarket.com"
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"
POLYMARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Binance endpoints
BINANCE_REST_API = "https://api.binance.com"
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"

# Simulateur
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", 1000))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", 0.02))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", 0.25))

# Frais Polymarket (indicatifs)
POLYMARKET_FEE = 0.0              # 0% mais spread ~1-2%
ASSUMED_SPREAD = 0.015            # 1.5% de slippage/spread moyen

# Dashboard
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", 8000))

# DB
DB_PATH = DATA_DIR / "bot.db"
DB_URL = f"sqlite:///{DB_PATH}"
