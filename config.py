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
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", 0.01))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", 0.25))

# Garde-fous trading/paper trading
MIN_TRADE_EDGE = float(os.getenv("MIN_TRADE_EDGE", 0.08))
MIN_MODEL_AUC = float(os.getenv("MIN_MODEL_AUC", 0.54))
MAX_MODEL_BRIER = float(os.getenv("MAX_MODEL_BRIER", 0.245))
MAX_MODEL_LOGLOSS = float(os.getenv("MAX_MODEL_LOGLOSS", 0.72))
MIN_MODEL_TEST_ROWS = int(os.getenv("MIN_MODEL_TEST_ROWS", 100))
MIN_TRAIN_LABELS = int(os.getenv("MIN_TRAIN_LABELS", 300))
MIN_SECONDS_TO_CLOSE = int(os.getenv("MIN_SECONDS_TO_CLOSE", 45))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", 1))
ALLOW_BASELINE_TRADES = os.getenv("ALLOW_BASELINE_TRADES", "0").lower() in {"1", "true", "yes"}

# Frais Polymarket (indicatifs)
POLYMARKET_FEE = 0.0              # 0% mais spread ~1-2%
ASSUMED_SPREAD = 0.015            # 1.5% de slippage/spread moyen

# Dashboard
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", 8000))

# DB
DB_PATH = DATA_DIR / "bot.db"
DB_URL = f"sqlite:///{DB_PATH}"
