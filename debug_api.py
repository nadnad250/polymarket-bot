"""Debug Polymarket API — affiche ce que retourne Gamma."""
import json
import httpx

BASE = "https://gamma-api.polymarket.com"
client = httpx.Client(timeout=15.0, headers={"User-Agent": "polymarket-bot-debug/0.1"})


def test(label, url, params=None):
    print(f"\n{'='*60}\n{label}\nURL: {url}")
    if params:
        print(f"Params: {params}")
    try:
        r = client.get(url, params=params)
        print(f"Status: {r.status_code}")
        data = r.json()
        if isinstance(data, list):
            print(f"Type: list[{len(data)}]")
            for item in data[:3]:
                print(f"  - title={item.get('title', '?')!r:60.60} slug={item.get('slug', '?')!r}")
            if data:
                print(f"\n  Premier item (clés): {sorted(data[0].keys())}")
        else:
            print(f"Type: dict, keys={sorted(data.keys()) if isinstance(data, dict) else '?'}")
            print(f"Preview: {json.dumps(data, default=str)[:500]}")
    except Exception as e:
        print(f"ERREUR: {e}")


# Test 1 : events actifs
test("1. Events actifs (sans filtre)", f"{BASE}/events",
     {"limit": 20, "active": "true", "closed": "false"})

# Test 2 : events par slug exact
test("2. Event par slug exact",
     f"{BASE}/events",
     {"slug": "btc-updown-5m-1777061400"})

# Test 3 : recherche BTC
test("3. Events filtrés BTC",
     f"{BASE}/events",
     {"limit": 30, "active": "true", "closed": "false", "tag": "crypto"})

# Test 4 : markets plats
test("4. Markets (endpoint plat)",
     f"{BASE}/markets",
     {"limit": 20, "active": "true", "closed": "false"})

# Test 5 : avec tag_id
test("5. Events tag_slug=crypto",
     f"{BASE}/events",
     {"limit": 30, "active": "true", "closed": "false", "tag_slug": "crypto"})

client.close()
