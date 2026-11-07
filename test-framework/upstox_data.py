"""Upstox market-data client + local cache.
Read-only market-data endpoints only (/historical-candle/...). No order endpoints."""
import gzip
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
import os

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

IST = timezone(timedelta(hours=5, minutes=30))
BASE = "https://api.upstox.com/v3"
INSTR_URL = "https://assets.upstox.com/market-quote/instruments/exchange/{ex}.json.gz"   # docs: NSE / BSE / MCX / complete
DATA = ROOT / "data"
CACHE = DATA / "cache"
MAX_CHUNK_DAYS = 28      # 28 days worked and 45 failed in the M1 probes; 28 = safe margin
MIN_GAP = 0.06           # seconds between calls (~16/s, docs allow 50/s)

# exchange = which instrument file to read; match = lowercase name or trading_symbol (exact);
# hint = word used to list nearby entries if the match is not exactly one
SYMBOLS = {
    "NIFTY50":   {"exchange": "NSE", "match": "nifty 50",         "hint": "nifty 50"},
    "BANKNIFTY": {"exchange": "NSE", "match": "nifty bank",       "hint": "bank"},
    "MIDCAP100": {"exchange": "NSE", "match": "nifty midcap 100", "hint": "midcap"},
    "SENSEX":    {"exchange": "BSE", "match": "sensex",           "hint": "sensex"},
}

COLS = ["ts_open", "ts_close", "open", "high", "low", "close", "volume", "oi"]
_last_call = [0.0]


def _token():
    t = os.getenv("UPSTOX_ANALYTICS_TOKEN", "").strip()
    if not t or t.startswith("paste_"):
        raise SystemExit("UPSTOX_ANALYTICS_TOKEN is missing in .env")
    return t


def _get(path, retries=4):
    """Throttled GET. Hard guard: market-data endpoints only."""
    assert path.startswith("/historical-candle/"), "market-data endpoints only"
    url = BASE + path
    r = None
    for attempt in range(retries):
        wait = MIN_GAP - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        r = requests.get(
            url,
            headers={"Accept": "application/json", "Authorization": "Bearer " + _token()},
            timeout=60,
        )
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        return r
    return r


# ---------- instruments ----------
def _download_instruments(ex):
    r = requests.get(INSTR_URL.format(ex=ex), timeout=180)
    r.raise_for_status()
    try:
        raw = gzip.decompress(r.content)
    except OSError:          # server may already have decompressed it
        raw = r.content
    return json.loads(raw)


def resolve_keys(refresh=False):
    """Instrument keys per symbol. Cached in index_keys.json; only missing symbols are looked up."""
    f = DATA / "index_keys.json"
    out = json.loads(f.read_text()) if (f.exists() and not refresh) else {}
    missing = [s for s in SYMBOLS if s not in out]
    if not missing:
        return out
    DATA.mkdir(exist_ok=True)
    idx_by_ex, lines_by_ex = {}, {}
    for ex in sorted({SYMBOLS[s]["exchange"] for s in missing}):
        items = _download_instruments(ex)
        idx = [i for i in items if i.get("segment") == f"{ex}_INDEX"]
        lines = [f'{i.get("instrument_key")}\t{i.get("name")}\t{i.get("trading_symbol")}' for i in idx]
        (DATA / f"{ex.lower()}_index_list.txt").write_text("\n".join(sorted(lines)), encoding="utf-8")
        idx_by_ex[ex], lines_by_ex[ex] = idx, lines
    for sym in missing:
        cfg = SYMBOLS[sym]
        ex = cfg["exchange"]
        m = [i for i in idx_by_ex[ex]
             if cfg["match"] in (str(i.get("name", "")).lower(), str(i.get("trading_symbol", "")).lower())]
        keys = sorted({i["instrument_key"] for i in m})
        if len(keys) != 1:
            near = [l for l in lines_by_ex[ex] if cfg["hint"] in l.lower()][:30]
            raise SystemExit(f"{sym}: expected exactly 1 match in {ex} index list, got {keys}. "
                             f"Nearby entries:\n" + "\n".join(near))
        out[sym] = keys[0]
    f.write_text(json.dumps(out, indent=2))
    return out


# ---------- candles ----------
def _to_df(candles):
    if not candles:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame([c[:7] for c in candles],
                      columns=["ts", "open", "high", "low", "close", "volume", "oi"])
    df["ts_open"] = pd.to_datetime(df.pop("ts"), utc=True).dt.tz_convert(IST)
    df = df.sort_values("ts_open").drop_duplicates("ts_open").reset_index(drop=True)
    # Docs: candle[0] is the START time of the candle -> bar-open stamp (re-verified per symbol by m5a_check.py).
    df["ts_close"] = df["ts_open"] + pd.Timedelta(minutes=1)
    return df[COLS]


def range_path(key, d_from, d_to):
    return f"/historical-candle/{quote(key, safe='')}/minutes/1/{d_to:%Y-%m-%d}/{d_from:%Y-%m-%d}"


def fetch_range(key, d_from, d_to):
    r = _get(range_path(key, d_from, d_to))
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return _to_df(r.json()["data"]["candles"])


def intraday_path(key):
    return f"/historical-candle/intraday/{quote(key, safe='')}/minutes/1"


def fetch_intraday(key):
    r = _get(intraday_path(key))
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return _to_df(r.json()["data"]["candles"])


# ---------- cache: one parquet per symbol per past day ----------
def _day_file(sym, d):
    return CACHE / sym / f"{d:%Y-%m-%d}.parquet"


def _nodata_file(sym):
    return CACHE / sym / "_no_data.json"


def _read_nodata(sym):
    f = _nodata_file(sym)
    return set(json.loads(f.read_text())) if f.exists() else set()


def _write_nodata(sym, s):
    _nodata_file(sym).write_text(json.dumps(sorted(s)))


def nodata_days(sym):
    return sorted(_read_nodata(sym))


def _save_day(sym, d, g):
    g = g.copy()
    g["ts_open"] = g["ts_open"].dt.tz_convert("UTC")
    g["ts_close"] = g["ts_close"].dt.tz_convert("UTC")
    g.to_parquet(_day_file(sym, d), index=False)


def load_day(sym, d):
    f = _day_file(sym, d)
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    df["ts_open"] = pd.to_datetime(df["ts_open"], utc=True).dt.tz_convert(IST)
    df["ts_close"] = pd.to_datetime(df["ts_close"], utc=True).dt.tz_convert(IST)
    return df


def ensure_days(sym, key, start, end):
    """Fetch only weekdays before today that are not cached yet. Returns number of day files written."""
    today = datetime.now(IST).date()
    (CACHE / sym).mkdir(parents=True, exist_ok=True)
    nd = _read_nodata(sym)
    need, d = [], start
    while d <= end:
        if d.weekday() < 5 and d < today and not _day_file(sym, d).exists() and d.isoformat() not in nd:
            need.append(d)
        d += timedelta(days=1)
    written, i = 0, 0
    while i < len(need):
        c0, j = need[i], i
        while j + 1 < len(need) and (need[j + 1] - c0).days <= MAX_CHUNK_DAYS - 1:
            j += 1
        chunk = need[i:j + 1]
        df = fetch_range(key, c0, need[j])
        by_day = {k: g for k, g in df.groupby(df["ts_open"].dt.date)} if len(df) else {}
        for d in chunk:
            g = by_day.get(d)
            if g is None or g.empty:
                nd.add(d.isoformat())          # holiday / no data
            else:
                _save_day(sym, d, g)
                written += 1
        _write_nodata(sym, nd)
        i = j + 1
    return written


def get_range(sym, start, end):
    """All cached 1-minute bars for past days in [start, end] (today is never cached)."""
    key = resolve_keys()[sym]
    ensure_days(sym, key, start, end)
    frames, d = [], start
    while d <= end:
        df = load_day(sym, d)
        if df is not None:
            frames.append(df)
        d += timedelta(days=1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)