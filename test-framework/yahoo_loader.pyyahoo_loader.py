"""Optional fallback: load your existing Yahoo 1-minute CSV files into the same schema.
CSV columns: timestamp (ISO +05:30), open, high, low, close, volume, symbol, source."""
import sys
from collections import Counter

import pandas as pd

from upstox_data import COLS, IST


def load_yahoo_csv(path, symbol=None, bar_stamp="open"):
    """bar_stamp='open' if the CSV timestamp is the bar start (assumed, UNVERIFIED), 'close' otherwise."""
    df = pd.read_csv(path)
    if symbol and "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
    out = pd.DataFrame({
        "ts_open": ts if bar_stamp == "open" else ts - pd.Timedelta(minutes=1),
        "open": df["open"].astype(float), "high": df["high"].astype(float),
        "low": df["low"].astype(float), "close": df["close"].astype(float),
        "volume": df["volume"].astype(float), "oi": 0.0,
    })
    out["ts_close"] = out["ts_open"] + pd.Timedelta(minutes=1)
    out = out.sort_values("ts_open").drop_duplicates("ts_open").reset_index(drop=True)
    return out[COLS]


def describe(df):
    days = df.groupby(df["ts_open"].dt.date)
    pairs = Counter((g["ts_open"].iloc[0].strftime("%H:%M"), g["ts_open"].iloc[-1].strftime("%H:%M"))
                    for _, g in days)
    print(f"rows={len(df)} days={days.ngroups} range={df['ts_open'].min()} -> {df['ts_open'].max()}")
    print("(first,last) bar-stamp pairs per day:", pairs.most_common(3))
    print("total volume:", df["volume"].sum())


if __name__ == "__main__":
    p = sys.argv[1]
    s = sys.argv[2] if len(sys.argv) > 2 else None
    describe(load_yahoo_csv(p, s))