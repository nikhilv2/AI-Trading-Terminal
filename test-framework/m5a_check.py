"""M5a check: resolve keys for all four benchmarks, fetch + cache recent days, verify structure per symbol."""
import argparse
from collections import Counter
from datetime import date, datetime, timedelta

import pandas as pd

import upstox_data as ud


def probe_2022(key):
    r = ud._get(ud.range_path(key, date(2022, 1, 3), date(2022, 1, 7)))
    if r.status_code != 200:
        return f"HTTP {r.status_code}: {r.text[:150]}"
    c = r.json().get("data", {}).get("candles", [])
    ts = [x[0] for x in c]
    return f"{len(c)} candles, {min(ts)} .. {max(ts)}" if ts else "0 candles (no data that early)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=25, help="calendar days back from yesterday")
    a = ap.parse_args()
    end = datetime.now(ud.IST).date() - timedelta(days=1)
    start = end - timedelta(days=a.days - 1)

    print("== keys ==")
    keys = ud.resolve_keys()
    for s in ud.SYMBOLS:
        print(f"{s:10s} -> {keys[s]}")

    summary = []
    for sym in ud.SYMBOLS:
        key = keys[sym]
        print(f"\n== {sym} {start}..{end} ==")
        n1 = ud.ensure_days(sym, key, start, end)
        n2 = ud.ensure_days(sym, key, start, end)
        print(f"first pass wrote {n1} day files; second pass wrote {n2} (must be 0)")
        print("no-data days:", ud.nodata_days(sym))
        rows, d = [], start
        last_close = None
        while d <= end:
            df = ud.load_day(sym, d)
            if df is not None:
                t_ = df["ts_open"].dt.strftime("%H:%M")
                exp = set(pd.date_range(f"{d} 09:15", f"{d} 15:29", freq="min").strftime("%H:%M"))
                rows.append((d, len(df), t_.iloc[0], t_.iloc[-1], len(exp - set(t_)), len(set(t_) - exp),
                             float(df["volume"].sum())))
                last_close = float(df["close"].iloc[-1])
            d += timedelta(days=1)
        for r in rows[-3:]:
            print(f"  {r[0]}  bars {r[1]:4d}  first {r[2]} last {r[3]}  missing {r[4]} extra {r[5]}  volume {r[6]:.0f}")
        pairs = Counter((r[2], r[3]) for r in rows)
        top = pairs.most_common(1)[0][0] if pairs else None
        conv = ("BAR-OPEN (09:15..15:29)" if top == ("09:15", "15:29")
                else "BAR-CLOSE (09:16..15:30)" if top == ("09:16", "15:30") else f"INCONCLUSIVE {top}")
        miss = sum(r[4] for r in rows)
        extra = sum(r[5] for r in rows)
        vol = sum(r[6] for r in rows)
        p = probe_2022(key)
        print("  convention:", conv)
        print("  2022-01-03..07 probe:", p)
        summary.append((sym, len(rows), n2, miss, extra, vol, conv, last_close, p))

    print("\n==================== RESULT SUMMARY (M5a) ====================")
    for sym, nd, n2, miss, extra, vol, conv, lc, p in summary:
        print(f"{sym:10s} key={keys[sym]} | days {nd} | rerun fetched {n2} | missing min {miss} | "
              f"extra min {extra} | volume {vol:.0f} | {conv} | last close {lc}")
        print(f"{'':10s} 2022 probe: {p}")
    print("==============================================================")


if __name__ == "__main__":
    main()