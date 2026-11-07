"""M1 check: auth, instrument key, fetch + cache, bar-open test, limits probes, summary."""
import argparse
import base64
import json
import time
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd

import upstox_data as ud


def token_info():
    try:
        p = ud._token().split(".")[1]
        p += "=" * (-len(p) % 4)
        d = json.loads(base64.urlsafe_b64decode(p))
        f = lambda k: datetime.fromtimestamp(d[k], ud.IST).strftime("%Y-%m-%d") if k in d else "n/a"
        return f"issued {f('iat')}, expires {f('exp')} (best-effort JWT decode)"
    except Exception as e:
        return f"token is not a decodable JWT ({type(e).__name__}); trust the docs: 1-year validity"


def probe(label, path):
    t = time.time()
    r = ud._get(path)
    dt = time.time() - t
    if r.status_code == 200:
        c = r.json().get("data", {}).get("candles", [])
        ts = [x[0] for x in c]
        span = f"{min(ts)} .. {max(ts)}" if ts else "no candles"
        msg = f"HTTP 200, {len(c)} candles, {span}"
    else:
        msg = f"HTTP {r.status_code}: {r.text[:200]}"
    line = f"[{label}] {dt:.2f}s  {msg}"
    print(line)
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=25, help="calendar days back from yesterday")
    ap.add_argument("--yahoo", help="optional path to a Yahoo CSV to inspect")
    a = ap.parse_args()
    sym = "NIFTY50"

    print("== token ==")
    print(token_info())

    print("\n== instrument key ==")
    key = ud.resolve_keys()[sym]
    print(sym, "->", key, "(full NSE index list saved in data/nse_index_list.txt)")

    end = datetime.now(ud.IST).date() - timedelta(days=1)
    start = end - timedelta(days=a.days - 1)
    print(f"\n== fetch {start} .. {end} ==")
    t = time.time()
    n1 = ud.ensure_days(sym, key, start, end)
    t1 = time.time() - t
    n2 = ud.ensure_days(sym, key, start, end)
    print(f"first pass: {n1} day files written in {t1:.1f}s; second pass: {n2} written (must be 0 = cache works)")
    print("days with no data (holidays etc):", ud.nodata_days(sym))

    print("\n== per-day structure ==")
    rows, d = [], start
    while d <= end:
        df = ud.load_day(sym, d)
        if df is not None:
            t_ = df["ts_open"].dt.strftime("%H:%M")
            exp = set(pd.date_range(f"{d} 09:15", f"{d} 15:29", freq="min").strftime("%H:%M"))
            rows.append((d, len(df), t_.iloc[0], t_.iloc[-1],
                         len(exp - set(t_)), len(set(t_) - exp), float(df["volume"].sum())))
        d += timedelta(days=1)
    print("date        bars first last  missing extra volume_sum")
    for r in rows[-8:]:
        print(f"{r[0]}  {r[1]:4d}  {r[2]} {r[3]}  {r[4]:5d} {r[5]:5d}  {r[6]:.0f}")
    pairs = Counter((r[2], r[3]) for r in rows)
    top = pairs.most_common(1)[0][0] if pairs else None
    if top == ("09:15", "15:29"):
        verdict = "BAR-OPEN stamps confirmed (first 09:15, last 15:29): bar at 09:15 closes at 09:16"
    elif top == ("09:16", "15:30"):
        verdict = "BAR-CLOSE stamps (first 09:16, last 15:30)"
    else:
        verdict = f"INCONCLUSIVE, most common (first,last) = {top}"
    print("\nconvention:", verdict)
    tot_missing = sum(r[4] for r in rows)
    print(f"days cached: {len(rows)}; total missing minutes: {tot_missing}; total volume: {sum(r[6] for r in rows):.0f}")

    print("\n== limits probes (read-only) ==")
    q = lambda a_, b_: ud.range_path(key, a_, b_)
    from datetime import date
    out = []
    out.append(probe("A  2022-01-03..07 (docs: data starts Jan 2022)", q(date(2022, 1, 3), date(2022, 1, 7))))
    out.append(probe("B  2021-12-01..10 (before documented start)", q(date(2021, 12, 1), date(2021, 12, 10))))
    out.append(probe("C  45-day window (docs: 1 month max -> expect error)", q(end - timedelta(days=44), end)))
    out.append(probe("D  28-day window (what the cache uses)", q(end - timedelta(days=27), end)))
    out.append(probe("E  intraday endpoint (today is a non-trading day if weekend)", ud.intraday_path(key)))

    if a.yahoo:
        print("\n== yahoo csv ==")
        import yahoo_loader
        yahoo_loader.describe(yahoo_loader.load_yahoo_csv(a.yahoo))

    print("\n==================== RESULT SUMMARY (M1) ====================")
    print("token:", token_info())
    print("key:", sym, key)
    print(f"cached days: {len(rows)}, rerun fetched {n2} (cache ok if 0), missing minutes {tot_missing}")
    print("convention:", verdict)
    for l in out:
        print(l)
    print("==============================================================")


if __name__ == "__main__":
    main()