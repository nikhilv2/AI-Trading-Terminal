"""No-look-ahead tests: the forecast at T must not depend on any bar that closes after T."""
import argparse
from datetime import date

import numpy as np
import pandas as pd

import engine as eg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NIFTY50")
    ap.add_argument("--date")
    ap.add_argument("--time", default="11:00")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--skip-model", action="store_true")
    a = ap.parse_args()
    day = date.fromisoformat(a.date) if a.date else eg.latest_cached_day(a.symbol)
    T = eg.make_ts(day, a.time)
    settings = {**eg.DEFAULT_SETTINGS, "context_len": a.ctx}
    bars = eg.load_history(a.symbol, day, a.ctx)
    rp = eg.Replay(bars)
    results = []

    def check(name, ok, detail=""):
        print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
        results.append(bool(ok))

    # T1: bar-open convention handled explicitly
    ctx = rp.context(T, a.ctx)
    last = ctx.iloc[-1]
    check("T1 last context bar closed at or before T", last["ts_close"] <= T,
          f"last open {last['ts_open']:%H:%M}, close {last['ts_close']:%H:%M}, T {T:%H:%M}")
    check("T1 last context bar opened exactly 1 minute before T", last["ts_open"] == T - eg.MIN)
    check("T1 no context bar opens at or after T", int((ctx["ts_open"] >= T).sum()) == 0)
    future = bars[bars["ts_open"] >= T]
    check("T1 future bars exist in the data but are excluded", len(future) > 0, f"{len(future)} future bars")

    # T2: poisoned / truncated future leaves the context identical
    poisoned = bars.copy()
    m = poisoned["ts_close"] > T
    for c in ["open", "high", "low", "close"]:
        poisoned.loc[m, c] = 1e9
    trunc = bars[bars["ts_close"] <= T].copy()
    rp_p, rp_t = eg.Replay(poisoned), eg.Replay(trunc)
    base = ctx["close"].to_numpy()
    check("T2 poison is armed (future rows changed)", int(m.sum()) > 0, f"{int(m.sum())} rows poisoned")
    check("T2 context identical with poisoned future",
          np.array_equal(base, rp_p.context(T, a.ctx)["close"].to_numpy()))
    check("T2 context identical with all later data removed",
          np.array_equal(base, rp_t.context(T, a.ctx)["close"].to_numpy()))

    if not a.skip_model:
        runner = eg.ModelRunner()
        H = a.horizon
        q1, _, _ = eg.forecast_at(runner, rp, T, settings, H)
        q2, _, _ = eg.forecast_at(runner, rp, T, settings, H)
        noise = float(np.abs(q1 - q2).max())
        tol = max(10 * noise, 1e-3)
        print(f"      GPU repeat-run noise floor: {noise:.3e}, tolerance {tol:.1e}")
        qt, _, _ = eg.forecast_at(runner, rp_t, T, settings, H)
        qp, _, _ = eg.forecast_at(runner, rp_p, T, settings, H)
        dt, dp = float(np.abs(q1 - qt).max()), float(np.abs(q1 - qp).max())
        check("T3 forecast identical when all later data is removed", dt <= tol,
              f"max|diff|={dt:.3e}, bitwise={np.array_equal(q1, qt)}")
        check("T3 forecast identical when later data is poisoned", dp <= tol,
              f"max|diff|={dp:.3e}, bitwise={np.array_equal(q1, qp)}")
        # T4: negative control. Deliberately leak 5 future bars (x1.02) into the context.
        fut = bars[bars["ts_open"] >= T].head(5)["close"].to_numpy() * 1.02
        leaked = np.concatenate([base, fut])
        ql = runner.forecast(leaked, H)
        dl = float(np.abs(q1 - ql).max())
        check("T4 negative control: a deliberate leak IS detected by this test", dl > tol, f"max|diff|={dl:.3f}")

    print("\n==================== RESULT SUMMARY (M2 test) ====================")
    print(f"{sum(results)}/{len(results)} checks passed  (symbol {a.symbol}, T={T}, ctx={a.ctx})")
    print("ALL PASS" if all(results) else "SOME FAILED")
    print("==================================================================")


if __name__ == "__main__":
    main()