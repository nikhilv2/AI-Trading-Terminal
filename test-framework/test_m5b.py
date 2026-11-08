"""M5b tests: input forms, time-of-day / volume / range covariates, volatility-band baseline, no-look-ahead for every switch."""
import argparse
from datetime import date

import numpy as np
import pandas as pd

import engine as eg
import metrics as mt

results = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    results.append(bool(ok))


def near(a, b, tol=1e-9):
    return a is not None and abs(a - b) <= tol


def full_day(d, closes=None):
    t0 = eg.make_ts(d, "09:15")
    opens = [t0 + i * eg.MIN for i in range(375)]
    c = np.array(closes if closes is not None else [100.0 * 1.001 ** i for i in range(375)])
    return pd.DataFrame({"ts_open": opens, "ts_close": [o + eg.MIN for o in opens],
                         "open": c, "high": c * 1.001, "low": c * 0.999, "close": c,
                         "volume": np.random.randint(1000, 5000, size=375).astype(float), "oi": 0.0})


FORMS = ("raw_close", "log_price_stitched", "log_returns")
COVS = ("none", "time_of_day", "volume", "range", "tod_volume")


def pure_tests():
    # --- stitched returns: the overnight gap (102 -> 110) must not appear
    d1, d2 = date(2026, 11, 5), date(2026, 11, 6)
    rows = [(d1, "09:15", 100, 101), (d1, "09:16", 101, 102), (d1, "09:17", 102, 103),
            (d2, "09:15", 110, 111), (d2, "09:16", 111, 112)]
    df = pd.DataFrame({"ts_open": [eg.make_ts(d, h) for d, h, _, _ in rows],
                       "open": [float(r[2]) for r in rows], "close": [float(r[3]) for r in rows]})
    df["ts_close"] = df["ts_open"] + eg.MIN
    df["high"] = df["close"] * 1.001
    df["low"] = df["close"] * 0.999
    df["volume"] = 1000.0
    r = eg._session_returns(df)
    exp = np.log([101 / 100, 102 / 101, 103 / 102, 111 / 110, 112 / 111])
    check("S1 session returns exclude the overnight gap", np.allclose(r, exp, atol=1e-12))
    T = eg.make_ts(d2, "09:17")
    st = {**eg.DEFAULT_SETTINGS, "input_form": "log_price_stitched"}
    x, cov, back = eg.prepare_input(df, T, st, 2)
    check("S2 stitched log price: last value is 0", near(x[-1], 0.0))
    check("S3 stitched back-conversion reproduces the previous bar (111)", near(back(x)[-2], 111.0, 1e-8))
    check("S4 stitched path closes the gap (day-1 end maps to day-2 open 110)", near(back(x)[2], 110.0, 1e-8))
    st = {**eg.DEFAULT_SETTINGS, "input_form": "log_returns"}
    x, cov, back = eg.prepare_input(df, T, st, 3)
    check("S5 log returns input is 1e4 x returns", np.allclose(x, 1e4 * exp, atol=1e-8))
    q = np.full((3, 9), 10.0)
    check("S6 log returns back: 10 bps/step -> last*exp(0.003) at step 3", near(back(q)[2, 4], 112.0 * np.exp(0.003), 1e-8))
    check("S7 log returns back: zero returns -> flat at last close", np.allclose(back(np.zeros((3, 9))), 112.0))
    st = {**eg.DEFAULT_SETTINGS, "input_form": "raw_close", "covariates": "none"}   # DEFAULT_SETTINGS now has time_of_day
    x, cov, back = eg.prepare_input(df, T, st, 2)
    check("S8 raw close input unchanged, no covariate", np.array_equal(x, df["close"].to_numpy()) and cov is None)

    # --- settings validation
    bad = 0
    for b in ({"input_form": "x"}, {"covariates": "y"}):
        try:
            eg.check_settings({**eg.DEFAULT_SETTINGS, **b})
        except ValueError:
            bad += 1
    check("S9 invalid settings are rejected", bad == 2)

    # --- time-of-day covariate
    day = date(2026, 11, 6)
    rp = eg.Replay(full_day(day))
    T = eg.make_ts(day, "10:00")
    ctx = rp.context(T, 5)
    c = eg._tod_cov(ctx, T, 3)
    check("C1 covariate shape (1, ctx+H) = (1, 8)", c.shape == (1, 8), str(c.shape))
    check("C2 values: 09:55 -> 40/374, last context 09:59 -> 44/374, horizon end 10:02 -> 47/374",
          near(c[0, 0], 40 / 374, 1e-6) and near(c[0, 4], 44 / 374, 1e-6) and near(c[0, 7], 47 / 374, 1e-6))
    last_ctx = rp.context(eg.make_ts(day, "15:30"), 1)
    check("C3 15:29 bar -> 1.0", near(eg._tod_cov(last_ctx, eg.make_ts(day, "15:30"), 1)[0, 0], 1.0, 1e-6))

    # --- volume & range covariates
    vol = eg._volume_cov(ctx, T, 3)
    check("C4 volume covariate shape (1, 8)", vol.shape == (1, 8))
    rng = eg._range_cov(ctx, T, 3)
    check("C5 range covariate shape (1, 8)", rng.shape == (1, 8))
    tod_vol = np.concatenate([eg._tod_cov(ctx, T, 3), eg._volume_cov(ctx, T, 3)], axis=0)
    check("C6 tod_volume shape (2, 8)", tod_vol.shape == (2, 8))

    # --- inputs identical when later data is removed / poisoned
    df = full_day(day, [100 + np.sin(i / 7.0) * 3 + i * 0.01 for i in range(375)])
    rp = eg.Replay(df)
    poisoned = df.copy()
    m = poisoned["ts_close"] > T
    for col in ("open", "high", "low", "close", "volume"):
        poisoned.loc[m, col] = 1e9
    rp_p, rp_t = eg.Replay(poisoned), eg.Replay(df[df["ts_close"] <= T].copy())
    ok = True
    for form in FORMS:
        for cv in COVS:
            st = {**eg.DEFAULT_SETTINGS, "input_form": form, "covariates": cv, "context_len": 200}
            xa, ca, _ = eg.prepare_input(rp.context(T, 200), T, st, 10)
            for other in (rp_p, rp_t):
                xb, cb, _ = eg.prepare_input(other.context(T, 200), T, st, 10)
                ok &= np.array_equal(xa, xb) and ((ca is None and cb is None) or np.array_equal(ca, cb))
    check("N1 input and covariate identical with poisoned / removed future (all settings)", ok)

    # --- volatility band
    rp = eg.Replay(full_day(day))
    T = eg.make_ts(day, "10:00")
    anchor, q = mt.vol_band(rp, T, 4)
    sigma = np.log(1.001)
    exp_anchor = 100.0 * 1.001 ** 44
    check("V1 anchor = last closed close (09:59 bar)", near(anchor, exp_anchor, 1e-9))
    check("V2 median equals anchor at every step", np.allclose(q[:, 4], exp_anchor, atol=1e-9))
    check("V3 step-4 P90 = anchor*(1+sigma*2*z90)", near(q[3, 8], exp_anchor * (1 + sigma * 2 * mt.Z[8]), 1e-7))
    check("V4 band symmetric around the median", near(q[0, 0] + q[0, 8], 2 * exp_anchor, 1e-7))
    check("V5 too few returns -> unavailable (None)", mt.vol_band(rp, eg.make_ts(day, "09:20"), 3)[1] is None)


def model_tests(a):
    sym = a.symbol
    day = date.fromisoformat(a.date) if a.date else eg.latest_cached_day(sym)
    T = eg.make_ts(day, a.time)
    H = a.horizon
    bars = eg.load_history(sym, day, 2048)
    rp = eg.Replay(bars)
    poisoned = bars.copy()
    m = poisoned["ts_close"] > T
    for col in ("open", "high", "low", "close", "volume"):
        poisoned.loc[m, col] = 1e9
    rp_p, rp_t = eg.Replay(poisoned), eg.Replay(bars[bars["ts_close"] <= T].copy())
    runner = eg.ModelRunner()
    qs = {}
    combos = [(f, c) for f in FORMS for c in COVS]
    for form, cv in combos:
        st = {**eg.DEFAULT_SETTINGS, "input_form": form, "covariates": cv}
        q, ctx, ms = eg.forecast_at(runner, rp, T, st, H)
        qt, _, _ = eg.forecast_at(runner, rp_t, T, st, H)
        qp, _, _ = eg.forecast_at(runner, rp_p, T, st, H)
        last = float(ctx["close"].iloc[-1])
        tag = f"{form}/{cv}"
        check(f"M {tag}: shape (H,9), finite, quantiles non-decreasing",
              q.shape == (H, 9) and bool(np.isfinite(q).all()) and bool((np.diff(q, axis=1) >= -1e-9).all()))
        dt, dp = float(np.abs(q - qt).max()), float(np.abs(q - qp).max())
        check(f"M {tag}: identical when all later data is removed", dt <= 1e-3, f"max|diff|={dt:.2e}")
        check(f"M {tag}: identical when later data is poisoned", dp <= 1e-3, f"max|diff|={dp:.2e}")
        dev = abs(q[0, 4] - last) / last
        check(f"M {tag}: step-1 median within 1% of last close", dev < 0.01, f"last {last:.2f}, q50[0] {q[0, 4]:.2f}")
        print(f"      q50 step1 {q[0, 4]:.2f} step{H} {q[-1, 4]:.2f} | P10-P90 width step{H}: {q[-1, 8] - q[-1, 0]:.2f} | {ms:.0f} ms")
        qs[(form, cv)] = q

    # Check that covariates actually change the forecast
    for form in FORMS:
        base = qs[(form, "none")]
        for cv in ("time_of_day", "volume", "range", "tod_volume"):
            d = float(np.abs(base - qs[(form, cv)]).max())
            check(f"M {form}/{cv}: covariate changes the forecast", d > 1e-6, f"max|diff|={d:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NIFTY50")
    ap.add_argument("--date")
    ap.add_argument("--time", default="11:00")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--skip-model", action="store_true")
    a = ap.parse_args()
    pure_tests()
    if not a.skip_model:
        model_tests(a)
    print("\n==================== RESULT SUMMARY (M5b test) ====================")
    print(f"{sum(results)}/{len(results)} checks passed")
    print("ALL PASS" if all(results) else "SOME FAILED")
    print("===================================================================")


if __name__ == "__main__":
    main()