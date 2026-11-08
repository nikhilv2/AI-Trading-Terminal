"""Bar-interval tests (1/5/10/15/30 min). Offline: no model, no network, no GPU.

Covers: resampling correctness, no look-ahead for every interval, bar-grid snapping, time-of-day covariate spacing,
the full run_simulation -> verify -> metrics.compare path, the server API, and reading legacy (1-minute) paired_eval data.
"""
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import engine as eg
import metrics as mt
import paired_eval as pe
import upstox_data as ud

results = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    results.append(bool(ok))


def near(a, b, tol=1e-9):
    return a is not None and abs(a - b) <= tol


def day_bars(d, closes=None, drop=()):
    """One full session of 1-minute bars; bar i closes at 100+i by default (open = previous close)."""
    t0 = eg.make_ts(d, "09:15")
    idx = [i for i in range(375) if i not in set(drop)]
    c = np.array([100.0 + i for i in range(375)] if closes is None else closes)
    o = np.concatenate([[c[0]], c[:-1]])
    opens = [t0 + i * eg.MIN for i in idx]
    return pd.DataFrame({"ts_open": opens, "ts_close": [x + eg.MIN for x in opens],
                         "open": o[idx], "high": np.maximum(o, c)[idx] + 0.5, "low": np.minimum(o, c)[idx] - 0.5,
                         "close": c[idx], "volume": np.full(len(idx), 10.0), "oi": 0.0})[ud.COLS]


class MockRunner:
    """Stand-in for the model: median = last context value, band proportional to horizon (in model space)."""
    info = {"class": "Mock", "path": "-", "device": "cpu", "per_core_batch_size": 1, "precision": "fp32"}
    output_info = None

    def forecast(self, x, horizon, cov=None):
        z = np.array([-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28])
        last = float(x[-1])
        scale = max(float(np.std(np.diff(x[-100:]))), 1e-6)
        h = np.sqrt(np.arange(1, horizon + 1))[:, None]
        return last + z[None, :] * scale * h


def many_days(n=30, seed=3):
    rng = np.random.default_rng(seed)
    frames, d, price, made = [], date(2026, 9, 1), 24000.0, 0
    while made < n:
        if d.weekday() < 5:
            c = price * np.exp(np.cumsum(rng.normal(0, 0.0004, 375)))
            frames.append(day_bars(d, c))
            price = float(c[-1]) * (1 + rng.normal(0, 0.003))
            made += 1
        d = date.fromordinal(d.toordinal() + 1)
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ 1. resampling
def test_resample():
    d = date(2026, 11, 6)
    b1 = day_bars(d)
    check("R0 interval 1 returns the bars unchanged", eg.resample_bars(b1, 1) is b1)
    for iv, n_exp, last_open, last_close in ((5, 75, "15:25", "15:30"), (10, 37, "15:15", "15:25"),
                                             (15, 25, "15:15", "15:30"), (30, 12, "14:45", "15:15")):
        r = eg.resample_bars(b1, iv)
        check(f"R1 {iv}-min: {n_exp} complete bars per session (== bars_per_day)", len(r) == n_exp == eg.bars_per_day(iv), str(len(r)))
        check(f"R2 {iv}-min: first bar opens 09:15, last opens {last_open} and closes {last_close}",
              f"{r.ts_open.iloc[0]:%H:%M}" == "09:15" and f"{r.ts_open.iloc[-1]:%H:%M}" == last_open
              and f"{r.ts_close.iloc[-1]:%H:%M}" == last_close)
        check(f"R3 {iv}-min: ts_close = ts_open + {iv} min, bars are contiguous",
              bool(((r.ts_close - r.ts_open) == pd.Timedelta(minutes=iv)).all())
              and bool((r.ts_open.diff().dropna() == pd.Timedelta(minutes=iv)).all()))
        check(f"R4 {iv}-min: columns match the cache schema", list(r.columns) == ud.COLS)
    r = eg.resample_bars(b1, 5)
    first, second = r.iloc[0], r.iloc[1]          # minutes 0..4 and 5..9
    check("R5 OHLCV aggregation: open = first open, close = last close",
          near(first.open, 100.0) and near(first.close, 104.0) and near(second.open, 104.0) and near(second.close, 109.0))
    check("R6 OHLCV aggregation: high = max high, low = min low, volume = sum",
          near(first.high, 104.5) and near(first.low, 99.5) and near(first.volume, 50.0))

    # a missing 1-minute bar drops exactly the bucket it belongs to
    r = eg.resample_bars(day_bars(d, drop=[7]), 5)          # minute 7 is in bucket 1 (09:20-09:25)
    opens = [f"{t:%H:%M}" for t in r.ts_open]
    check("R7 a missing minute drops that bucket only", len(r) == 74 and "09:20" not in opens and "09:15" in opens and "09:25" in opens)

    # sessions never mix
    two = pd.concat([day_bars(date(2026, 11, 5)), day_bars(d)], ignore_index=True)
    r = eg.resample_bars(two, 30)
    check("R8 two sessions -> 2 x 12 bars; no bar spans the overnight gap",
          len(r) == 24 and bool((r.ts_close - r.ts_open == pd.Timedelta(minutes=30)).all()))
    check("R9 bars outside 09:15-15:29 are ignored",
          len(eg.resample_bars(pd.concat([b1, b1.assign(ts_open=b1.ts_open + pd.Timedelta(minutes=400),
                                                        ts_close=b1.ts_close + pd.Timedelta(minutes=400))]), 15)) == 25)
    bad = 0
    for iv in (2, 7, 60, 0):
        try:
            eg.resample_bars(b1, iv)
        except ValueError:
            bad += 1
    check("R10 unsupported intervals are rejected", bad == 4)
    check("R11 load_history window grows with coarser bars (context in bars -> more calendar days)",
          eg.bars_per_day(30) == 12 and eg.bars_per_day(5) == 75 and eg.bars_per_day(1) == 375)


# ------------------------------------------------------------------ 2. no look-ahead
def test_no_lookahead():
    bars = many_days(8)
    ok_all = True
    armed = True
    for iv in eg.INTERVALS:
        full = eg.Replay(eg.resample_bars(bars, iv))
        for hhmm in ("09:45", "10:00", "10:17", "11:30", "13:05", "14:45"):     # on-grid and off-grid origins
            T = eg.make_ts(date(2026, 9, 10), hhmm)
            if T not in set(bars.ts_open) and T - eg.MIN not in set(bars.ts_open):
                continue
            poisoned = bars.copy()
            m = poisoned["ts_close"] > T
            armed &= bool(m.sum() > 0)
            for col in ("open", "high", "low", "close", "volume"):
                poisoned.loc[m, col] = 1e9
            trunc = bars[bars["ts_close"] <= T].copy()
            a = full.context(T, 60)
            for other in (eg.Replay(eg.resample_bars(poisoned, iv)), eg.Replay(eg.resample_bars(trunc, iv))):
                b = other.context(T, 60)
                ok_all &= (len(a) == len(b)) and bool(np.array_equal(a[["open", "high", "low", "close", "volume"]].to_numpy(),
                                                                     b[["open", "high", "low", "close", "volume"]].to_numpy()))
            ok_all &= bool((a["ts_close"] <= T).all())
    check("N0 poison is armed (future 1-minute rows were changed)", armed)
    check("N1 resampled context identical with poisoned / removed future, every interval, on- and off-grid origins", ok_all)

    # every setting x every interval: prepare_input unchanged by the future
    T = eg.make_ts(date(2026, 9, 10), "11:00")
    poisoned = bars.copy()
    poisoned.loc[poisoned["ts_close"] > T, ["open", "high", "low", "close", "volume"]] = 1e9
    ok = True
    for iv in eg.INTERVALS:
        ra, rb = eg.Replay(eg.resample_bars(bars, iv)), eg.Replay(eg.resample_bars(poisoned, iv))
        for form in pe.eg.INPUT_FORMS:
            for cv in eg.COVARIATES:
                st = {**eg.DEFAULT_SETTINGS, "input_form": form, "covariates": cv, "interval": iv, "context_len": 100}
                xa, ca, _ = eg.prepare_input(ra.context(T, 100), T, st, 3)
                xb, cb, _ = eg.prepare_input(rb.context(T, 100), T, st, 3)
                ok &= np.array_equal(xa, xb) and ((ca is None and cb is None) or np.array_equal(ca, cb))
    check("N2 model input + covariates identical with poisoned future for all forms x covariates x intervals", ok)


# ------------------------------------------------------------------ 3. grid, covariate, settings
def test_grid_cov_settings():
    d = date(2026, 11, 6)
    snap = lambda hhmm, iv: f"{eg.snap_to_grid(eg.make_ts(d, hhmm), iv):%H:%M}"
    check("G1 09:15 and on-grid times are unchanged", snap("09:15", 30) == "09:15" and snap("10:15", 30) == "10:15" and snap("10:00", 15) == "10:00")
    check("G2 off-grid times move forward to the next boundary",
          snap("10:00", 10) == "10:05" and snap("10:00", 30) == "10:15" and snap("10:01", 5) == "10:05" and snap("10:31", 15) == "10:45")
    check("G3 intervals 1 is the identity", snap("10:07", 1) == "10:07")

    ctx = eg.resample_bars(day_bars(d), 15)
    T = eg.make_ts(d, "10:15")
    c = eg._tod_cov(ctx[ctx.ts_close <= T], T, 3, 15)
    check("C1 interval-15 covariate: future bars are 15 min apart (10:15, 10:30, 10:45 -> 60/75/90 of 374)",
          near(c[0, -3], 60 / 374, 1e-6) and near(c[0, -2], 75 / 374, 1e-6) and near(c[0, -1], 90 / 374, 1e-6))
    c1 = eg._tod_cov(eg.Replay(day_bars(d)).context(T, 4), T, 2, 1)
    check("C2 interval-1 covariate unchanged (spacing 1 min)", near(c1[0, -1] - c1[0, -2], 1 / 374, 1e-6))
    st = {**eg.DEFAULT_SETTINGS, "interval": 15}
    x, cov, _ = eg.prepare_input(ctx[ctx.ts_close <= T], T, st, 3)
    check("C3 prepare_input covariate length = context bars + horizon bars", cov.shape == (1, len(x) + 3))

    bad = 0
    for iv in (2, 45, None):
        try:
            eg.check_settings({**eg.DEFAULT_SETTINGS, "interval": iv})
        except ValueError:
            bad += 1
    check("S1 invalid interval rejected by check_settings", bad == 3)
    check("S2 settings without an interval still validate (read as 1 minute)",
          eg.check_settings({"input_form": "raw_close", "covariates": "none"}) is None)
    check("S3 per-interval defaults exist and are valid for every interval",
          set(eg.INTERVAL_DEFAULTS) == set(eg.INTERVALS)
          and all(v["context_len"] in (256, 512, 1024, 2048) and 1 <= v["horizon"] <= 60 for v in eg.INTERVAL_DEFAULTS.values()))


# ------------------------------------------------------------------ 4. simulation + metrics end to end
def test_simulation(tmp):
    bars = many_days(30)
    day = sorted({t.date() for t in bars.ts_open})[-1]
    ud_get, eg_runs = ud.get_range, eg.RUNS
    ud.get_range = lambda sym, a, b: bars[(bars.ts_open.dt.date >= a) & (bars.ts_open.dt.date <= b)].reset_index(drop=True)
    eg.RUNS = Path(tmp) / "runs"
    eg.RUNS.mkdir()
    try:
        for iv in eg.INTERVALS:
            logs = []
            run_dir, snaps, rp, T0 = eg.run_simulation(
                MockRunner(), "TEST", day, "10:00", horizon=3, n_steps=4,
                settings={"context_len": 64, "interval": iv, "input_form": "log_price_stitched", "covariates": "time_of_day"},
                log=logs.append)
            on_grid = ((T0 - eg.make_ts(day, "09:15")).total_seconds() / 60) % iv == 0
            check(f"E1 {iv}-min: origin is on the bar grid (10:00 requested -> {T0:%H:%M})", on_grid)
            check(f"E2 {iv}-min: snapshots carry interval, targets are {iv} min apart",
                  all(s["interval"] == iv for s in snaps)
                  and all(pd.Timestamp(s["target_open_ts"][1]) - pd.Timestamp(s["target_open_ts"][0]) == pd.Timedelta(minutes=iv) for s in snaps))
            check(f"E3 {iv}-min: successive origins are {iv} min apart, no more than 5 snapshots",
                  len(snaps) == 5 and all(pd.Timestamp(snaps[k + 1]["origin_ts"]) - pd.Timestamp(snaps[k]["origin_ts"]) == pd.Timedelta(minutes=iv)
                                          for k in range(len(snaps) - 1)))
            check(f"E4 {iv}-min: last context bar of every snapshot closed at or before its origin",
                  all(pd.Timestamp(s["last_context_bar_open"]) + pd.Timedelta(minutes=iv) <= pd.Timestamp(s["origin_ts"]) for s in snaps))
            check(f"E5 {iv}-min: immutable snapshots verify", eg.verify_run(run_dir) == [])
            res = mt.compare(mt.load_snaps(run_dir), rp, lag=0)
            check(f"E6 {iv}-min: metrics.compare runs; rows = revealed bars; interval reported",
                  res["interval"] == iv and res["n_revealed"] == 4 and res["summary_all"]["n_latest"] == 4)
            tgt = [pd.Timestamp(r["target"]) for r in res["rows"]]
            check(f"E7 {iv}-min: scored targets are {iv} min apart and every scored bar had closed by the last origin",
                  all(b - a == pd.Timedelta(minutes=iv) for a, b in zip(tgt, tgt[1:]))
                  and all(t + pd.Timedelta(minutes=iv) <= T0 + 4 * pd.Timedelta(minutes=iv) for t in tgt))

        # a run that hits the end of the session stops cleanly instead of crashing
        run_dir, snaps, rp, T0 = eg.run_simulation(
            MockRunner(), "TEST", day, "14:45", horizon=2, n_steps=10,
            settings={"context_len": 64, "interval": 30}, log=lambda *_: None)
        check("E8 30-min run started at 14:45 stops at the session end (14:45 and 15:15 origins only)", len(snaps) == 2, str(len(snaps)))
        try:
            eg.run_simulation(MockRunner(), "TEST", day, "15:20", horizon=2, n_steps=3,
                              settings={"context_len": 64, "interval": 30}, log=lambda *_: None)
            refused = False
        except SystemExit:
            refused = True
        check("E9 a start with no bar boundary left in the session is refused", refused)
        # fast back-to-back runs (same HHMMSS) must not collide on the run directory
        dirs = []
        for form in ("raw_close", "log_price_stitched", "raw_close", "log_price_stitched"):
            rd, *_ = eg.run_simulation(MockRunner(), "TEST", day, "10:15", horizon=2, n_steps=2,
                                       settings={"context_len": 64, "interval": 30, "input_form": form, "covariates": "none"},
                                       log=lambda *_: None)
            dirs.append(rd)
        check("E10 four identical-looking fast runs get four distinct run directories", len({str(d) for d in dirs}) == 4)
    finally:
        ud.get_range, eg.RUNS = ud_get, eg_runs
    return bars, day


# ------------------------------------------------------------------ 5. server API
def test_server(bars, day):
    from starlette.testclient import TestClient
    import server as sv
    ud_get, eg_runs, mr = ud.get_range, eg.RUNS, eg.ModelRunner
    tmp = Path(tempfile.mkdtemp())
    ud.get_range = lambda sym, a, b: bars[(bars.ts_open.dt.date >= a) & (bars.ts_open.dt.date <= b)].reset_index(drop=True)
    eg.RUNS = tmp
    eg.ModelRunner = lambda *a, **k: MockRunner()
    try:
        with TestClient(sv.app) as c:
            m = c.get("/api/meta").json()
            check("A1 /api/meta lists intervals and per-interval defaults",
                  m["intervals"] == list(eg.INTERVALS) and m["defaults"]["30"]["context_len"] == 256)
            body = {"symbol": "NIFTY50", "date": str(day), "time": "10:00", "interval": 10, "horizon": 3, "ctx": 256,
                    "steps": 3, "input_form": "log_price_stitched", "covariates": "time_of_day"}
            r = c.post("/api/load", json=body)
            j = r.json()
            check("A2 load with interval 10 at 10:00 snaps the start to 10:05 and says so",
                  r.status_code == 200 and j["snapped"] is True and j["T0"].endswith("10:05:00") and j["interval"] == 10, str(j.get("T0")))
            check("A3 plotted history is in 10-min bars",
                  pd.Timestamp(j["today"]["x"][1]) - pd.Timestamp(j["today"]["x"][0]) == pd.Timedelta(minutes=10))
            s0 = c.post("/api/start").json()["snap"]
            check("A4 initial forecast targets are 10 min apart",
                  pd.Timestamp(s0["targets"][1]) - pd.Timestamp(s0["targets"][0]) == pd.Timedelta(minutes=10) and len(s0["targets"]) == 3)
            s1 = c.post("/api/step").json()
            check("A5 step reveals one 10-min bar and moves the origin by 10 min",
                  s1["done"] is False and pd.Timestamp(s1["snap"]["origin"]) - pd.Timestamp(s0["origin"]) == pd.Timedelta(minutes=10)
                  and pd.Timestamp(s1["bar"]["x"]) == pd.Timestamp(s0["origin"]))
            c.post("/api/step"); c.post("/api/step")
            cmp = c.get("/api/compare").json()
            check("A6 compare works and reports the interval", cmp["interval"] == 10 and cmp["n_revealed"] == 3)
            check("A7 an invalid interval is rejected with a clear 400",
                  c.post("/api/load", json={**body, "interval": 7}).status_code == 400)
            late = c.post("/api/load", json={**body, "time": "15:20", "interval": 30})
            check("A8 a start after the last bar boundary is rejected", late.status_code == 400, late.text[:80])
            old = c.post("/api/load", json={k: v for k, v in body.items() if k != "interval"})
            check("A9 requests without an interval still work as 1-minute (older clients)",
                  old.status_code == 200 and old.json()["interval"] == 1 and old.json()["snapped"] is False)
    finally:
        ud.get_range, eg.RUNS, eg.ModelRunner = ud_get, eg_runs, mr


# ------------------------------------------------------------------ 6. paired_eval: legacy data + pruning
def test_paired(tmp):
    old_root = pe.OUT_ROOT
    pe.OUT_ROOT = Path(tmp) / "paired"
    try:
        bars = pe.synthetic_bars(n_days=30)
        a = __import__("argparse").Namespace(symbol="T", tag="new", start=None, end=None, days_limit=0, first="09:45", last="14:30",
                                              every=30, ctx=64, batch=16, variants=None, include_retired=False,
                                              intervals=[1, 10], steps=None)
        pe.run(a, runner=pe._MockRunner(), bars=bars)
        df = pd.read_csv(pe.OUT_ROOT / "new" / "rows.csv")
        # a legacy file: no interval / minutes columns, 1-minute only
        leg = df[df.interval == 1].drop(columns=["interval", "minutes"])
        (pe.OUT_ROOT / "legacy").mkdir()
        leg.to_csv(pe.OUT_ROOT / "legacy" / "rows.csv", index=False)
        try:
            pe.summarize("legacy")
            ok = (pe.OUT_ROOT / "legacy" / "summary.md").exists() and "1 min" in (pe.OUT_ROOT / "legacy" / "summary.md").read_text()
        except Exception as e:                                   # noqa: BLE001
            ok = False
            print("      legacy summarize raised:", repr(e))
        check("P1 a pre-interval rows.csv (no interval column) still summarizes, as 1-minute", ok)
        a2 = __import__("argparse").Namespace(**{**vars(a), "tag": "legacy"})
        try:
            pe.run(a2, runner=pe._MockRunner(), bars=bars)
            refused = False
        except SystemExit as e:
            refused = "before bar intervals existed" in str(e)
        check("P2 extending a legacy tag is refused with a clear message (no silent column mismatch)", refused)
        check("P3 --every not a multiple of the interval is refused",
              _raises(lambda: pe.make_origins(pe.FastReplay(eg.resample_bars(bars, 30)), [date(2026, 1, 6)], "09:45", "14:30", 20, 64, 3, 30)))
        check("P4 default variants exclude the retired ones; retired list carries evidence",
              set(pe.VARIANTS) == {"A_raw_none", "B_stitch_none", "D_stitch_tod"} and all(len(v["why"]) > 40 for v in pe.RETIRED.values()))
        txt = (pe.OUT_ROOT / "new" / "rows.csv")
        pe.summarize("new")
        md = (pe.OUT_ROOT / "new" / "summary.md").read_text()
        check("P5 summary has the cross-interval headline and per-interval detail",
              "MAE ratio vs no-change" in md and "+30 min" in md and "## Interval 10 min: detail" in md and "## Interval 1 min: detail" in md)
    finally:
        pe.OUT_ROOT = old_root


def _raises(fn):
    try:
        fn()
    except SystemExit:
        return True
    return False


def main():
    tmp = tempfile.mkdtemp()
    test_resample()
    test_no_lookahead()
    test_grid_cov_settings()
    bars, day = test_simulation(tmp)
    test_server(bars, day)
    test_paired(tmp)
    print("\n==================== RESULT SUMMARY (interval tests) ====================")
    print(f"{sum(results)}/{len(results)} checks passed")
    print("ALL PASS" if all(results) else "SOME FAILED")
    print("=========================================================================")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
