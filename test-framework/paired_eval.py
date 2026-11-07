"""paired_eval.py - many-origin, paired evaluation of TimesFM-3 variants vs honest baselines.

Reuses engine.py (prepare_input, Replay, ModelRunner) and metrics.py (vol_band, pinball).
No look-ahead: a forecast at origin T only sees bars whose ts_close <= T.

Commands (run with the TimesFM venv python):
  python paired_eval.py run --symbol NIFTY50 --days-limit 120 --every 30
  python paired_eval.py summarize --tag <tag>
  python paired_eval.py selftest            # no GPU, no network: mock model on synthetic data

Each (origin, variant) is forecast ONCE at horizon 30; steps 1, 5, 10, 30 are scored from that run.
Results are appended to data/paired/<tag>/rows.csv, so a stopped run can be resumed (same --tag).
"""
import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import engine as eg
import metrics as mt
import upstox_data as ud

HORIZON = 30
STEPS = (1, 5, 10, 30)
VARIANTS = {
    "A_raw_none":    dict(input_form="raw_close",          covariates="none"),
    "B_stitch_none": dict(input_form="log_price_stitched", covariates="none"),
    "C_logret_none": dict(input_form="log_returns",        covariates="none"),
    "D_stitch_tod":  dict(input_form="log_price_stitched", covariates="time_of_day"),
    "E_raw_tod":     dict(input_form="raw_close",          covariates="time_of_day"),
}
OUT_ROOT = ud.DATA / "paired"
MIN = pd.Timedelta(minutes=1)


# ------------------------------------------------------------------ fast replay (same rule: ts_close <= T)
class FastReplay(eg.Replay):
    """Same contract as engine.Replay but uses searchsorted instead of scanning all bars."""

    def __init__(self, bars):
        super().__init__(bars)
        self._close = self.bars["ts_close"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
        self._open = self.bars["ts_open"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
        self.closes = self.bars["close"].to_numpy(dtype=np.float64)

    def n_closed(self, T):
        t = np.datetime64(pd.Timestamp(T).tz_convert("UTC").tz_localize(None))
        return int(np.searchsorted(self._close, t, side="right"))

    def closed_until(self, T):
        return self.bars.iloc[: self.n_closed(T)]

    def context(self, T, n):
        k = self.n_closed(T)
        return self.bars.iloc[max(0, k - n):k]


# ------------------------------------------------------------------ origins
def make_origins(rp, days, first, last, every, ctx_len, max_step):
    """Origins T on each day. Needs a full context and the next max_step bars present and contiguous."""
    out = []
    for d in days:
        t = eg.make_ts(d, first)
        t_last = eg.make_ts(d, last)
        while t <= t_last:
            k = rp.n_closed(t)
            if k >= ctx_len and k + max_step <= len(rp.bars):
                t0 = np.datetime64(t.tz_convert("UTC").tz_localize(None))
                want = t0 + np.arange(max_step) * np.timedelta64(1, "m")
                if np.array_equal(rp._open[k:k + max_step], want.astype(rp._open.dtype)):
                    out.append((d, t, k))
            t += every * MIN
    return out


# ------------------------------------------------------------------ batched model call
def batch_forecast(runner, xs, covs, horizon):
    arrs = [np.asarray(x, dtype=np.float32) for x in xs]
    kw = {}
    if covs[0] is not None:
        kw["past_future_covariates"] = [np.asarray(c, dtype=np.float32) for c in covs]
    outs = list(runner.f.predict_batch(arrs, horizon=horizon, return_quantiles=True, **kw))
    if len(outs) != len(arrs):
        raise RuntimeError(f"expected {len(arrs)} outputs, got {len(outs)}")
    res = []
    for out in outs:
        q = getattr(out, "quantiles", None)
        if q is None and isinstance(out, dict):
            q = out.get("quantiles")
        q = eg._to_np(q)
        if q.ndim == 3 and q.shape[0] == 1:
            q = q[0]
        if q.shape != (horizon, 9) or not np.all(np.isfinite(q)):
            raise RuntimeError(f"bad quantiles shape/values: {q.shape}")
        res.append(q)
    return res


def _sgn(x):
    return int(bool(x > 0)) - int(bool(x < 0))


# ------------------------------------------------------------------ run
def run(args, runner=None, bars=None):
    sym = args.symbol
    tag = args.tag or f"{sym}_{datetime.now():%Y%m%d_%H%M}"
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.csv"

    if bars is None:
        end = date.fromisoformat(args.end) if args.end else datetime.now(ud.IST).date() - timedelta(days=1)
        start = date.fromisoformat(args.start) if args.start else end - timedelta(days=400)
        bars = ud.get_range(sym, start - timedelta(days=15), end)
    rp = FastReplay(bars)
    days = sorted({t.date() for t in rp.bars["ts_open"]})
    if args.start:
        days = [d for d in days if d >= date.fromisoformat(args.start)]
    if args.end:
        days = [d for d in days if d <= date.fromisoformat(args.end)]
    if args.days_limit:
        days = days[-args.days_limit:]
    origins = make_origins(rp, days, args.first, args.last, args.every, args.ctx, HORIZON)
    print(f"{sym}: {len(days)} sessions, {len(origins)} origins, variants {list(args.variants)}, ctx {args.ctx}")
    if not origins:
        raise SystemExit("no usable origins (not enough cached history for the context?)")

    done = set()
    if rows_path.exists():
        prev = pd.read_csv(rows_path, usecols=["origin_ts", "variant"]).drop_duplicates()
        done = set(zip(prev["origin_ts"], prev["variant"]))
        print(f"resuming: {len(done)} (origin, variant) pairs already done")

    if runner is None:
        runner = eg.ModelRunner()

    # origin-level quantities shared by all variants
    shared = {}
    for d, T, k in origins:
        anchor, vb = mt.vol_band(rp, T, HORIZON)
        c6 = rp.closes[k - 6:k]
        actual = rp.closes[k:k + HORIZON]
        shared[T] = (anchor, vb, float(c6[-1] - c6[0]), actual)

    t_start, n_done_total, n_total = time.time(), 0, len(origins) * len(args.variants)
    for vname in args.variants:
        settings = {"context_len": args.ctx, **VARIANTS[vname]}
        todo = [(d, T, k) for d, T, k in origins if (T.isoformat(), vname) not in done]
        for b0 in range(0, len(todo), args.batch):
            batch = todo[b0:b0 + args.batch]
            items = []
            for d, T, k in batch:
                ctx = rp.context(T, args.ctx)
                assert len(ctx) == args.ctx and ctx["ts_close"].iloc[-1] <= T      # no look-ahead
                x, cov, back = eg.prepare_input(ctx, T, settings, HORIZON)
                items.append((x, cov, back))
            qs_raw = batch_forecast(runner, [i[0] for i in items], [i[1] for i in items], HORIZON)
            rows = []
            for (d, T, k), (x, cov, back), q_raw in zip(batch, items, qs_raw):
                q = back(q_raw)
                anchor, vb, mom5, actual = shared[T]
                for s in STEPS:
                    a = float(actual[s - 1])
                    qm = q[s - 1]
                    rows.append({
                        "day": str(d), "origin_ts": T.isoformat(), "hhmm": f"{T:%H:%M}", "variant": vname, "step": s,
                        "anchor": anchor, "actual": a, "q10": qm[0], "q50": qm[4], "q90": qm[8],
                        "abs_m": abs(qm[4] - a), "abs_nc": abs(anchor - a),
                        "pin_m": mt.pinball(a, qm),
                        "pin_vb": mt.pinball(a, vb[s - 1]) if vb is not None else np.nan,
                        "in_m": int(qm[0] <= a <= qm[8]),
                        "in_vb": int(vb[s - 1][0] <= a <= vb[s - 1][8]) if vb is not None else np.nan,
                        "sg_model": _sgn(qm[4] - anchor), "sg_actual": _sgn(a - anchor), "sg_mom5": _sgn(mom5),
                        "band_approx": int(settings["input_form"] == "log_returns")})
            pd.DataFrame(rows).to_csv(rows_path, mode="a", header=not rows_path.exists(), index=False)
            n_done_total += len(batch)
            el = time.time() - t_start
            print(f"  {vname}: {b0 + len(batch)}/{len(todo)} origins | elapsed {el / 60:.1f} min", flush=True)
    print(f"\nDone. Rows in {rows_path}\nNext: python paired_eval.py summarize --tag {tag}")
    return tag


# ------------------------------------------------------------------ summary
def boot_ratio(a, b, reps=2000, seed=0):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = len(a)
    if n < 3 or b.sum() == 0:
        return (float("nan"),) * 3
    idx = np.random.default_rng(seed).integers(0, n, (reps, n))
    den = b[idx].sum(1)
    r = a[idx].sum(1) / np.where(den == 0, np.nan, den)
    return a.sum() / b.sum(), float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))


def f3(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def pct(t):
    return f"{100 * t[0]:.1f}% [{100 * t[1]:.1f}, {100 * t[2]:.1f}]"


def day_sums(g, cols):
    return g.groupby("day")[cols].sum()


def table(df, title, lines):
    lines.append(f"\n### {title}\n")
    lines.append("| variant | n origins | days | MAE ratio vs no-change | origins better | pinball ratio vs vol band | P10-P90 model / vol band | direction hit (model) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for v, g in df.groupby("variant"):
        ds = day_sums(g.assign(one=1, dirn=((g.sg_actual != 0) & (g.sg_model != 0)).astype(int),
                               hit=((g.sg_actual != 0) & (g.sg_model == g.sg_actual)).astype(int)),
                      ["abs_m", "abs_nc", "pin_m", "pin_vb", "in_m", "in_vb", "one", "dirn", "hit"])
        mae = boot_ratio(ds.abs_m, ds.abs_nc)
        pin = boot_ratio(ds.pin_m, ds.pin_vb)
        cov_m = ds.in_m.sum() / ds.one.sum()
        cov_v = ds.in_vb.sum() / ds.one.sum()
        better = float((g.abs_m < g.abs_nc).sum() / max(1, (g.abs_m != g.abs_nc).sum()))
        dr = boot_ratio(ds.hit, ds.dirn)
        approx = " *(band approx)*" if g.band_approx.iloc[0] == 1 else ""
        lines.append(f"| {v}{approx} | {int(ds.one.sum())} | {len(ds)} | {f3(mae)} | {100 * better:.1f}% | {f3(pin)} | "
                     f"{100 * cov_m:.1f}% / {100 * cov_v:.1f}% | {pct(dr)} (n={int(ds.dirn.sum())}) |")


def summarize(tag):
    d = OUT_ROOT / tag
    df = pd.read_csv(d / "rows.csv")
    df = df.dropna(subset=["pin_vb"])
    lines = [f"# Paired evaluation: {tag}\n",
             f"{df.origin_ts.nunique()} origins, {df.day.nunique()} sessions, variants: {', '.join(sorted(df.variant.unique()))}.",
             "Ratios < 1 mean the model beats the baseline. Intervals are 95%, resampling whole days. "
             "Overlapping windows are handled by the day-level resampling, but all numbers are still correlated within a day.\n"]
    for s in STEPS:
        table(df[df.step == s], f"Step {s} (forecast {s} minute(s) ahead)", lines)

    # direction baselines (variant-independent)
    lines.append("\n### Direction baselines (hit rate on bars that moved; 50% = chance)\n")
    lines.append("| step | momentum (last 5 min) | mean-reversion | always up |")
    lines.append("|---|---|---|---|")
    base = df[df.variant == df.variant.iloc[0]]
    for s in STEPS:
        g = base[base.step == s]
        mv = g[g.sg_actual != 0]
        mm = mv[mv.sg_mom5 != 0]
        ds_m = mm.assign(h=(mm.sg_mom5 == mm.sg_actual).astype(int), n=1).groupby("day")[["h", "n"]].sum()
        ds_r = mm.assign(h=(mm.sg_mom5 == -mm.sg_actual).astype(int), n=1).groupby("day")[["h", "n"]].sum()
        ds_u = mv.assign(h=(mv.sg_actual > 0).astype(int), n=1).groupby("day")[["h", "n"]].sum()
        lines.append(f"| {s} | {pct(boot_ratio(ds_m.h, ds_m.n))} | {pct(boot_ratio(ds_r.h, ds_r.n))} | {pct(boot_ratio(ds_u.h, ds_u.n))} |")

    # select on the first two thirds of days, confirm on the last third
    days = sorted(df.day.unique())
    cut = days[int(len(days) * 2 / 3)] if len(days) >= 6 else None
    if cut:
        lines.append(f"\n### Select on first two thirds of days, confirm once on the last third (cut at {cut}), step 10\n")
        lines.append("| variant | MAE ratio (selection) | MAE ratio (confirmation) | pinball ratio (selection) | pinball ratio (confirmation) |")
        lines.append("|---|---|---|---|---|")
        g10 = df[df.step == 10]
        for v, g in g10.groupby("variant"):
            row = []
            for part in (g[g.day < cut], g[g.day >= cut]):
                ds = day_sums(part, ["abs_m", "abs_nc", "pin_m", "pin_vb"])
                row.append((boot_ratio(ds.abs_m, ds.abs_nc), boot_ratio(ds.pin_m, ds.pin_vb)))
            lines.append(f"| {v} | {f3(row[0][0])} | {f3(row[1][0])} | {f3(row[0][1])} | {f3(row[1][1])} |")

    # by time of day, step 10
    lines.append("\n### MAE ratio by time of day, step 10\n")
    g10 = df[df.step == 10].copy()
    hh = g10.hhmm.str[:2].astype(int) * 60 + g10.hhmm.str[3:].astype(int)
    g10["bucket"] = pd.cut(hh, [0, 11 * 60, 13 * 60, 15 * 60 + 1], right=False, labels=["before 11:00", "11:00-13:00", "13:00 on"])
    lines.append("| variant | before 11:00 | 11:00-13:00 | 13:00 on |")
    lines.append("|---|---|---|---|")
    for v, g in g10.groupby("variant"):
        cells = []
        for b in ["before 11:00", "11:00-13:00", "13:00 on"]:
            ds = day_sums(g[g.bucket == b], ["abs_m", "abs_nc"])
            cells.append(f3(boot_ratio(ds.abs_m, ds.abs_nc)))
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    n_cmp = df.variant.nunique() * len(STEPS)
    lines.append(f"\nNote: {n_cmp} variant-step comparisons are shown. With that many, expect one or two to look better than 1.0 by chance; "
                 "only trust a gain that also holds in the confirmation block and across steps.")
    text = "\n".join(lines)
    (d / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved {d / 'summary.md'}")


# ------------------------------------------------------------------ selftest (no GPU, no network)
class _Out:
    def __init__(self, q):
        self.quantiles = q


class _MockF:
    """Pretends to be TimesFM: persistence + noise band. Used only to test the plumbing."""

    def predict_batch(self, arrs, horizon, return_quantiles=True, past_future_covariates=None):
        z = np.array([-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28])
        for a in arrs:
            s = np.std(np.diff(a[-200:])) + 1e-6
            h = np.sqrt(np.arange(1, horizon + 1))[:, None]
            yield _Out(a[-1] + z[None, :] * s * h)


class _MockRunner:
    f = _MockF()


def synthetic_bars(n_days=40, seed=1):
    rng = np.random.default_rng(seed)
    rows, price = [], 24000.0
    d = date(2026, 1, 5)
    made = 0
    while made < n_days:
        if d.weekday() < 5:
            price *= 1 + rng.normal(0, 0.004)
            for m in range(375):
                o = price
                price *= 1 + rng.normal(0, 0.0003)
                t_open = eg.make_ts(d, "09:15") + m * MIN
                rows.append((t_open, t_open + MIN, o, max(o, price) + 1, min(o, price) - 1, price, 0.0, 0.0))
            made += 1
        d += timedelta(days=1)
    return pd.DataFrame(rows, columns=ud.COLS)


def selftest():
    import tempfile
    global OUT_ROOT
    OUT_ROOT = Path(tempfile.mkdtemp())
    bars = synthetic_bars()
    a = argparse.Namespace(symbol="TEST", tag="selftest", start=None, end=None, days_limit=0, first="09:45", last="14:30",
                           every=30, ctx=512, batch=8, variants=list(VARIANTS))
    run(a, runner=_MockRunner(), bars=bars)
    summarize("selftest")


# ------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--symbol", default="NIFTY50")
    r.add_argument("--tag", default="")
    r.add_argument("--start", default="")
    r.add_argument("--end", default="")
    r.add_argument("--days-limit", type=int, default=120, help="use only the most recent N sessions (0 = all cached)")
    r.add_argument("--first", default="09:45")
    r.add_argument("--last", default="14:30")
    r.add_argument("--every", type=int, default=30, help="minutes between forecast origins")
    r.add_argument("--ctx", type=int, default=2048)
    r.add_argument("--batch", type=int, default=16)
    r.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    s = sub.add_parser("summarize")
    s.add_argument("--tag", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    elif a.cmd == "summarize":
        summarize(a.tag)
    else:
        selftest()


if __name__ == "__main__":
    main()