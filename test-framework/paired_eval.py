"""paired_eval.py - many-origin, paired evaluation of TimesFM-3 variants vs honest baselines.

Reuses engine.py (prepare_input, Replay, ModelRunner) and metrics.py (vol_band, pinball).
No look-ahead: a forecast at origin T only sees bars whose ts_close <= T.

Bar interval: the cache holds 1-minute bars; --intervals picks the bar sizes to evaluate (1, 5, 10, 15, 30 minutes).
Each interval is run on bars resampled from the 1-minute cache (engine.resample_bars). Horizon, context and the
scored steps are counted in BARS of that interval; every report labels them in wall-clock minutes as well.

Commands (run with the TimesFM venv python):
  python paired_eval.py run --symbol NIFTY50 --days-limit 120 --every 30 --intervals 5 10 15 30
  python paired_eval.py summarize --tag <tag> [--detail]
  python paired_eval.py selftest            # no GPU, no network: mock model on synthetic data

Each (origin, variant, interval) is forecast ONCE at that interval's max scored step; the other steps are scored
from the same run. Results are appended to data/paired/<tag>/rows.csv, so a stopped run can be resumed (same --tag).
Old tags made before intervals existed (no `interval` column) can still be summarized (they are read as 1-minute),
but not resumed or extended: start a new tag.
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

INTERVALS = eg.INTERVALS
# Steps scored per interval, in BARS (wall-clock minutes = step x interval). The horizon of the run is the max step.
# 1-min keeps the original 1/5/10/30. Coarser bars score fewer, nearer steps: past ~30-90 minutes ahead the
# intraday regime has usually changed and the forecast is not worth scoring.
STEPS_BY_INTERVAL = {
    1:  (1, 5, 10, 30),   # +1, +5, +10, +30 min
    5:  (1, 2, 3, 6),     # +5, +10, +15, +30 min
    10: (1, 2, 3),        # +10, +20, +30 min
    15: (1, 2, 3),        # +15, +30, +45 min
    30: (1, 2, 3),        # +30, +60, +90 min
}
REF_MINUTES = 10          # the "reference step" used for select/confirm and time-of-day tables = nearest to this

# Active variants. Letters are kept from the first study so old and new tags line up.
VARIANTS = {
    "A_raw_none":    dict(input_form="raw_close",          covariates="none"),     # plain reference
    "B_stitch_none": dict(input_form="log_price_stitched", covariates="none"),
    "D_stitch_tod":  dict(input_form="log_price_stitched", covariates="time_of_day"),
}
# Retired after the 1-minute study NIFTY50_20261107_2044 (120 sessions, 1200 origins). Not run unless
# --include-retired is given. The evidence was gathered at 1-minute bars only.
RETIRED = {
    "C_logret_none": dict(
        spec=dict(input_form="log_returns", covariates="none"),
        why="Its P10-P90 band is an approximation (per-step quantiles summed), so it is far too wide: coverage "
            "98.5-99.9% and pinball ratio vs vol band 1.19 / 1.47 / 2.14 at steps 5 / 10 / 30. Its probabilistic "
            "output cannot be scored fairly, and its MAE was no better than B or D."),
    "E_raw_tod": dict(
        spec=dict(input_form="raw_close", covariates="time_of_day"),
        why="Worst point forecast: MAE ratio 1.018 [1.008, 1.030] at step 1 and 1.020 [1.004, 1.037] at step 30, "
            "both intervals wholly above 1 (worse than no-change); direction hit 48.9% at step 1."),
}
ALL_SPECS = {**VARIANTS, **{k: v["spec"] for k, v in RETIRED.items()}}
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
def make_origins(rp, days, first, last, every, ctx_len, max_step, interval=1):
    """Origins T on each day. Needs a full context and the next max_step bars present and contiguous.
    T must lie on the bar grid (09:15 + n * interval), and `every` must be a multiple of the interval."""
    if every % interval:
        raise SystemExit(f"--every {every} must be a multiple of the interval ({interval} min)")
    out = []
    for d in days:
        t = eg.snap_to_grid(eg.make_ts(d, first), interval)
        t_last = eg.make_ts(d, last)
        while t <= t_last:
            k = rp.n_closed(t)
            if k >= ctx_len and k + max_step <= len(rp.bars):
                t0 = np.datetime64(t.tz_convert("UTC").tz_localize(None))
                want = t0 + np.arange(max_step) * np.timedelta64(interval, "m")
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
def _resolve_variants(args):
    names = list(args.variants) if args.variants else list(VARIANTS)
    allowed = ALL_SPECS if getattr(args, "include_retired", False) else VARIANTS
    for v in names:
        if v not in allowed:
            if v in RETIRED:
                raise SystemExit(f"{v} is retired: {RETIRED[v]['why']}\nPass --include-retired to run it anyway.")
            raise SystemExit(f"unknown variant {v!r}; choose from {list(allowed)}")
    return names


def _resolve_intervals(args):
    ivs = sorted(set(int(i) for i in (args.intervals or INTERVALS)))
    for iv in ivs:
        eg.check_interval(iv)
    return ivs


def run(args, runner=None, bars=None):
    sym = args.symbol
    variants = _resolve_variants(args)
    intervals = _resolve_intervals(args)
    tag = args.tag or f"{sym}_{datetime.now():%Y%m%d_%H%M}"
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.csv"

    done = set()
    if rows_path.exists():
        have = pd.read_csv(rows_path, nrows=0).columns
        if "interval" not in have:
            raise SystemExit(f"{rows_path} was made before bar intervals existed (no 'interval' column). "
                             "It can still be summarized, but not extended. Use a new --tag for this run.")
        prev = pd.read_csv(rows_path, usecols=["origin_ts", "variant", "interval"]).drop_duplicates()
        done = set(zip(prev["origin_ts"], prev["variant"], prev["interval"].astype(int)))
        print(f"resuming: {len(done)} (origin, variant, interval) triples already done")

    ctx_for = lambda iv: args.ctx or eg.INTERVAL_DEFAULTS[iv]["context_len"]
    steps_for = lambda iv: tuple(sorted(args.steps)) if args.steps else STEPS_BY_INTERVAL[iv]

    if bars is None:
        end = date.fromisoformat(args.end) if args.end else datetime.now(ud.IST).date() - timedelta(days=1)
        start = date.fromisoformat(args.start) if args.start else end - timedelta(days=400)
        # history margin before `start`: enough sessions for the biggest context among the chosen intervals
        sessions_needed = max(ctx_for(iv) / eg.bars_per_day(iv) for iv in intervals)
        margin = max(15, int(sessions_needed * 1.6) + 5)
        bars = ud.get_range(sym, start - timedelta(days=margin), end)
    days = sorted({t.date() for t in bars["ts_open"]})          # sessions are chosen on the 1-minute bars,
    if args.start:                                               # so every interval is scored on the same days
        days = [d for d in days if d >= date.fromisoformat(args.start)]
    if args.end:
        days = [d for d in days if d <= date.fromisoformat(args.end)]
    if args.days_limit:
        days = days[-args.days_limit:]

    if runner is None:
        runner = eg.ModelRunner()

    t_start = time.time()
    for iv in intervals:
        ctx_len, steps = ctx_for(iv), steps_for(iv)
        horizon = max(steps)
        rp = FastReplay(eg.resample_bars(bars, iv))
        origins = make_origins(rp, days, args.first, args.last, args.every, ctx_len, horizon, iv)
        print(f"\n{sym} | interval {iv} min | ctx {ctx_len} bars | steps {steps} bars "
              f"(= +{', +'.join(str(s * iv) for s in steps)} min) | {len(days)} sessions, {len(origins)} origins | "
              f"variants {variants}")
        if not origins:
            print("  no usable origins for this interval (not enough cached history for the context?) - skipped")
            continue

        # origin-level quantities shared by all variants
        shared = {}
        for d, T, k in origins:
            anchor, vb = mt.vol_band(rp, T, horizon)
            c6 = rp.closes[k - 6:k]
            actual = rp.closes[k:k + horizon]
            shared[T] = (anchor, vb, float(c6[-1] - c6[0]), actual)

        for vname in variants:
            settings = {"context_len": ctx_len, "interval": iv, **ALL_SPECS[vname]}
            todo = [(d, T, k) for d, T, k in origins if (T.isoformat(), vname, iv) not in done]
            for b0 in range(0, len(todo), args.batch):
                batch = todo[b0:b0 + args.batch]
                items = []
                for d, T, k in batch:
                    ctx = rp.context(T, ctx_len)
                    assert len(ctx) == ctx_len and ctx["ts_close"].iloc[-1] <= T      # no look-ahead
                    x, cov, back = eg.prepare_input(ctx, T, settings, horizon)
                    items.append((x, cov, back))
                qs_raw = batch_forecast(runner, [i[0] for i in items], [i[1] for i in items], horizon)
                rows = []
                for (d, T, k), (x, cov, back), q_raw in zip(batch, items, qs_raw):
                    q = back(q_raw)
                    anchor, vb, mom5, actual = shared[T]
                    for st in steps:
                        a = float(actual[st - 1])
                        qm = q[st - 1]
                        rows.append({
                            "day": str(d), "origin_ts": T.isoformat(), "hhmm": f"{T:%H:%M}", "variant": vname,
                            "interval": iv, "step": st, "minutes": st * iv,
                            "anchor": anchor, "actual": a, "q10": qm[0], "q50": qm[4], "q90": qm[8],
                            "abs_m": abs(qm[4] - a), "abs_nc": abs(anchor - a),
                            "pin_m": mt.pinball(a, qm),
                            "pin_vb": mt.pinball(a, vb[st - 1]) if vb is not None else np.nan,
                            "in_m": int(qm[0] <= a <= qm[8]),
                            "in_vb": int(vb[st - 1][0] <= a <= vb[st - 1][8]) if vb is not None else np.nan,
                            "sg_model": _sgn(qm[4] - anchor), "sg_actual": _sgn(a - anchor), "sg_mom5": _sgn(mom5),
                            "band_approx": int(settings["input_form"] == "log_returns")})
                pd.DataFrame(rows).to_csv(rows_path, mode="a", header=not rows_path.exists(), index=False)
                el = time.time() - t_start
                print(f"  {iv:2d}m {vname}: {b0 + len(batch)}/{len(todo)} origins | elapsed {el / 60:.1f} min", flush=True)
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
    if not np.isfinite(t[0]):
        return "n/a"
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def pct(t):
    if not np.isfinite(t[0]):
        return "n/a"
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


def _load_rows(tag):
    df = pd.read_csv(OUT_ROOT / tag / "rows.csv")
    df = df.dropna(subset=["pin_vb"])
    if "interval" not in df.columns:          # tag made before intervals existed: 1-minute bars
        df["interval"] = 1
    df["interval"] = df["interval"].astype(int)
    if "minutes" not in df.columns:
        df["minutes"] = df["step"] * df["interval"]
    return df


def ref_step(df_i, iv):
    """Step (in bars) nearest to REF_MINUTES ahead; ties go to the larger step. 1-minute data gives step 10."""
    steps = sorted(df_i.step.unique())
    return min(steps, key=lambda st: (abs(st * iv - REF_MINUTES), -st))


def _cell(g):
    """Bootstrap stats for the rows of ONE (interval, step, variant)."""
    ds = day_sums(g.assign(one=1, dirn=((g.sg_actual != 0) & (g.sg_model != 0)).astype(int),
                           hit=((g.sg_actual != 0) & (g.sg_model == g.sg_actual)).astype(int)),
                  ["abs_m", "abs_nc", "pin_m", "pin_vb", "one", "dirn", "hit"])
    return {"mae": boot_ratio(ds.abs_m, ds.abs_nc), "pin": boot_ratio(ds.pin_m, ds.pin_vb),
            "dir": boot_ratio(ds.hit, ds.dirn), "n": int(ds.one.sum())}


def _momentum(g):
    mv = g[(g.sg_actual != 0) & (g.sg_mom5 != 0)]
    ds = mv.assign(h=(mv.sg_mom5 == mv.sg_actual).astype(int), n=1).groupby("day")[["h", "n"]].sum()
    return boot_ratio(ds.h, ds.n)


def headline(df, lines):
    """Cross-interval comparison. One row per (interval, horizon in minutes), one column per variant.
    Bold = the 95% interval is wholly on the good side (ratio < 1, direction hit > 50%)."""
    variants = sorted(df.variant.unique())
    approx = {v for v in variants if df[df.variant == v].band_approx.iloc[0] == 1}
    hdr = [f"{v}{chr(8224) if v in approx else ''}" for v in variants]
    cells, n_cmp, n_good = {}, 0, 0
    keys = sorted(df[["interval", "step", "minutes"]].drop_duplicates().itertuples(index=False, name=None))
    for iv, st, mins in keys:
        g = df[(df.interval == iv) & (df.step == st)]
        cells[(iv, st)] = {v: _cell(g[g.variant == v]) for v in variants}
        cells[(iv, st)]["_mom"] = _momentum(g[g.variant == variants[0]])
        cells[(iv, st)]["_n"] = g[g.variant == variants[0]].origin_ts.nunique()

    def md(title, key, good, fmt, extra=None):
        nonlocal n_cmp, n_good
        lines.append(f"\n### {title}\n")
        cols = hdr + ([extra[0]] if extra else [])
        lines.append("| interval | horizon | origins | " + " | ".join(cols) + " |")
        lines.append("|---|---|---|" + "---|" * len(cols))
        last_iv = None
        for iv, st, mins in keys:
            row = [f"{iv} min" if iv != last_iv else "", f"+{mins} min (step {st})", str(cells[(iv, st)]["_n"])]
            last_iv = iv
            for v in variants:
                t = cells[(iv, st)][v][key]
                txt = fmt(t)
                n_cmp += int(np.isfinite(t[0]))
                if np.isfinite(t[0]) and good(t):
                    txt, n_good = f"**{txt}**", n_good + 1
                row.append(txt)
            if extra:
                row.append(extra[1](cells[(iv, st)]))
            lines.append("| " + " | ".join(row) + " |")

    md("MAE ratio vs no-change (point forecast; < 1 = model beats 'price stays put')", "mae",
       lambda t: t[2] < 1, f3)
    md("Pinball ratio vs volatility band (full distribution; < 1 = model beats the plain vol band)", "pin",
       lambda t: t[2] < 1, f3)
    if approx:
        lines.append(f"\n{chr(8224)} log-returns band is an approximation (summed per-step quantiles); its pinball is not comparable.")
    md("Direction hit rate (on bars that moved; 50% = chance)", "dir",
       lambda t: t[1] > 0.5, pct, extra=("momentum (last 5 bars)", lambda c: pct(c["_mom"])))
    lines.append(f"\nBold = 95% interval (resampling whole days) wholly on the good side. {n_cmp} cells are shown, so about "
                 f"{n_cmp * 0.025:.0f} would be bold by chance alone; a real edge should show up at neighbouring "
                 "horizons and intervals and in the confirmation block, not in one cell. Cells of the same day are correlated, "
                 "and different horizons of one origin are too.")


def interval_detail(df, iv, lines):
    steps = sorted(df.step.unique())
    lines.append(f"\n## Interval {iv} min: detail\n")
    for st in steps:
        table(df[df.step == st], f"Interval {iv} min, step {st} (forecast {st * iv} minute(s) ahead)", lines)

    lines.append(f"\n### Direction baselines, interval {iv} min (hit rate on bars that moved; 50% = chance)\n")
    lines.append("| step | momentum (last 5 bars) | mean-reversion | always up |")
    lines.append("|---|---|---|---|")
    base = df[df.variant == df.variant.iloc[0]]
    for st in steps:
        g = base[base.step == st]
        mv = g[g.sg_actual != 0]
        mm = mv[mv.sg_mom5 != 0]
        ds_m = mm.assign(h=(mm.sg_mom5 == mm.sg_actual).astype(int), n=1).groupby("day")[["h", "n"]].sum()
        ds_r = mm.assign(h=(mm.sg_mom5 == -mm.sg_actual).astype(int), n=1).groupby("day")[["h", "n"]].sum()
        ds_u = mv.assign(h=(mv.sg_actual > 0).astype(int), n=1).groupby("day")[["h", "n"]].sum()
        lines.append(f"| {st} | {pct(boot_ratio(ds_m.h, ds_m.n))} | {pct(boot_ratio(ds_r.h, ds_r.n))} | {pct(boot_ratio(ds_u.h, ds_u.n))} |")

    rs = ref_step(df, iv)
    days = sorted(df.day.unique())
    cut = days[int(len(days) * 2 / 3)] if len(days) >= 6 else None
    if cut:
        lines.append(f"\n### Interval {iv} min: select on first two thirds of days, confirm once on the last third "
                     f"(cut at {cut}), step {rs} (+{rs * iv} min)\n")
        lines.append("| variant | MAE ratio (selection) | MAE ratio (confirmation) | pinball ratio (selection) | pinball ratio (confirmation) |")
        lines.append("|---|---|---|---|---|")
        gr = df[df.step == rs]
        for v, g in gr.groupby("variant"):
            row = []
            for part in (g[g.day < cut], g[g.day >= cut]):
                ds = day_sums(part, ["abs_m", "abs_nc", "pin_m", "pin_vb"])
                row.append((boot_ratio(ds.abs_m, ds.abs_nc), boot_ratio(ds.pin_m, ds.pin_vb)))
            lines.append(f"| {v} | {f3(row[0][0])} | {f3(row[1][0])} | {f3(row[0][1])} | {f3(row[1][1])} |")

    lines.append(f"\n### Interval {iv} min: MAE ratio by time of day, step {rs} (+{rs * iv} min)\n")
    gr = df[df.step == rs].copy()
    hh = gr.hhmm.str[:2].astype(int) * 60 + gr.hhmm.str[3:].astype(int)
    gr["bucket"] = pd.cut(hh, [0, 11 * 60, 13 * 60, 15 * 60 + 1], right=False, labels=["before 11:00", "11:00-13:00", "13:00 on"])
    lines.append("| variant | before 11:00 | 11:00-13:00 | 13:00 on |")
    lines.append("|---|---|---|---|")
    for v, g in gr.groupby("variant"):
        cells = []
        for bk in ["before 11:00", "11:00-13:00", "13:00 on"]:
            ds = day_sums(g[g.bucket == bk], ["abs_m", "abs_nc"])
            cells.append(f3(boot_ratio(ds.abs_m, ds.abs_nc)))
        lines.append(f"| {v} | " + " | ".join(cells) + " |")


def summarize(tag, detail=False):
    d = OUT_ROOT / tag
    df = _load_rows(tag)
    ivs = sorted(df.interval.unique())
    head = [f"# Paired evaluation: {tag}\n",
            f"Variants: {', '.join(sorted(df.variant.unique()))}. Intervals: {', '.join(str(i) + ' min' for i in ivs)}. "
            f"Sessions: {df.day.nunique()}.",
            "Ratios < 1 mean the model beats the baseline. Intervals are 95%, resampling whole days. "
            "Overlapping windows are handled by the day-level resampling, but all numbers are still correlated within a day.",
            "Horizons are in wall-clock minutes (= step x interval) so different intervals can be compared directly."]
    headline(df, head)
    body = []
    for iv in ivs:
        interval_detail(df[df.interval == iv], iv, body)
    body.append("\nNote: select/confirm and time-of-day tables use the step nearest to "
                f"{REF_MINUTES} minutes ahead. With this many comparisons, expect some to look better than 1.0 by chance; "
                "only trust a gain that also holds in the confirmation block and across horizons.")
    full = "\n".join(head + body)
    (d / "summary.md").write_text(full, encoding="utf-8")
    print(full if detail else "\n".join(head))
    print(f"\nSaved {d / 'summary.md'}" + ("" if detail else "  (full per-interval detail inside; print it with --detail)"))


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
                           every=30, ctx=128, batch=8, variants=None, include_retired=False, intervals=list(INTERVALS),
                           steps=None)
    run(a, runner=_MockRunner(), bars=bars)
    rows_path = OUT_ROOT / "selftest" / "rows.csv"
    df = pd.read_csv(rows_path)
    ok = []

    def check(name, cond, detail=""):
        print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
        ok.append(bool(cond))

    check("every interval produced rows", sorted(df.interval.unique()) == list(INTERVALS), str(sorted(df.interval.unique())))
    check("only active variants ran", set(df.variant) == set(VARIANTS), str(sorted(df.variant.unique())))
    check("minutes = step x interval on every row", bool((df.minutes == df.step * df.interval).all()))
    check("scored steps match STEPS_BY_INTERVAL",
          all(tuple(sorted(df[df.interval == iv].step.unique())) == STEPS_BY_INTERVAL[iv] for iv in INTERVALS))
    on_grid = df.apply(lambda r: ((int(r.hhmm[:2]) * 60 + int(r.hhmm[3:])) - (9 * 60 + 15)) % r.interval == 0, axis=1)
    check("every origin lies on its interval's bar grid", bool(on_grid.all()))
    check("no NaN in model or baseline columns", not df[["abs_m", "abs_nc", "pin_m", "pin_vb"]].isna().any().any())
    # resume: a second identical run must add nothing
    n0 = len(df)
    run(a, runner=_MockRunner(), bars=bars)
    check("resume adds no duplicate rows", len(pd.read_csv(rows_path)) == n0)
    # retired variants are refused, and allowed with the override
    try:
        _resolve_variants(argparse.Namespace(variants=["E_raw_tod"], include_retired=False))
        refused = False
    except SystemExit:
        refused = True
    check("retired variant is refused by default", refused)
    check("retired variant allowed with --include-retired",
          _resolve_variants(argparse.Namespace(variants=["E_raw_tod"], include_retired=True)) == ["E_raw_tod"])
    summarize("selftest", detail=True)
    print("\n==================== RESULT SUMMARY (paired_eval selftest) ====================")
    print(f"{sum(ok)}/{len(ok)} checks passed")
    print("ALL PASS" if all(ok) else "SOME FAILED")
    if not all(ok):
        sys.exit(1)


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
    r.add_argument("--every", type=int, default=30,
                   help="minutes between forecast origins; must be a multiple of every interval run")
    r.add_argument("--intervals", type=int, nargs="+", default=list(INTERVALS), choices=list(INTERVALS),
                   help="bar sizes in minutes (default: all of %s)" % (list(INTERVALS),))
    r.add_argument("--ctx", type=int, default=0,
                   help="context length in BARS for every interval (0 = per-interval default: %s)"
                        % {k: v["context_len"] for k, v in eg.INTERVAL_DEFAULTS.items()})
    r.add_argument("--steps", type=int, nargs="+", default=None,
                   help="steps (in bars) to score, applied to every interval (default: per-interval, see STEPS_BY_INTERVAL)")
    r.add_argument("--batch", type=int, default=16)
    r.add_argument("--variants", nargs="+", default=None, help="default: %s" % list(VARIANTS))
    r.add_argument("--include-retired", action="store_true",
                   help="also allow variants retired after the 1-minute study: %s" % list(RETIRED))
    s = sub.add_parser("summarize")
    s.add_argument("--tag", required=True)
    s.add_argument("--detail", action="store_true", help="also print the per-interval detail tables (always saved to summary.md)")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    elif a.cmd == "summarize":
        summarize(a.tag, detail=a.detail)
    else:
        selftest()


if __name__ == "__main__":
    main()
