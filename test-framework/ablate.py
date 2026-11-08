"""
Long-run ablation for TimesFM trading terminal.
Tests days x start times x bar intervals x input forms x covariates. Model is loaded only once.

Bar interval: --intervals picks the bar sizes (1, 5, 10, 15, 30 minutes). Horizon and run length are counted in
BARS of that interval (see HORIZON_BARS / STEPS_BARS below). Start times are on the 30-minute grid from 09:15 so
they are valid bar boundaries for every interval.

What the two MAE ratios mean (both are model MAE / no-change MAE, < 1 = model wins):
  latest  = each bar is forecast ONE bar ahead from the previous origin (lag 0), vs the previous bar's close.
  initial = the frozen forecast made at the start, up to HORIZON bars ahead, vs the close at the start.
The "latest" ratio does not depend on the horizon at all, so the old horizon loop (5/10/15/30) re-measured the same
thing four times. Use paired_eval.py for rigorous multi-horizon scoring; this script is the quick screen.

Usage:
  python ablate.py                                  # all intervals, default days / start times
  python ablate.py --intervals 5 15 30              # subset
  python ablate.py --analyse data/ablation_intervals_<stamp>.json   # re-read saved results, no model needed
"""
import argparse
import json
import statistics
from collections import defaultdict
from datetime import date, datetime
from itertools import product
from pathlib import Path

import engine as eg
import metrics as mt

# Bars per forecast and bars revealed per run, by interval (revealed bars are the evaluation window).
HORIZON_BARS = {k: v["horizon"] for k, v in eg.INTERVAL_DEFAULTS.items()}      # 1:10, 5:6, 10:3, 15:3, 30:3
STEPS_BARS = {1: 30, 5: 12, 10: 6, 15: 4, 30: 4}
# On the 30-minute grid from 09:15, so every start is a bar boundary for 1, 5, 10, 15 and 30 minute bars.
START_TIMES = ["10:15", "11:15", "12:15", "13:15", "14:15"]
FORMS = ["raw_close", "log_price_stitched", "log_returns"]
COVS = ["none", "time_of_day", "volume", "tod_volume"]

# Combinations not worth retrying, from the 1-minute paired study NIFTY50_20261107_2044 (see RETIRED in paired_eval.py).
PRUNED = [
    (lambda form, cov: form == "log_returns",
     "log_returns: its P10-P90 band is an approximation (coverage 98-100%, pinball ratio 1.2-2.1 vs the vol band)"),
    (lambda form, cov: (form, cov) == ("raw_close", "time_of_day"),
     "raw_close + time_of_day: MAE ratio 1.018 [1.008, 1.030] at +1 min and 1.020 [1.004, 1.037] at +30 min, worse than no-change"),
]


def pruned_reason(form, cov):
    for pred, why in PRUNED:
        if pred(form, cov):
            return why
    return None


def _ratio(a, b):
    return (a / b) if (a is not None and b) else None


def run_one(runner, sym, day, start_hhmm, interval, horizon, steps, settings):
    base = {"day": str(day), "start": start_hhmm, "interval": interval, "horizon": horizon,
            "steps": steps, "settings": settings}
    try:
        run_dir, snaps, rp, T0 = eg.run_simulation(
            runner, sym, day, start_hhmm,
            horizon=horizon, n_steps=steps, settings=settings,
            log=lambda *a, **k: None,  # silence
        )
        result = mt.compare(snaps, rp, lag=0)
        s = result["summary_all"]
        return {
            **base,
            "mae_initial": s["mae"]["initial"],
            "mae_latest": s["mae"]["latest"],
            "mae_nc": s["mae"]["nc_roll"],
            "mae_ratio_initial": s["mae_ratio"]["initial"],
            "mae_ratio_latest": s["mae_ratio"]["latest"],
            "dir_latest": s["direction"]["latest"],
            "coverage_latest": s["coverage"]["latest"],
            "pinball_latest": s["pinball"]["latest"],
            "pinball_ratio_latest": _ratio(s["pinball"]["latest"], s["pinball_vb"]["latest"]),
            "run_dir": str(run_dir),
            "status": "ok",
        }
    except (Exception, SystemExit) as e:
        return {**base, "status": "failed", "error": str(e)}


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def _f(v, w=5):
    return f"{v:.3f}".rjust(w) if v is not None else "n/a".rjust(w)


def analyse(results):
    ok = [r for r in results if r["status"] == "ok" and r.get("mae_ratio_latest") is not None]
    if not ok:
        print("no successful runs to analyse")
        return

    print("\n==================== TOP 15 RUNS (by latest MAE ratio, 1 bar ahead) ====================")
    for r in sorted(ok, key=lambda x: x["mae_ratio_latest"])[:15]:
        s = r["settings"]
        print(f"{r['day']} {r['start']} | {r['interval']:2d}m H={r['horizon']:2d} | "
              f"{s['input_form']:18s} | {s['covariates']:11s} -> "
              f"latest {r['mae_ratio_latest']:.3f} | initial {_f(r.get('mae_ratio_initial')).strip()} | "
              f"dir {r['dir_latest']} | pinball/vb {_f(r.get('pinball_ratio_latest')).strip()}")

    print("\n---------- Average by bar interval (lower MAE / pinball ratio = better; 1.000 = no edge) ----------")
    print(f"{'interval':>9} | {'MAE latest':>10} | {'MAE initial':>11} | {'pinball/vb':>10} | runs")
    by_iv = defaultdict(list)
    for r in ok:
        by_iv[r["interval"]].append(r)
    for iv in sorted(by_iv):
        rs = by_iv[iv]
        print(f"{str(iv) + ' min':>9} | {_f(_mean([r['mae_ratio_latest'] for r in rs]), 10)} | "
              f"{_f(_mean([r.get('mae_ratio_initial') for r in rs]), 11)} | "
              f"{_f(_mean([r.get('pinball_ratio_latest') for r in rs]), 10)} | {len(rs)}")

    for label, key in (("input form", "input_form"), ("covariates", "covariates")):
        print(f"\n---------- Average latest MAE ratio by {label} ----------")
        g = defaultdict(list)
        for r in ok:
            g[r["settings"][key]].append(r["mae_ratio_latest"])
        for k, vals in sorted(g.items(), key=lambda kv: statistics.mean(kv[1])):
            print(f"{k:20s}: {statistics.mean(vals):.3f}  (n={len(vals)})")

    print("\n---------- Best combinations per interval (average latest MAE ratio) ----------")
    for iv in sorted(by_iv):
        g = defaultdict(list)
        for r in by_iv[iv]:
            g[(r["settings"]["input_form"], r["settings"]["covariates"])].append(r)
        ranked = sorted(g.items(), key=lambda kv: _mean([r["mae_ratio_latest"] for r in kv[1]]))
        print(f"{iv} min:")
        for (form, cov), rs in ranked[:3]:
            print(f"   {form:18s} + {cov:11s} latest {_f(_mean([r['mae_ratio_latest'] for r in rs]))} | "
                  f"initial {_f(_mean([r.get('mae_ratio_initial') for r in rs]))} | "
                  f"pinball/vb {_f(_mean([r.get('pinball_ratio_latest') for r in rs]))}  (n={len(rs)})")
    print("\nThis is a screen over a few days: differences of ~0.01 are noise. Confirm any candidate with paired_eval.py.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NIFTY50")
    ap.add_argument("--days", nargs="+", default=None, help="YYYY-MM-DD (default: 2026-11-05 2026-11-06)")
    ap.add_argument("--starts", nargs="+", default=START_TIMES, help="HH:MM, on the 30-min grid from 09:15")
    ap.add_argument("--intervals", type=int, nargs="+", default=list(eg.INTERVALS), choices=list(eg.INTERVALS))
    ap.add_argument("--analyse", metavar="JSON", help="analyse a saved results file and exit (no model needed)")
    a = ap.parse_args()

    if a.analyse:
        analyse(json.loads(Path(a.analyse).read_text()))
        return

    sym = a.symbol
    days = [date.fromisoformat(d) for d in a.days] if a.days else [date(2026, 11, 5), date(2026, 11, 6)]
    for st in a.starts:                       # a start off the grid would be silently shifted by the engine
        t = eg.make_ts(days[0], st)
        if eg.snap_to_grid(t, 30) != t:
            raise SystemExit(f"start {st} is not on the 30-min grid from 09:15, so it is not a bar boundary for 10/30-min bars")

    # Build all combinations, minus the pruned ones
    combos, skipped = [], defaultdict(int)
    for day, start, iv, form, cov in product(days, a.starts, a.intervals, FORMS, COVS):
        why = pruned_reason(form, cov)
        if why:
            skipped[why] += 1
        else:
            combos.append((day, start, iv, form, cov))
    total = len(combos)
    print(f"Total combinations to run: {total}")
    for why, n in skipped.items():
        print(f"  skipped {n:4d}: {why}")

    print("\nLoading model (only once)...")
    runner = eg.ModelRunner()
    print("Model loaded.\n")

    results = []
    out_path = Path(f"data/ablation_intervals_{datetime.now():%Y%m%d_%H%M}.json")
    out_path.parent.mkdir(exist_ok=True)

    for i, (day, start, iv, form, cov) in enumerate(combos, 1):
        horizon, steps = HORIZON_BARS[iv], STEPS_BARS[iv]
        settings = {"context_len": eg.INTERVAL_DEFAULTS[iv]["context_len"], "interval": iv,
                    "input_form": form, "covariates": cov}

        print(f"[{i:3d}/{total}] {day} {start} | {iv:2d}m H={horizon:2d} | {form:18s} | {cov:11s}", end=" -> ", flush=True)

        r = run_one(runner, sym, day, start, iv, horizon, steps, settings)
        results.append(r)

        if r["status"] == "ok" and r.get("mae_ratio_latest") is not None:
            print(f"latest={r['mae_ratio_latest']:.3f} | dir={r['dir_latest']} | cov={r['coverage_latest']}")
        elif r["status"] == "ok":
            print("ok (no ratio)")
        else:
            print(f"FAILED: {r.get('error')}")

        # Save progress after every run (so you don't lose data if it crashes)
        out_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"\nFinished. Saved {len(results)} results -> {out_path}")
    analyse(results)


if __name__ == "__main__":
    main()
