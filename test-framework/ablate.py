"""
Long-run ablation for TimesFM trading terminal.
Tests multiple days × start times × horizons × input forms × covariates.
Model is loaded only once.
"""

import json
from datetime import date, time
from pathlib import Path
from itertools import product

import engine as eg
import metrics as mt


def run_one(runner, sym, day, start_hhmm, horizon, steps, settings):
    try:
        run_dir, snaps, rp, T0 = eg.run_simulation(
            runner,
            sym,
            day,
            start_hhmm,
            horizon=horizon,
            n_steps=steps,
            settings=settings,
            log=lambda *a, **k: None,  # silence
        )
        result = mt.compare(snaps, rp, lag=0)
        s = result["summary_all"]

        return {
            "day": str(day),
            "start": start_hhmm,
            "horizon": horizon,
            "steps": steps,
            "settings": settings,
            "mae_initial": s["mae"]["initial"],
            "mae_latest": s["mae"]["latest"],
            "mae_nc": s["mae"]["nc_roll"],
            "mae_ratio_latest": s["mae_ratio"]["latest"],
            "dir_latest": s["direction"]["latest"],
            "coverage_latest": s["coverage"]["latest"],
            "pinball_latest": s["pinball"]["latest"],
            "run_dir": str(run_dir),
            "status": "ok",
        }
    except Exception as e:
        return {
            "day": str(day),
            "start": start_hhmm,
            "horizon": horizon,
            "steps": steps,
            "settings": settings,
            "status": "failed",
            "error": str(e),
        }


def main():
    # ====================== CONFIGURE HERE ======================
    sym = "NIFTY50"

    days = [
        date(2026, 11, 5),
        date(2026, 11, 6),
        # date(2026, 11, 7),
        # date(2026, 11, 4),
    ]

    # Different start times across the day
    start_times = ["10:00", "10:30", "11:30", "13:00", "14:00"]

    # Different forecast horizons
    horizons = [5, 10, 15, 30]

    # How many minutes to roll forward (evaluation window)
    # For longer horizons, keep steps reasonable
    steps_map = {
        5: 30,
        10: 30,
        15: 30,
        30: 45,
    }

    forms = ["raw_close", "log_price_stitched", "log_returns"]
    covs = ["none", "time_of_day", "volume", "tod_volume"]

    # ============================================================

    print("Loading model (only once)...")
    runner = eg.ModelRunner()
    print("Model loaded.\n")

    # Build all combinations
    combos = list(product(days, start_times, horizons, forms, covs))
    total = len(combos)
    print(f"Total combinations to run: {total}\n")

    results = []
    out_path = Path("data/ablation_long_results.json")
    out_path.parent.mkdir(exist_ok=True)

    for i, (day, start, horizon, form, cov) in enumerate(combos, 1):
        steps = steps_map.get(horizon, 30)
        settings = {
            "context_len": 2048,
            "input_form": form,
            "covariates": cov,
        }

        print(f"[{i:3d}/{total}] {day} {start} | H={horizon:2d} | {form:20s} | {cov:12s}", end=" → ", flush=True)

        r = run_one(runner, sym, day, start, horizon, steps, settings)
        results.append(r)

        if r["status"] == "ok":
            ratio = r.get("mae_ratio_latest")
            print(f"ratio={ratio:.3f} | dir={r['dir_latest']} | cov={r['coverage_latest']}")
        else:
            print(f"FAILED: {r.get('error')}")

        # Save progress after every run (so you don't lose data if it crashes)
        out_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"\nFinished. Saved {len(results)} results → {out_path}")

    # ---------- Quick analysis ----------
    ok_results = [r for r in results if r["status"] == "ok" and r.get("mae_ratio_latest") is not None]

    if ok_results:
        print("\n==================== TOP 15 COMBINATIONS (by MAE ratio) ====================")
        sorted_res = sorted(ok_results, key=lambda x: x["mae_ratio_latest"])
        for r in sorted_res[:15]:
            s = r["settings"]
            print(
                f"{r['day']} {r['start']} | H={r['horizon']:2d} | "
                f"{s['input_form']:20s} | {s['covariates']:12s} → "
                f"ratio {r['mae_ratio_latest']:.3f} | "
                f"dir {r['dir_latest']} | pinball {r['pinball_latest']:.2f}"
            )

        # Average by input_form
        print("\n---------- Average MAE Ratio by Input Form ----------")
        from collections import defaultdict
        import statistics

        by_form = defaultdict(list)
        for r in ok_results:
            by_form[r["settings"]["input_form"]].append(r["mae_ratio_latest"])

        for form, vals in sorted(by_form.items(), key=lambda x: statistics.mean(x[1])):
            print(f"{form:20s}: {statistics.mean(vals):.3f}  (n={len(vals)})")

        # Average by covariates
        print("\n---------- Average MAE Ratio by Covariates ----------")
        by_cov = defaultdict(list)
        for r in ok_results:
            by_cov[r["settings"]["covariates"]].append(r["mae_ratio_latest"])

        for cov, vals in sorted(by_cov.items(), key=lambda x: statistics.mean(x[1])):
            print(f"{cov:12s}: {statistics.mean(vals):.3f}  (n={len(vals)})")


if __name__ == "__main__":
    main()