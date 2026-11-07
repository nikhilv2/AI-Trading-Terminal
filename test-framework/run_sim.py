"""M2 command-line simulation: initial forecast + one update per revealed minute."""
import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np

import engine as eg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NIFTY50")
    ap.add_argument("--date", help="YYYY-MM-DD (default: latest cached day)")
    ap.add_argument("--time", default="10:30", help="HH:MM start (forecast origin)")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--steps", type=int, default=10, help="minutes to reveal")
    ap.add_argument("--ctx", type=int, default=2048)
    a = ap.parse_args()
    day = date.fromisoformat(a.date) if a.date else eg.latest_cached_day(a.symbol)

    t = time.time()
    runner = eg.ModelRunner()
    print(f"model loaded in {time.time() - t:.1f}s")
    print(f"run: {a.symbol} {day} start {a.time} horizon {a.horizon} steps {a.steps} ctx {a.ctx}")
    run_dir, snaps, rp, T0 = eg.run_simulation(runner, a.symbol, day, a.time, a.horizon, a.steps,
                                               {"context_len": a.ctx})

    H = a.horizon
    last_close = float(rp.context(T0, 1)["close"].iloc[-1])
    ini = snaps[0]
    print("\ntarget  init_P10  init_Q50  init_P90  latest_Q50  actual   no_change  in_band")
    e_ini, e_lat, e_nc, inband, n = [], [], [], 0, 0
    for j in range(min(H, len(snaps))):
        bar = rp.bar_opening_at(T0 + j * eg.MIN)
        if bar is None:
            break
        actual = float(bar["close"])
        lat = snaps[j]["median"][0]          # forecast made when the bar opened (origin T0+j)
        p10, q50, p90 = ini["quantiles"][j][0], ini["median"][j], ini["quantiles"][j][8]
        ib = p10 <= actual <= p90
        inband += ib
        n += 1
        e_ini.append(abs(q50 - actual)); e_lat.append(abs(lat - actual)); e_nc.append(abs(last_close - actual))
        print(f"{(T0 + j * eg.MIN):%H:%M}  {p10:9.2f} {q50:9.2f} {p90:9.2f} {lat:11.2f} {actual:9.2f} {last_close:10.2f}  {ib}")

    problems = eg.verify_run(run_dir)
    sha0_now = json.loads((Path(run_dir) / "snap_0000.json").read_text())["sha256"]
    print("\n==================== RESULT SUMMARY (M2 run) ====================")
    print("run dir:", run_dir)
    print(f"snapshots: {len(snaps)} (seq 0 = initial), verify problems: {problems or 'none'}")
    print("initial snapshot unchanged after run:", sha0_now == ini["sha256"])
    print("infer ms (median/max):", float(np.median([s['infer_ms'] for s in snaps])),
          max(s['infer_ms'] for s in snaps))
    print("quantile crossings in any snapshot:", any(s["quantile_crossing"] for s in snaps))
    print("ForecastOutput info:", runner.output_info)
    if n:
        print(f"quick MAE over {n} targets: initial {np.mean(e_ini):.2f} | latest {np.mean(e_lat):.2f} | "
              f"no-change {np.mean(e_nc):.2f}; initial P10-P90 coverage {inband}/{n}")
    print("=================================================================")


if __name__ == "__main__":
    main()