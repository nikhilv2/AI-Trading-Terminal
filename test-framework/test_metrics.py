"""M4 self-test: metrics.compare on synthetic data with hand-computed answers. No model, no network."""
from datetime import date

import pandas as pd

import engine as eg
import metrics as mt

results = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    results.append(bool(ok))


def near(a, b, tol=1e-9):
    return a is not None and abs(a - b) <= tol


def make_rp():
    day = date(2026, 11, 6)
    t0 = eg.make_ts(day, "09:15")
    opens = [t0 + i * eg.MIN for i in range(375)]
    close = [100.0 + i for i in range(375)]               # bar i closes at 100+i
    df = pd.DataFrame({"ts_open": opens, "ts_close": [o + eg.MIN for o in opens],
                       "open": close, "high": close, "low": close, "close": close,
                       "volume": 0.0, "oi": 0.0})
    return eg.Replay(df), eg.make_ts(day, "10:00")        # T0=10:00: last closed bar opened 09:59 (close 144)


def make_snaps(T0, n_rev, horizon, lat):
    snaps = []
    for k in range(n_rev + 1):
        origin = T0 + k * eg.MIN
        m0 = 147.0 if k == 0 else lat[k]                  # initial: median = actual + 2
        med = [m0 + i for i in range(horizon)]
        q = [[m + (c - 4) * 1.0 for c in range(9)] for m in med]    # P10 = m-4 ... P90 = m+4
        snaps.append({"seq": k, "kind": "initial" if k == 0 else "update", "origin_ts": origin.isoformat(),
                      "horizon": horizon,
                      "target_open_ts": [(origin + i * eg.MIN).isoformat() for i in range(horizon)],
                      "quantiles": q, "median": med})
    return snaps


rp, T0 = make_rp()
lat = {1: 145.5, 2: 145.0, 3: 158.0, 4: 148.5, 5: 149.5}
snaps = make_snaps(T0, 5, 10, lat)
res = mt.compare(snaps, rp, lag=0)
rows, sm, si = res["rows"], res["summary_all"], res["summary_indep"]

check("pinball(0,[1]*9) = 0.5", near(mt.pinball(0.0, [1.0] * 9), 0.5))
check("pinball(2,[1]*9) = 0.5", near(mt.pinball(2.0, [1.0] * 9), 0.5))
check("5 revealed bars -> 5 rows (6th bar not used)", len(rows) == 5, f"{len(rows)} rows")
check("anchor at T0 = close of the 09:59 bar = 144", near(res["anchor_T0"], 144.0))
check("actuals are 145..149", [r["actual"] for r in rows] == [145.0, 146.0, 147.0, 148.0, 149.0])
check("only row 0 has latest == initial", [r["latest_is_initial"] for r in rows] == [True, False, False, False, False])
check("latest for row j was made at origin T0+j",
      all(pd.Timestamp(r["latest_origin"]) == T0 + r["j"] * eg.MIN for r in rows))
check("every scored bar had closed by the last origin",
      all(pd.Timestamp(r["target"]) + eg.MIN <= T0 + 5 * eg.MIN for r in rows))
check("MAE initial = 2", near(sm["mae"]["initial"], 2.0))
check("MAE no-change fixed = 3", near(sm["mae"]["nc_fixed"], 3.0))
check("MAE latest = 3", near(sm["mae"]["latest"], 3.0))
check("MAE no-change rolling = 1", near(sm["mae"]["nc_roll"], 1.0))
check("MAE ratios = 2/3 and 3", near(sm["mae_ratio"]["initial"], 2 / 3) and near(sm["mae_ratio"]["latest"], 3.0))
check("coverage initial 5/5, latest 4/5", sm["coverage"]["initial"] == [5, 5] and sm["coverage"]["latest"] == [4, 5])
check("direction initial 5/5, latest 4/5", sm["direction"]["initial"] == [5, 5] and sm["direction"]["latest"] == [4, 5])
check("pinball initial = 6/9", near(sm["pinball"]["initial"], 6.0 / 9.0))
check("excl. shared row: MAE initial 2, nc_fixed 3.5, latest 3.25, nc_roll 1",
      near(si["mae"]["initial"], 2.0) and near(si["mae"]["nc_fixed"], 3.5)
      and near(si["mae"]["latest"], 3.25) and near(si["mae"]["nc_roll"], 1.0))
check("excl. shared row: coverage latest 3/4, direction latest 3/4",
      si["coverage"]["latest"] == [3, 4] and si["direction"]["latest"] == [3, 4])

# no look-ahead in the table: poison every bar that closes after the last origin
df2 = rp.bars.copy()
m = df2["ts_close"] > T0 + 5 * eg.MIN
df2.loc[m, "close"] = 1e9
res_p = mt.compare(snaps, eg.Replay(df2), lag=0)
check("poison is armed", int(m.sum()) > 0, f"{int(m.sum())} rows poisoned")
check("table identical when later bars are poisoned", res_p["rows"] == rows)

# lag=1: latest = forecast made one bar before the previous origin
r1 = mt.compare(snaps, rp, lag=1)["rows"]
check("lag 1: row 1 uses the initial snapshot, step 1 (148)", near(r1[1]["latest_q50"], 148.0) and r1[1]["latest_is_initial"])
check("lag 1: row 2 uses snapshot 1, step 1 (146.5), origin T0+1",
      near(r1[2]["latest_q50"], 146.5) and pd.Timestamp(r1[2]["latest_origin"]) == T0 + eg.MIN)

# horizon shorter than the run: initial forecast covers only 3 rows
r3 = mt.compare(make_snaps(T0, 5, 3, lat), rp, lag=0)
check("horizon 3: initial blank from row 3", [x["init_q50"] is None for x in r3["rows"]] == [False, False, False, True, True])
check("horizon 3: n_initial 3, n_latest 5", r3["summary_all"]["n_initial"] == 3 and r3["summary_all"]["n_latest"] == 5)

print("\n==================== RESULT SUMMARY (M4 test) ====================")
print(f"{sum(results)}/{len(results)} checks passed")
print("ALL PASS" if all(results) else "SOME FAILED")
print("==================================================================")