"""M5b: comparison table + summary metrics from STORED snapshots and REVEALED bars.
Pure functions (no model, no network). Includes the volatility-band baseline."""
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from upstox_data import IST

MIN = pd.Timedelta(minutes=1)
QLEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
P10, MED, P90 = 0, 4, 8
Z = np.array([NormalDist().inv_cdf(float(p)) for p in QLEVELS])
VOL_N = 60        # up to this many recent 1-minute returns
VOL_MIN = 10      # fewer than this -> baseline unavailable (None)


def load_snaps(run_dir):
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(Path(run_dir).glob("snap_*.json"))]


def pinball(y, q):
    """Mean pinball loss over the 9 quantile levels for one observation y."""
    d = y - np.asarray(q, dtype=np.float64)
    return float(np.mean(np.maximum(QLEVELS * d, (QLEVELS - 1.0) * d)))


def _sgn(x):
    return (x > 0) - (x < 0)


def _dir(ref, fc, actual):
    """True = direction hit, False = miss, None = the actual did not move (tie, excluded)."""
    mv = actual - ref
    if mv == 0:
        return None
    return _sgn(fc - ref) == _sgn(mv)


def vol_band(rp, T, steps):
    """Volatility-band baseline made at origin T from bars CLOSED at or before T.
    Median = last close. Spread = RMS of the last up to VOL_N same-session 1-minute log returns
    x sqrt(h), h = 1..steps, normal quantiles, linear in price.
    Returns (anchor, q) with q shape (steps, 9), or (anchor, None) if fewer than VOL_MIN returns."""
    ctx = rp.context(T, VOL_N + 1)
    c = ctx["close"].to_numpy(dtype=np.float64)
    anchor = float(c[-1])
    d = ctx["ts_open"].dt.date.to_numpy()
    same = d[1:] == d[:-1]
    r = np.log(c[1:] / c[:-1])[same]
    if len(r) < VOL_MIN:
        return anchor, None
    sigma = float(np.sqrt(np.mean(r ** 2)))
    h = np.sqrt(np.arange(1, steps + 1, dtype=np.float64))[:, None]
    return anchor, anchor * (1.0 + sigma * h * Z[None, :])


def compare(snaps, rp, lag=0):
    """snaps: snapshots of ONE run in seq order (snaps[0] = initial). rp: engine.Replay.
    Row j = bar opening at T0+j. 'latest' = snapshot with seq max(j-lag,0), step j-seq."""
    if len(snaps) < 2:
        raise ValueError("need the initial snapshot plus at least one update")
    ini = snaps[0]
    if ini["kind"] != "initial":
        raise ValueError("snaps[0] is not the initial snapshot")
    T0 = pd.Timestamp(ini["origin_ts"]).tz_convert(IST)
    n_rev = len(snaps) - 1
    T_last = T0 + n_rev * MIN                      # last forecast origin = reveal horizon
    anchor0 = float(rp.context(T0, 1)["close"].iloc[-1])
    _, vb0 = vol_band(rp, T0, ini["horizon"])
    rows = []
    for j in range(n_rev):
        t = T0 + j * MIN
        bar = rp.bar_opening_at(t)
        if bar is None or bar["ts_close"] > T_last:   # never score a bar that has not been revealed
            break
        actual = float(bar["close"])

        if j < ini["horizon"]:
            qi = ini["quantiles"][j]
            i10, i50, i90 = qi[P10], qi[MED], qi[P90]
            e_i, in_i, d_i, p_i = i50 - actual, bool(i10 <= actual <= i90), _dir(anchor0, i50, actual), pinball(actual, qi)
            if vb0 is not None:
                v = vb0[j]
                vi10, vi90 = float(v[P10]), float(v[P90])
                in_vi, pv_i = bool(vi10 <= actual <= vi90), pinball(actual, v)
            else:
                vi10 = vi90 = in_vi = pv_i = None
        else:
            i10 = i50 = i90 = e_i = in_i = d_i = p_i = None
            vi10 = vi90 = in_vi = pv_i = None

        seq = max(j - lag, 0)
        s = snaps[seq]
        idx = j - seq
        if idx >= s["horizon"]:
            raise ValueError(f"snapshot {seq} horizon {s['horizon']} too short for step {idx}")
        if pd.Timestamp(s["target_open_ts"][idx]) != t:
            raise ValueError(f"snapshot {seq} step {idx} does not point at {t}")
        ql = s["quantiles"][idx]
        l10, l50, l90 = ql[P10], ql[MED], ql[P90]
        T_or = T0 + seq * MIN
        anchor_roll, vbl = vol_band(rp, T_or, idx + 1)
        if vbl is not None:
            v = vbl[idx]
            vl10, vl90 = float(v[P10]), float(v[P90])
            in_vl, pv_l = bool(vl10 <= actual <= vl90), pinball(actual, v)
        else:
            vl10 = vl90 = in_vl = pv_l = None

        rows.append({
            "j": j, "target": t.isoformat(), "actual": actual,
            "init_p10": i10, "init_q50": i50, "init_p90": i90,
            "latest_p10": l10, "latest_q50": l50, "latest_p90": l90,
            "latest_origin": T_or.isoformat(), "latest_seq": seq, "latest_is_initial": seq == 0,
            "nc_fixed": anchor0, "nc_roll": anchor_roll,
            "err_init": e_i, "err_latest": l50 - actual,
            "err_nc_fixed": anchor0 - actual, "err_nc_roll": anchor_roll - actual,
            "in_init": in_i, "in_latest": bool(l10 <= actual <= l90),
            "dir_init": d_i, "dir_latest": _dir(anchor_roll, l50, actual),
            "pin_init": p_i, "pin_latest": pinball(actual, ql),
            "vb_init_p10": vi10, "vb_init_p90": vi90, "in_vb_init": in_vi, "pin_vb_init": pv_i,
            "vb_latest_p10": vl10, "vb_latest_p90": vl90, "in_vb_latest": in_vl, "pin_vb_latest": pv_l})

    return {"rows": rows, "lag": lag, "origin": T0.isoformat(), "n_revealed": len(rows),
            "anchor_T0": anchor0,
            "summary_all": summarize(rows),
            "summary_indep": summarize([r for r in rows if not r["latest_is_initial"]])}


def _m(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if len(v) else None


def _frac(flags):
    f = [x for x in flags if x is not None]
    return [int(sum(f)), len(f)]


def summarize(rows):
    ri = [r for r in rows if r["init_q50"] is not None]      # rows the initial forecast covers
    mae_i = _m([abs(r["err_init"]) for r in ri])
    mae_nf = _m([abs(r["err_nc_fixed"]) for r in ri])        # same rows as initial
    mae_l = _m([abs(r["err_latest"]) for r in rows])
    mae_nr = _m([abs(r["err_nc_roll"]) for r in rows])
    ratio = lambda a, b: (a / b) if (a is not None and b) else None
    return {
        "n_initial": len(ri), "n_latest": len(rows),
        "mae": {"initial": mae_i, "latest": mae_l, "nc_fixed": mae_nf, "nc_roll": mae_nr},
        "mae_ratio": {"initial": ratio(mae_i, mae_nf), "latest": ratio(mae_l, mae_nr)},
        "direction": {"initial": _frac([r["dir_init"] for r in ri]),
                      "latest": _frac([r["dir_latest"] for r in rows])},
        "coverage": {"initial": _frac([r["in_init"] for r in ri]),
                     "latest": _frac([r["in_latest"] for r in rows])},
        "pinball": {"initial": _m([r["pin_init"] for r in ri]),
                    "latest": _m([r["pin_latest"] for r in rows])},
        "coverage_vb": {"initial": _frac([r["in_vb_init"] for r in ri]),
                        "latest": _frac([r["in_vb_latest"] for r in rows])},
        "pinball_vb": {"initial": _m([r["pin_vb_init"] for r in ri]),
                       "latest": _m([r["pin_vb_latest"] for r in rows])}}