"""M5b engine: replay (no-look-ahead gateway), input forms, covariates, model runner, immutable snapshots.
Market data comes only from the M1 cache (upstox_data.get_range). No order endpoints.

Improvements for accuracy:
- Default to log_price_stitched + time_of_day
- Support for volume / range / combined covariates
- Multi-covariate support in ModelRunner
"""
import hashlib
import json
import os
import stat
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import upstox_data as ud

IST = ud.IST
RUNS = ud.DATA / "runs"
BARS_PER_DAY = 375
QLEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MEDIAN_IDX = 4
STUDIO = Path(os.getenv("TIMESFM_STUDIO_DIR", r"D:\timesfm3\timesfm3-studio"))
MODEL_DIR = Path(os.getenv("TIMESFM_MODEL_DIR", str(STUDIO / "models" / "timesfm-3.0-pytorch")))
INPUT_FORMS = ("raw_close", "log_price_stitched", "log_returns")
COVARIATES = ("none", "time_of_day", "volume", "range", "tod_volume")
DEFAULT_SETTINGS = {
    "context_len": 2048,
    "input_form": "log_price_stitched",   # better than raw_close for financial data
    "covariates": "time_of_day"           # free known-future signal
}
MIN = pd.Timedelta(minutes=1)
BPS = 1e4
SESSION_OPEN_MIN = 9 * 60 + 15      # 09:15 in minutes since midnight
SESSION_SPAN = 374.0                # 15:29 - 09:15 in minutes


# ---------------------------------------------------------------- time / data helpers
def make_ts(d, hhmm):
    h, m = (int(x) for x in hhmm.split(":"))
    return pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=h, minute=m, tz=IST)


def latest_cached_day(sym):
    files = sorted((ud.CACHE / sym).glob("*.parquet"))
    if not files:
        raise SystemExit("No cached days. Run m1_check.py first.")
    return date.fromisoformat(files[-1].stem)


def load_history(sym, day, context_len):
    """Bars for `day` plus enough earlier sessions to fill context_len bars (all from the M1 cache)."""
    days_back = max(10, int(context_len / BARS_PER_DAY * 2) + 8)
    return ud.get_range(sym, day - timedelta(days=days_back), day)


# ---------------------------------------------------------------- replay: the ONLY gateway to context
class Replay:
    def __init__(self, bars):
        self.bars = bars.sort_values("ts_open").reset_index(drop=True)

    def closed_until(self, T):
        """Bars that have CLOSED at or before T. Bar-open stamps: ts_close = ts_open + 1 min."""
        return self.bars[self.bars["ts_close"] <= T]

    def context(self, T, n):
        return self.closed_until(T).tail(n)

    def bar_closing_at(self, T):
        m = self.bars[self.bars["ts_close"] == T]
        return None if m.empty else m.iloc[0]

    def bar_opening_at(self, T):
        m = self.bars[self.bars["ts_open"] == T]
        return None if m.empty else m.iloc[0]


# ---------------------------------------------------------------- settings, input forms, covariates
def check_settings(s):
    if s.get("input_form") not in INPUT_FORMS:
        raise ValueError(f"input_form must be one of {INPUT_FORMS}, got {s.get('input_form')!r}")
    if s.get("covariates") not in COVARIATES:
        raise ValueError(f"covariates must be one of {COVARIATES}, got {s.get('covariates')!r}")


def _session_returns(ctx_df):
    """Within-session log returns. The overnight gap is excluded: each session's first bar
    contributes only its own open->close move. Uses only bars that are in ctx_df."""
    c = ctx_df["close"].to_numpy(dtype=np.float64)
    o = ctx_df["open"].to_numpy(dtype=np.float64)
    if not (np.all(c > 0) and np.all(o > 0)):
        raise ValueError("non-positive price in context")
    d = ctx_df["ts_open"].dt.date.to_numpy()
    prev = np.empty_like(c)
    prev[0] = o[0]
    prev[1:] = c[:-1]
    newday = np.ones(len(c), dtype=bool)
    newday[1:] = d[1:] != d[:-1]
    base = np.where(newday, o, prev)
    return np.log(c / base)


def _tod_cov(ctx_df, T, horizon):
    """Time-of-day covariate, shape (1, ctx+horizon). Depends on timestamps only (known in advance)."""
    stamps = list(ctx_df["ts_open"]) + [T + i * MIN for i in range(horizon)]
    mins = np.array([t.hour * 60 + t.minute - SESSION_OPEN_MIN for t in stamps], dtype=np.float64)
    return (mins / SESSION_SPAN)[None, :].astype(np.float32)


def _volume_cov(ctx_df, T, horizon):
    """Log-normalized volume. Future volume is unknown → carry forward last value."""
    vol = ctx_df["volume"].to_numpy(dtype=np.float64)
    recent = vol[-60:] if len(vol) >= 60 else vol
    scale = float(np.mean(recent)) + 1e-6
    v = np.log1p(vol / scale).astype(np.float32)
    future = np.full(horizon, v[-1], dtype=np.float32)
    return np.concatenate([v, future])[None, :]


def _range_cov(ctx_df, T, horizon):
    """Normalized high-low range (a simple volatility proxy)."""
    hl = (ctx_df["high"] - ctx_df["low"]).to_numpy(dtype=np.float64)
    close = ctx_df["close"].to_numpy(dtype=np.float64)
    r = hl / (close + 1e-6)
    recent = r[-60:] if len(r) >= 60 else r
    mu = float(np.mean(recent))
    sigma = float(np.std(recent)) + 1e-6
    r = ((r - mu) / sigma).astype(np.float32)
    future = np.full(horizon, r[-1], dtype=np.float32)
    return np.concatenate([r, future])[None, :]


def prepare_input(ctx_df, T, settings, horizon):
    """Returns (x, cov, back). x: 1-D model input. cov: covariate array or None.
    back(q_raw) maps the model's (horizon, 9) output back to PRICE space.

    cov shape is (n_covariates, context_len + horizon) when present.
    """
    check_settings(settings)
    form, cov_mode = settings["input_form"], settings["covariates"]
    close = ctx_df["close"].to_numpy(dtype=np.float64)
    last = float(close[-1])

    if form == "raw_close":
        x, back = close, (lambda q: q)
    else:
        r = _session_returns(ctx_df)
        if form == "log_price_stitched":
            s = np.cumsum(r)
            x = BPS * (s - s[-1])
            back = lambda q: last * np.exp(q / BPS)
        else:  # log_returns
            x = BPS * r
            back = lambda q: last * np.exp(np.cumsum(q / BPS, axis=0))

    # Build covariates
    if cov_mode == "none":
        cov = None
    elif cov_mode == "time_of_day":
        cov = _tod_cov(ctx_df, T, horizon)
    elif cov_mode == "volume":
        cov = _volume_cov(ctx_df, T, horizon)
    elif cov_mode == "range":
        cov = _range_cov(ctx_df, T, horizon)
    elif cov_mode == "tod_volume":
        tod = _tod_cov(ctx_df, T, horizon)
        vol = _volume_cov(ctx_df, T, horizon)
        cov = np.concatenate([tod, vol], axis=0)  # (2, ctx+horizon)
    else:
        raise ValueError(f"Unknown covariates mode: {cov_mode}")

    return x, cov, back


# ---------------------------------------------------------------- model runner
def _to_np(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=np.float64)


class ModelRunner:
    def __init__(self, path=MODEL_DIR, device="cuda", per_core_batch_size=16):
        if str(STUDIO) not in sys.path:
            sys.path.insert(0, str(STUDIO))
        try:
            from timesfm3 import TimesFM3Forecaster
        except ImportError as e:
            raise SystemExit(f"Cannot import timesfm3 ({e}). Tell me where the timesfm3 package lives.")
        self.info = {"class": "TimesFM3Forecaster", "path": str(path), "device": device,
                     "per_core_batch_size": per_core_batch_size, "precision": "fp32"}
        self.f = TimesFM3Forecaster.from_pretrained(str(path), device=device,
                                                    per_core_batch_size=per_core_batch_size)
        self.output_info = None

    def forecast(self, x, horizon, cov=None):
        """x: 1-D float array.
        cov: None or array of shape (n_covariates, len(x)+horizon).
        Returns quantiles (horizon, 9).
        """
        arr = np.asarray(x, dtype=np.float32)
        kw = {}
        if cov is not None:
            cov = np.asarray(cov, dtype=np.float32)
            expected_len = len(arr) + horizon
            if cov.ndim == 1:
                cov = cov[None, :]
            if cov.shape[1] != expected_len:
                raise ValueError(f"covariate length {cov.shape[1]}, expected {expected_len}")
            # TimesFM3 expects a list of covariate arrays (one per series)
            kw["past_future_covariates"] = [cov]

        outs = list(self.f.predict_batch([arr], horizon=horizon, return_quantiles=True, **kw))
        if len(outs) != 1:
            raise RuntimeError(f"expected 1 ForecastOutput, got {len(outs)}")
        out = outs[0]
        q = getattr(out, "quantiles", None)
        if q is None and isinstance(out, dict):
            q = out.get("quantiles")
        if self.output_info is None:
            self.output_info = {"type": type(out).__name__,
                                "attrs": [a for a in dir(out) if not a.startswith("_")][:30]}
        if q is None:
            raise RuntimeError(f"ForecastOutput has no 'quantiles'. {self.output_info}")
        q = _to_np(q)
        if q.ndim == 3 and q.shape[0] == 1:
            q = q[0]
        if q.shape != (horizon, 9):
            raise RuntimeError(f"quantiles shape {q.shape}, expected ({horizon}, 9). {self.output_info}")
        if not np.all(np.isfinite(q)):
            raise RuntimeError("non-finite quantiles")
        return q


def forecast_at(runner, rp, T, settings, horizon):
    """The single production path: context via the gateway, input form, model, back to price.
    Returns (q in PRICE space (horizon, 9), ctx_df, ms).
    """
    ctx = rp.context(T, settings["context_len"])
    if ctx.empty:
        raise RuntimeError(f"no closed bars at or before {T}")
    x, cov, back = prepare_input(ctx, T, settings, horizon)
    t = time.perf_counter()
    q_raw = runner.forecast(x, horizon, cov)
    ms = (time.perf_counter() - t) * 1000
    q = back(q_raw)
    if not np.all(np.isfinite(q)):
        raise RuntimeError("non-finite quantiles after conversion to price")
    return q, ctx, ms


# ---------------------------------------------------------------- immutable snapshots
def _canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def _sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _write_once(path, obj):
    with open(path, "x", encoding="utf-8") as fh:       # 'x' = fail if the file exists
        json.dump(obj, fh, indent=1)
    os.chmod(path, stat.S_IREAD)                         # read-only attribute


def write_snapshot(run_dir, snap):
    snap["sha256"] = _sha(_canon({k: v for k, v in snap.items() if k != "sha256"}))
    path = Path(run_dir) / f"snap_{snap['seq']:04d}.json"
    _write_once(path, snap)
    return path


def verify_run(run_dir):
    problems, inits = [], 0
    files = sorted(Path(run_dir).glob("snap_*.json"))
    if not files:
        return ["no snapshots found"]
    for i, f in enumerate(files):
        s = json.loads(f.read_text(encoding="utf-8"))
        h = s.pop("sha256", None)
        if h != _sha(_canon(s)):
            problems.append(f"{f.name}: hash mismatch (file was modified)")
        if s["seq"] != i:
            problems.append(f"{f.name}: seq {s['seq']} != {i}")
        inits += s["kind"] == "initial"
        if i == 0 and s["kind"] != "initial":
            problems.append("snap_0000 is not the initial forecast")
    if inits != 1:
        problems.append(f"expected exactly 1 initial snapshot, found {inits}")
    return problems


# ---------------------------------------------------------------- simulation (command line)
def run_simulation(runner, sym, day, start_hhmm, horizon=10, n_steps=10, settings=None, log=print):
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    check_settings(settings)
    T0 = make_ts(day, start_hhmm)
    end_ts = make_ts(day, "15:30")
    if not (make_ts(day, "09:15") <= T0 < end_ts):
        raise SystemExit("start time must be 09:15..15:29")
    bars = load_history(sym, day, settings["context_len"])
    if bars[bars["ts_open"].dt.date == day].empty:
        raise SystemExit(f"No cached bars for {day} (today is never cached; pick a past trading day).")
    rp = Replay(bars)

    run_dir = RUNS / f"{sym}_{day}_{start_hhmm.replace(':', '')}_h{horizon}_c{settings['context_len']}_{datetime.now():%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_once(run_dir / "run.json", {
        "symbol": sym, "date": str(day), "start": start_hhmm, "horizon": horizon,
        "n_steps_requested": n_steps, "settings": settings, "model": runner.info,
        "bar_convention": "ts_open = bar open; context = bars with ts_close <= origin",
        "created": datetime.now(IST).isoformat()})

    snaps = []

    def make(seq, kind, T):
        q, ctx, ms = forecast_at(runner, rp, T, settings, horizon)
        snap = {
            "seq": seq, "kind": kind, "origin_ts": T.isoformat(),
            "last_context_bar_open": ctx["ts_open"].iloc[-1].isoformat(),
            "n_context_bars": int(len(ctx)),
            "context_sha256": hashlib.sha256(ctx["close"].to_numpy(np.float64).tobytes()).hexdigest(),
            "horizon": horizon,
            "target_open_ts": [(T + i * MIN).isoformat() for i in range(horizon)],
            "quantile_levels": QLEVELS, "quantiles": q.tolist(), "median": q[:, MEDIAN_IDX].tolist(),
            "quantile_crossing": bool(np.any(np.diff(q, axis=1) < 0)),
            "band_approx": settings["input_form"] == "log_returns",
            "settings": dict(settings), "model": runner.info,
            "infer_ms": round(ms, 1), "created_utc": datetime.utcnow().isoformat() + "Z"}
        write_snapshot(run_dir, snap)
        snaps.append(snap)
        log(f"  snap {seq:3d} {kind:7s} origin {T:%H:%M} ctx_bars={len(ctx)} "
            f"{ms:6.0f} ms  q50[0]={q[0, MEDIAN_IDX]:.2f}")

    make(0, "initial", T0)
    for k in range(1, n_steps + 1):
        Tk = T0 + k * MIN                      # the bar that opened at Tk-1 has just closed
        if Tk >= end_ts:
            log("  session end reached")
            break
        if rp.bar_closing_at(Tk) is None:
            log(f"  no bar closing at {Tk:%H:%M} (data gap); stopping")
            break
        make(k, "update", Tk)
    return run_dir, snaps, rp, T0