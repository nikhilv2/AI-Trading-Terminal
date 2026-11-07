"""M5b server: FastAPI backend serving one HTML page. Local cache + local model only; no order endpoints."""
import hashlib
import threading
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import engine as eg
import metrics as mt
import upstox_data as ud

HERE = Path(__file__).resolve().parent
LAG = 0      # 'latest' forecast for target j = the one made at origin T0+j-LAG (0 = last one before the bar is revealed)
LOCK = threading.Lock()
S = {"runner": None, "sess": None}


@asynccontextmanager
async def lifespan(app):
    S["runner"] = eg.ModelRunner()      # model loaded once at server start
    yield


app = FastAPI(lifespan=lifespan)


class LoadReq(BaseModel):
    symbol: str = "NIFTY50"
    date: str
    time: str
    horizon: int = 10
    ctx: int = 2048
    steps: int = 10
    input_form: str = "raw_close"
    covariates: str = "none"


class ExtReq(BaseModel):
    steps: int


def fmt(t):
    return t.strftime("%Y-%m-%d %H:%M:%S")      # naive IST wall-clock string for Plotly


def candles(df):
    d = {"x": [fmt(t) for t in df["ts_open"]]}
    for c in ("open", "high", "low", "close"):
        d[c] = df[c].round(2).tolist()
    return d


def need():
    if S["sess"] is None:
        raise HTTPException(400, "Press Load first")
    return S["sess"]


def make_snap(s, seq, kind, T):
    H = s["horizon"]
    try:
        q, ctx, ms = eg.forecast_at(S["runner"], s["rp"], T, s["settings"], H)
    except Exception as e:
        raise HTTPException(500, f"forecast failed: {e}")
    snap = {
        "seq": seq, "kind": kind, "origin_ts": T.isoformat(),
        "last_context_bar_open": ctx["ts_open"].iloc[-1].isoformat(),
        "n_context_bars": int(len(ctx)),
        "context_sha256": hashlib.sha256(ctx["close"].to_numpy(np.float64).tobytes()).hexdigest(),
        "horizon": H, "target_open_ts": [(T + i * eg.MIN).isoformat() for i in range(H)],
        "quantile_levels": eg.QLEVELS, "quantiles": q.tolist(), "median": q[:, eg.MEDIAN_IDX].tolist(),
        "quantile_crossing": bool(np.any(np.diff(q, axis=1) < 0)),
        "band_approx": s["settings"]["input_form"] == "log_returns",
        "settings": dict(s["settings"]), "model": S["runner"].info,
        "infer_ms": round(ms, 1), "created_utc": datetime.utcnow().isoformat() + "Z"}
    eg.write_snapshot(s["run_dir"], snap)
    return {"seq": seq, "kind": kind, "origin": fmt(T),
            "targets": [fmt(T + i * eg.MIN) for i in range(H)],
            "p10": q[:, 0].round(2).tolist(), "median": q[:, eg.MEDIAN_IDX].round(2).tolist(),
            "p90": q[:, 8].round(2).tolist(), "q": q.round(3).tolist(), "infer_ms": snap["infer_ms"]}


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index.html").read_text(encoding="utf-8")


@app.get("/api/meta")
def meta():
    sym = next(iter(ud.SYMBOLS))
    try:
        latest = str(eg.latest_cached_day(sym))
    except SystemExit:
        latest = ""
    return {"symbols": list(ud.SYMBOLS), "latest": latest}


@app.post("/api/load")
def load(r: LoadReq):
    with LOCK:
        try:
            if r.symbol not in ud.SYMBOLS:
                raise ValueError("unknown symbol")
            if not (1 <= r.horizon <= 60) or r.ctx not in (256, 512, 1024, 2048):
                raise ValueError("horizon must be 1..60, context one of 256/512/1024/2048")
            if not (1 <= r.steps <= 60):
                raise ValueError("run length must be 1..60")
            settings = {**eg.DEFAULT_SETTINGS, "context_len": r.ctx,
                        "input_form": r.input_form, "covariates": r.covariates}
            eg.check_settings(settings)
            day = date.fromisoformat(r.date)
            T0 = eg.make_ts(day, r.time)
            if not (eg.make_ts(day, "09:15") <= T0 < eg.make_ts(day, "15:30")):
                raise ValueError("time must be 09:15..15:29")
            bars = eg.load_history(r.symbol, day, r.ctx)
            if bars[bars["ts_open"].dt.date == day].empty:
                raise ValueError(f"no cached bars for {day} (weekend, holiday, or today)")
        except (Exception, SystemExit) as e:
            raise HTTPException(400, str(e))
        rp = eg.Replay(bars)
        today = rp.closed_until(T0)
        today = today[today["ts_open"].dt.date == day]
        dates = sorted({t.date() for t in bars["ts_open"] if t.date() < day})[-3:]
        prev = bars[bars["ts_open"].dt.date.isin(dates)]
        S["sess"] = {"rp": rp, "T0": T0, "sym": r.symbol, "day": day, "start": r.time,
                     "horizon": r.horizon, "settings": settings,
                     "steps": r.steps, "k": 0, "run_dir": None, "done": False, "done_why": None}
        return {"prev": candles(prev), "today": candles(today), "T0": fmt(T0), "horizon": r.horizon,
                "settings": settings}


@app.post("/api/start")
def start():
    with LOCK:
        s = need()
        if s["run_dir"] is not None:
            raise HTTPException(400, "Already started; press Reset first")
        rd = eg.RUNS / (f"{s['sym']}_{s['day']}_{s['start'].replace(':', '')}_h{s['horizon']}"
                        f"_c{s['settings']['context_len']}_{datetime.now():%H%M%S}")
        rd.mkdir(parents=True, exist_ok=False)
        eg._write_once(rd / "run.json", {
            "symbol": s["sym"], "date": str(s["day"]), "start": s["start"], "horizon": s["horizon"],
            "settings": s["settings"], "model": S["runner"].info, "source": "gui",
            "bar_convention": "ts_open = bar open; context = bars with ts_close <= origin",
            "created": datetime.now(eg.IST).isoformat()})
        s["run_dir"] = rd
        return {"snap": make_snap(s, 0, "initial", s["T0"])}


@app.post("/api/extend")
def extend(r: ExtReq):
    """Change the run length (10/30/60 in the GUI). Can be raised after the run finished."""
    with LOCK:
        s = need()
        if not (1 <= r.steps <= 60):
            raise HTTPException(400, "run length must be 1..60")
        if s["run_dir"] is not None and r.steps < s["k"]:
            raise HTTPException(400, f"already revealed {s['k']} bars; cannot go below that")
        s["steps"] = r.steps
        if s["run_dir"] is not None:
            if s["k"] >= s["steps"]:
                s["done"], s["done_why"] = True, "steps"
            elif s["done_why"] == "steps":
                s["done"], s["done_why"] = False, None
        return {"steps": s["steps"], "done": s["done"]}


def why_text(s):
    return f"reached {s['steps']} bars" if s["done_why"] == "steps" else s["done_why"]


@app.post("/api/step")
def step():
    with LOCK:
        s = need()
        if s["run_dir"] is None:
            raise HTTPException(400, "Press Start first")
        if s["done"]:
            return {"done": True, "reason": why_text(s)}
        k = s["k"] + 1
        Tk = s["T0"] + k * eg.MIN
        bar = s["rp"].bar_closing_at(Tk)
        why = None
        if k > s["steps"]:
            why = "steps"
        elif Tk >= eg.make_ts(s["day"], "15:30"):
            why = "session end"
        elif bar is None:
            why = "data gap"
        if why:
            s["done"], s["done_why"] = True, why
            return {"done": True, "reason": why_text(s)}
        s["k"] = k
        snap = make_snap(s, k, "update", Tk)      # context = bars closed at or before Tk
        b = {"x": fmt(bar["ts_open"])}
        for c in ("open", "high", "low", "close"):
            b[c] = round(float(bar[c]), 2)
        finished = None
        if k >= s["steps"]:
            s["done"], s["done_why"] = True, "steps"
            finished = why_text(s)
        return {"done": False, "bar": b, "snap": snap, "k": k, "finished": finished}


@app.get("/api/compare")
def compare_view():
    with LOCK:
        s = need()
        if s["run_dir"] is None:
            raise HTTPException(400, "Press Start first")
        snaps = mt.load_snaps(s["run_dir"])          # evaluated from the immutable files on disk
        if len(snaps) < 2:
            raise HTTPException(400, "Reveal at least one bar first (Step or Play)")
        try:
            out = mt.compare(snaps, s["rp"], lag=LAG)
        except Exception as e:
            raise HTTPException(500, f"compare failed: {e}")
        out["settings"] = dict(s["settings"])
        out["band_approx"] = s["settings"]["input_form"] == "log_returns"
        return out


@app.post("/api/reset")
def reset():
    with LOCK:
        s = need()
        s.update(k=0, run_dir=None, done=False, done_why=None)
        return {"ok": True}