"""
NIFTY / SENSEX / BANKNIFTY — Live Options & Technical Dashboard (Dhan API v2)
Run:  streamlit run app.py
Data source: DhanHQ v2 (Option Chain, Intraday Charts, LTP). Demo mode works without API.
Analysis tool only — not investment advice.
"""
import os
import math
import time
import datetime as dt

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------ config
BASE = "https://api.dhan.co/v2"
SCRIP_MASTER = "https://images.dhan.co/api-data/api-scrip-master.csv"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
RISK_FREE = 0.065
VIX_ID = 21
# Lot sizes change with NSE/BSE circulars — app tries to read them from Dhan scrip master; these are only fallbacks.
UNDERLYINGS = {
    "NIFTY": {"id": 13, "seg": "IDX_I", "fno": "NSE_FNO", "step": 50, "lot": 75},
    "SENSEX": {"id": 51, "seg": "IDX_I", "fno": "BSE_FNO", "step": 100, "lot": 20},
    "BANKNIFTY": {"id": 25, "seg": "IDX_I", "fno": "NSE_FNO", "step": 100, "lot": 35},
}

st.set_page_config(page_title="Index Options Command Center", page_icon="📈", layout="wide")

st.markdown(
    """
<style>
.stApp {background: radial-gradient(circle at 10% 0%, #1b1440 0%, #0b0f1e 45%, #070a14 100%); color:#e8eaf6;}
section[data-testid="stSidebar"] {background: rgba(20,18,45,0.92); border-right:1px solid rgba(140,120,255,0.18);}
.card {background: rgba(255,255,255,0.045); border:1px solid rgba(160,140,255,0.20); border-radius:16px;
       padding:14px 16px; backdrop-filter: blur(8px); transition: transform .15s ease, border-color .15s ease; height:100%;}
.card:hover {transform: translateY(-2px); border-color: rgba(120,200,255,0.55);}
.card .lbl {font-size:0.74rem; letter-spacing:.06em; text-transform:uppercase; color:#a7a9d6;}
.card .val {font-size:1.45rem; font-weight:700; margin-top:2px;}
.card .sub {font-size:0.78rem; color:#9aa0c3; margin-top:2px;}
.g {color:#34e5a0;} .r {color:#ff6b8a;} .b {color:#6cc6ff;} .y {color:#ffd166;} .p {color:#c49bff;}
.bias {border-radius:18px; padding:18px 20px; font-size:1.05rem;
       background: linear-gradient(135deg, rgba(108,198,255,0.16), rgba(196,155,255,0.16)); border:1px solid rgba(160,140,255,0.35);}
h1, h2, h3 {color:#f1f0ff;}
@media (max-width: 640px) {
  .card {padding:10px 12px; border-radius:12px; margin-bottom:8px;}
  .card .val {font-size:1.15rem;}
  .card .lbl {font-size:0.68rem;}
  .bias {font-size:0.95rem; padding:14px;}
  h1 {font-size:1.5rem !important;}
  .block-container {padding-left:0.8rem; padding-right:0.8rem; padding-top:2.5rem;}
}
</style>
""",
    unsafe_allow_html=True,
)


def card(label, value, sub="", color="b"):
    return f"<div class='card'><div class='lbl'>{label}</div><div class='val {color}'>{value}</div><div class='sub'>{sub}</div></div>"


def fmt(x, d=2):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{x:,.{d}f}"
    except (TypeError, ValueError):
        return str(x)


def now_ist():
    return dt.datetime.now(IST)


# ------------------------------------------------------------------ Black-Scholes (fallback greeks)
_erf = np.vectorize(math.erf)


def ncdf(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def npdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs_all(S, K, T, sigma, r=RISK_FREE, call=True):
    K = np.asarray(K, dtype=float)
    sigma = np.clip(np.asarray(sigma, dtype=float), 1e-4, 5)
    T = max(T, 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    gamma = npdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * npdf(d1) * math.sqrt(T) / 100
    if call:
        price = S * ncdf(d1) - K * math.exp(-r * T) * ncdf(d2)
        delta = ncdf(d1)
        theta = (-S * npdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * ncdf(d2)) / 365
    else:
        price = K * math.exp(-r * T) * ncdf(-d2) - S * ncdf(-d1)
        delta = ncdf(d1) - 1
        theta = (-S * npdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * ncdf(-d2)) / 365
    return price, delta, gamma, theta, vega


def years_to_expiry(expiry):
    exp = dt.datetime.strptime(expiry, "%Y-%m-%d").replace(hour=15, minute=30, tzinfo=IST)
    return max((exp - now_ist()).total_seconds() / (365 * 86400), 1 / (365 * 24 * 4))


# ------------------------------------------------------------------ Dhan API layer
@st.cache_resource
def _throttle_state():
    return {"oc": 0.0, "ltp": 0.0, "chart": 0.0}


def _wait(kind, gap):
    s = _throttle_state()
    delta = time.time() - s[kind]
    if delta < gap:
        time.sleep(gap - delta)
    s[kind] = time.time()


def dhan_post(path, body, cid, tok, timeout=15):
    r = requests.post(
        BASE + path,
        json=body,
        headers={"access-token": tok, "client-id": cid, "Content-Type": "application/json", "Accept": "application/json"},
        timeout=timeout,
    )
    try:
        js = r.json()
    except ValueError:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    if r.status_code != 200 or (isinstance(js, dict) and js.get("status") == "failure"):
        raise RuntimeError(f"HTTP {r.status_code}: {str(js)[:300]}")
    return js


@st.cache_data(ttl=600, show_spinner=False)
def api_expiries(cid, tok, uid, seg):
    _wait("oc", 3.1)
    js = dhan_post("/optionchain/expirylist", {"UnderlyingScrip": uid, "UnderlyingSeg": seg}, cid, tok)
    return list(js.get("data", []))


@st.cache_data(ttl=3, show_spinner=False)
def api_chain(cid, tok, uid, seg, expiry):
    _wait("oc", 3.1)
    return dhan_post("/optionchain", {"UnderlyingScrip": uid, "UnderlyingSeg": seg, "Expiry": expiry}, cid, tok)


@st.cache_data(ttl=2, show_spinner=False)
def api_ltp(cid, tok, payload_items):
    _wait("ltp", 1.1)
    payload = {seg: list(ids) for seg, ids in payload_items}
    js = dhan_post("/marketfeed/ltp", payload, cid, tok)
    out = {}
    for seg, m in (js.get("data") or {}).items():
        for sid, v in (m or {}).items():
            out[(seg, int(sid))] = float(v.get("last_price", np.nan))
    return out


@st.cache_data(ttl=30, show_spinner=False)
def api_intraday(cid, tok, sec_id, seg, instrument, interval, days, with_oi=False):
    _wait("chart", 0.6)
    to = now_ist()
    frm = to - dt.timedelta(days=days)
    body = {
        "securityId": str(sec_id), "exchangeSegment": seg, "instrument": instrument, "interval": str(interval),
        "oi": bool(with_oi), "fromDate": frm.strftime("%Y-%m-%d %H:%M:%S"), "toDate": to.strftime("%Y-%m-%d %H:%M:%S"),
    }
    js = dhan_post("/charts/intraday", body, cid, tok)
    if "open" not in js and isinstance(js.get("data"), dict):
        js = js["data"]
    return candles_from_arrays(js)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def scrip_master_futures():
    cols = ["SEM_EXM_EXCH_ID", "SEM_SMST_SECURITY_ID", "SEM_INSTRUMENT_NAME", "SEM_TRADING_SYMBOL",
            "SEM_LOT_UNITS", "SEM_EXPIRY_DATE"]
    df = pd.read_csv(SCRIP_MASTER, usecols=lambda c: c in cols, low_memory=False)
    df = df[df["SEM_INSTRUMENT_NAME"] == "FUTIDX"].copy()
    df["exp"] = pd.to_datetime(df["SEM_EXPIRY_DATE"], errors="coerce")
    return df


def nearest_future(symbol):
    """Returns (security_id, exchange_segment, lot) for the current-month index future, or None."""
    try:
        df = scrip_master_futures()
        today = pd.Timestamp(now_ist().date())
        f = df[df["SEM_TRADING_SYMBOL"].astype(str).str.upper().str.startswith(symbol + "-") & (df["exp"] >= today)]
        if f.empty:
            return None
        row = f.sort_values("exp").iloc[0]
        seg = "NSE_FNO" if str(row["SEM_EXM_EXCH_ID"]).upper() == "NSE" else "BSE_FNO"
        return int(row["SEM_SMST_SECURITY_ID"]), seg, int(float(row["SEM_LOT_UNITS"]))
    except Exception:
        return None


# ------------------------------------------------------------------ parsing
def candles_from_arrays(js):
    keys = ["open", "high", "low", "close", "volume", "timestamp"]
    n = len(js.get("timestamp", []))
    df = pd.DataFrame({k: js.get(k, [0] * n) for k in keys})
    oi = js.get("open_interest") or js.get("oi")
    if oi is not None and len(oi) == n:
        df["oi"] = oi
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    return df.drop(columns="timestamp").sort_values("time").reset_index(drop=True)


def parse_chain(js):
    d = js.get("data", js)
    spot = float(d.get("last_price") or 0)
    rows = []
    for k, v in (d.get("oc") or {}).items():
        row = {"strike": float(k)}
        for side in ("ce", "pe"):
            o = (v or {}).get(side) or {}
            g = o.get("greeks") or {}
            row[f"{side}_ltp"] = o.get("last_price")
            row[f"{side}_oi"] = o.get("oi")
            row[f"{side}_prev_oi"] = o.get("previous_oi")
            row[f"{side}_vol"] = o.get("volume")
            row[f"{side}_iv"] = o.get("implied_volatility")
            row[f"{side}_bid"] = o.get("top_bid_price")
            row[f"{side}_ask"] = o.get("top_ask_price")
            row[f"{side}_bid_qty"] = o.get("top_bid_quantity")
            row[f"{side}_ask_qty"] = o.get("top_ask_quantity")
            row[f"{side}_delta"] = g.get("delta")
            row[f"{side}_gamma"] = g.get("gamma")
            row[f"{side}_theta"] = g.get("theta")
            row[f"{side}_vega"] = g.get("vega")
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return spot, df
    df = df.apply(pd.to_numeric, errors="coerce").sort_values("strike").reset_index(drop=True)
    for c in [c for c in df.columns if c.endswith(("_oi", "_vol"))]:
        df[c] = df[c].fillna(0)
    df["ce_chg_oi"] = df["ce_oi"] - df["ce_prev_oi"]
    df["pe_chg_oi"] = df["pe_oi"] - df["pe_prev_oi"]
    return spot, df


def fill_missing_greeks(df, spot, T):
    for side, call in (("ce", True), ("pe", False)):
        iv = df[f"{side}_iv"].fillna(0) / 100
        miss = (df[f"{side}_gamma"].fillna(0) == 0) & (iv > 0)
        if miss.any():
            _, de, ga, th, ve = bs_all(spot, df.loc[miss, "strike"], T, iv[miss], call=call)
            df.loc[miss, f"{side}_delta"] = de
            df.loc[miss, f"{side}_gamma"] = ga
            df.loc[miss, f"{side}_theta"] = th
            df.loc[miss, f"{side}_vega"] = ve
    return df


# ------------------------------------------------------------------ demo data (no API needed)
def demo_chain(spot=25000.0, step=50, T=3 / 365):
    rng = np.random.default_rng(int(time.time() // 5))
    spot = spot + rng.normal(0, 25)
    strikes = np.arange(round(spot / step) * step - 30 * step, round(spot / step) * step + 31 * step, step)
    oc = {}
    for K in strikes:
        m = (K - spot) / spot
        iv = 0.12 + 1.8 * m * m + (0.02 if m < 0 else 0)
        node = {}
        for side, call in (("ce", True), ("pe", False)):
            p, de, ga, th, ve = (float(np.squeeze(x)) for x in bs_all(spot, K, T, iv, call=call))
            otm = (K >= spot) if call else (K <= spot)
            base = 9e6 if otm else 2.5e6
            bump = 2.2 if K % (step * 10) == 0 else (1.4 if K % (step * 2) == 0 else 1.0)
            oi = base * bump * math.exp(-abs(m) * 35) * rng.uniform(0.7, 1.3)
            spr = max(0.05, p * rng.uniform(0.002, 0.02))
            node[side] = {
                "last_price": round(max(p, 0.05), 2), "oi": int(oi), "previous_oi": int(oi * rng.uniform(0.75, 1.15)),
                "volume": int(oi * rng.uniform(2, 8)), "implied_volatility": iv * 100,
                "top_bid_price": round(max(p - spr / 2, 0.05), 2), "top_ask_price": round(p + spr / 2, 2),
                "top_bid_quantity": int(rng.integers(75, 6000)), "top_ask_quantity": int(rng.integers(75, 6000)),
                "greeks": {"delta": de, "gamma": ga, "theta": th, "vega": ve},
            }
        oc[f"{K:.6f}"] = node
    return {"data": {"last_price": spot, "oc": oc}, "status": "success"}


def demo_candles(spot=25000.0, days=5, interval=5, with_volume=True):
    rng = np.random.default_rng(7)
    rows, price, oi = [], spot * 0.985, 1.2e7
    today = now_ist().date()
    d, made = today - dt.timedelta(days=days + 3), 0
    while d <= today:
        if d.weekday() < 5:
            t = dt.datetime.combine(d, dt.time(9, 15), tzinfo=IST)
            end = dt.datetime.combine(d, dt.time(15, 30), tzinfo=IST)
            if d == today:
                end = min(end, max(now_ist(), t + dt.timedelta(hours=3)))
            while t < end:
                o = price
                c = o + rng.normal(0.6, spot * 0.0009)
                h, l = max(o, c) + abs(rng.normal(0, 8)), min(o, c) - abs(rng.normal(0, 8))
                oi = max(5e6, oi + rng.normal(0, 4e4) + (c - o) * rng.choice([-800, 800]))
                rows.append({"open": o, "high": h, "low": l, "close": c, "oi": oi if with_volume else 0.0,
                             "volume": float(rng.integers(5e4, 3e5)) if with_volume else 0.0, "time": t})
                price, t = c, t + dt.timedelta(minutes=interval)
            made += 1
        d += dt.timedelta(days=1)
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_convert("Asia/Kolkata")
    shift = spot - df["close"].iloc[-1]
    for c in ("open", "high", "low", "close"):
        df[c] += shift
    return df


# ------------------------------------------------------------------ analytics
def max_pain(df):
    K, ce, pe = df["strike"].values, df["ce_oi"].values, df["pe_oi"].values
    pains = np.array([(np.maximum(s - K, 0) * ce).sum() + (np.maximum(K - s, 0) * pe).sum() for s in K])
    return float(K[int(np.argmin(pains))]), pains


def gex_profile(df, spot, oi_mult, T):
    g = (df["ce_gamma"].fillna(0) * df["ce_oi"] - df["pe_gamma"].fillna(0) * df["pe_oi"]) * oi_mult * spot * spot * 0.01
    df["gex_cr"] = g / 1e7
    total = float(df["gex_cr"].sum())
    # Gamma flip: re-price gamma at hypothetical spot levels and find where net GEX changes sign (nearest to spot)
    K = df["strike"].values
    ce_iv = (df["ce_iv"].fillna(0) / 100).replace(0, np.nan).fillna(df["ce_iv"].median() / 100 or 0.15).values
    pe_iv = (df["pe_iv"].fillna(0) / 100).replace(0, np.nan).fillna(df["pe_iv"].median() / 100 or 0.15).values
    grid = np.linspace(spot * 0.94, spot * 1.06, 121)
    net = []
    for S in grid:
        gc = bs_all(S, K, T, ce_iv, call=True)[2]
        gp = bs_all(S, K, T, pe_iv, call=False)[2]
        net.append(float(np.sum(gc * df["ce_oi"].values - gp * df["pe_oi"].values) * oi_mult * S * S * 0.01))
    net = np.array(net)
    flips = [grid[i - 1] + (grid[i] - grid[i - 1]) * abs(net[i - 1]) / (abs(net[i - 1]) + abs(net[i]))
             for i in range(1, len(net)) if np.sign(net[i]) != np.sign(net[i - 1])]
    flip = float(min(flips, key=lambda x: abs(x - spot))) if flips else None
    return total, flip


def liquidity_table(df, atm, step, n):
    w = df[(df["strike"] >= atm - n * step) & (df["strike"] <= atm + n * step)].copy()
    for s in ("ce", "pe"):
        mid = (w[f"{s}_bid"] + w[f"{s}_ask"]) / 2
        w[f"{s}_spread%"] = np.where(mid > 0, (w[f"{s}_ask"] - w[f"{s}_bid"]) / mid * 100, np.nan)
    vol_floor = max(1.0, float(pd.concat([w["ce_vol"], w["pe_vol"]]).quantile(0.25)))

    def tag(sp, vol):
        if pd.isna(sp) or vol <= 0:
            return "⛔ No market"
        if sp <= 1 and vol >= vol_floor:
            return "🟢 Liquid"
        if sp <= 3:
            return "🟡 OK"
        return "🔴 Wide spread"

    w["CE liquidity"] = [tag(a, b) for a, b in zip(w["ce_spread%"], w["ce_vol"])]
    w["PE liquidity"] = [tag(a, b) for a, b in zip(w["pe_spread%"], w["pe_vol"])]
    return w


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def supertrend(df, period=10, mult=3.0):
    a = atr(df, period).values
    hl2 = ((df["high"] + df["low"]) / 2).values
    c = df["close"].values
    ub, lb = hl2 + mult * a, hl2 - mult * a
    fub, flb = ub.copy(), lb.copy()
    direction = np.ones(len(df), dtype=int)
    for i in range(1, len(df)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if direction[i - 1] == -1 and c[i] > fub[i - 1]:
            direction[i] = 1
        elif direction[i - 1] == 1 and c[i] < flb[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    line = np.where(direction == 1, flb, fub)
    return pd.Series(line, index=df.index), pd.Series(direction, index=df.index)


def macd_hist(s):
    m = ema(s, 12) - ema(s, 26)
    return m - ema(m, 9)


def adx(df, n=14):
    up, dn = df["high"].diff(), -df["low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n).replace(0, np.nan)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0), pdi.fillna(0), mdi.fillna(0)


BUILDUP_SCORE = {"Long Buildup": 2, "Short Covering": 1, "Neutral": 0, "Long Unwinding": -1, "Short Buildup": -2, "NA": 0}


def add_indicators(df):
    df = df.copy()
    df["date"] = df["time"].dt.date
    df["ema9"], df["ema21"] = ema(df["close"], 9), ema(df["close"], 21)
    df["rsi"] = rsi(df["close"])
    df["atr"] = atr(df, 14)
    df["st"], df["st_dir"] = supertrend(df)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].fillna(0)
    if vol.sum() > 0:
        df["vwap"] = (tp * vol).groupby(df["date"]).cumsum() / vol.groupby(df["date"]).cumsum().replace(0, np.nan)
        df.attrs["vwap_kind"] = "VWAP"
    else:
        df["vwap"] = tp.groupby(df["date"]).cumsum() / tp.groupby(df["date"]).cumcount().add(1)
        df.attrs["vwap_kind"] = "TWAP (no volume)"
    w = vol if vol.sum() > 0 else pd.Series(1.0, index=df.index)
    var = (w * (tp - df["vwap"]) ** 2).groupby(df["date"]).cumsum() / w.groupby(df["date"]).cumsum().replace(0, np.nan)
    sd = np.sqrt(var).fillna(0)
    df["vwap_u1"], df["vwap_l1"] = df["vwap"] + sd, df["vwap"] - sd
    df["vwap_u2"], df["vwap_l2"] = df["vwap"] + 2 * sd, df["vwap"] - 2 * sd
    df["macd_h"] = macd_hist(df["close"])
    df["adx"], df["pdi"], df["mdi"] = adx(df)
    df["rvol"] = (vol / vol.rolling(20, min_periods=5).mean()).fillna(1.0) if vol.sum() > 0 else 1.0
    if "oi" in df and df["oi"].fillna(0).sum() > 0:
        doi, dpx = df["oi"].diff(), df["close"].diff()
        df["buildup"] = np.select([(dpx > 0) & (doi > 0), (dpx < 0) & (doi > 0), (dpx > 0) & (doi < 0), (dpx < 0) & (doi < 0)],
                                  ["Long Buildup", "Short Buildup", "Short Covering", "Long Unwinding"], "Neutral")
        df["oi_flow"] = df["buildup"].map(BUILDUP_SCORE).rolling(3, min_periods=1).sum()
    else:
        df["buildup"], df["oi_flow"] = "NA", 0.0
    return df


CANDLE_FACTORS = ["VWAP", "EMA 9/21", "Supertrend", "RSI", "MACD", "DI+ vs DI−", "Futures OI buildup", "Volume spike"]


def candle_scores(df):
    s = pd.DataFrame(index=df.index)
    s["VWAP"] = np.sign(df["close"] - df["vwap"])
    s["EMA 9/21"] = np.sign(df["ema9"] - df["ema21"])
    s["Supertrend"] = df["st_dir"]
    s["RSI"] = np.where(df["rsi"] > 55, 1, np.where(df["rsi"] < 45, -1, 0))
    s["MACD"] = np.sign(df["macd_h"])
    s["DI+ vs DI−"] = np.sign(df["pdi"] - df["mdi"])
    s["Futures OI buildup"] = np.clip(np.round(df["oi_flow"] / 2), -2, 2)
    s["Volume spike"] = np.where(df["rvol"] > 1.5, np.sign(df["close"] - df["open"]), 0)
    return s.fillna(0).astype(int)


def option_flow(chain, spot, atm, step, snaps, pcr, chg_pcr):
    """Live option-chain / smart-money layer. Returns rows (Factor, Score, Reading) and total."""
    rows = []

    def add(f, sc, rd):
        rows.append({"Factor": f, "Score": int(sc), "Reading": rd})

    add("PCR (OI)", 1 if pcr > 1.2 else (-1 if pcr < 0.7 else 0), f"{pcr:.2f} (>1.2 put writers strong, <0.7 call writers strong)")
    add("PCR (fresh ΔOI)", 1 if chg_pcr > 1.2 else (-1 if chg_pcr < 0.8 else 0), f"{chg_pcr:.2f} — aaj kaun zyada likh raha hai")
    near = chain[(chain["strike"] >= atm - 3 * step) & (chain["strike"] <= atm + 3 * step)]
    ce_w, pe_w = near["ce_chg_oi"].sum(), near["pe_chg_oi"].sum()
    add("Near-ATM writing (±3 strikes)", 1 if pe_w > 1.2 * max(ce_w, 1) else (-1 if ce_w > 1.2 * max(pe_w, 1) else 0),
        f"PE ΔOI {pe_w:,.0f} vs CE ΔOI {ce_w:,.0f}")
    if {"ce_bid_qty", "ce_ask_qty"}.issubset(chain.columns):
        def imb(sd):
            b, a = near[f"{sd}_bid_qty"].fillna(0).sum(), near[f"{sd}_ask_qty"].fillna(0).sum()
            return (b - a) / max(b + a, 1)
        pr = imb("ce") - imb("pe")
        add("Order-book pressure (bid vs ask qty)", 1 if pr > 0.15 else (-1 if pr < -0.15 else 0),
            f"CE {imb('ce'):+.2f} · PE {imb('pe'):+.2f} (CE buyers vs PE buyers)")
    if len(snaps) >= 6:
        sn = pd.DataFrame(snaps)
        d_pcr = sn["pcr"].iloc[-1] - sn["pcr"].iloc[-6:].mean()
        add("PCR trend (session)", 1 if d_pcr > 0.03 else (-1 if d_pcr < -0.03 else 0), f"{d_pcr:+.3f} vs recent avg")
        if sn["vix"].notna().sum() >= 6:
            d_v = sn["vix"].iloc[-1] - sn["vix"].iloc[-6:].mean()
            add("India VIX trend", -1 if d_v > 0.15 else (1 if d_v < -0.15 else 0), f"{d_v:+.2f} (VIX up = fear)")
    else:
        add("PCR / VIX trend", 0, "Data jama ho raha hai (kuch refresh ke baad dikhega)")
    return rows, sum(r["Score"] for r in rows)


def generate_trades(df, scores, th=6, rr=2.0, sl_min_atr=0.6, sl_max_atr=2.0, adx_min=18):
    """Entry when combined candle score crosses ±th (with ADX trend filter, no chasing beyond VWAP ±2σ)."""
    df = df.reset_index(drop=True)
    tot = scores.sum(axis=1).reset_index(drop=True).values
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    at, ad, rs = df["atr"].values, df["adx"].values, df["rsi"].values
    u2, l2 = df["vwap_u2"].values, df["vwap_l2"].values
    tm, dates = df["time"].dt.time.values, df["date"].values
    trades, active = [], None
    for i in range(30, len(df)):
        if active:
            buy = active["side"] == "BUY"
            hit_sl = l[i] <= active["sl"] if buy else h[i] >= active["sl"]
            hit_tg = h[i] >= active["tg"] if buy else l[i] <= active["tg"]
            day_end = tm[i] >= dt.time(15, 15) or (i + 1 < len(df) and dates[i + 1] != dates[i])
            if hit_sl or hit_tg or day_end:
                out, px = ("SL", active["sl"]) if hit_sl else (("TARGET", active["tg"]) if hit_tg else ("DAY END", c[i]))
                pts = (px - active["entry"]) if buy else (active["entry"] - px)
                active.update(exit_i=i, exit_time=df["time"].iloc[i], exit=px, outcome=out, R=pts / active["risk"])
                trades.append(active)
                active = None
            continue
        if not (dt.time(9, 30) <= tm[i] <= dt.time(14, 45)) or not at[i] > 0 or ad[i] < adx_min:
            continue
        if tot[i] >= th and tot[i - 1] < th and rs[i] < 78 and c[i] < u2[i]:
            risk = float(np.clip(c[i] - l[max(0, i - 3): i + 1].min(), sl_min_atr * at[i], sl_max_atr * at[i]))
            active = dict(side="BUY", i=i, time=df["time"].iloc[i], entry=c[i], sl=c[i] - risk, tg=c[i] + rr * risk, risk=risk, score=int(tot[i]))
        elif tot[i] <= -th and tot[i - 1] > -th and rs[i] > 22 and c[i] > l2[i]:
            risk = float(np.clip(h[max(0, i - 3): i + 1].max() - c[i], sl_min_atr * at[i], sl_max_atr * at[i]))
            active = dict(side="SELL", i=i, time=df["time"].iloc[i], entry=c[i], sl=c[i] + risk, tg=c[i] - rr * risk, risk=risk, score=int(tot[i]))
    if active:
        last = c[-1]
        pts = (last - active["entry"]) if active["side"] == "BUY" else (active["entry"] - last)
        n = now_ist()
        stale = dates[-1] != n.date() or n.time() >= dt.time(15, 15) or n.weekday() >= 5
        active.update(outcome="DAY END" if stale else "OPEN", exit_i=len(df) - 1, exit_time=df["time"].iloc[-1], exit=last, R=pts / active["risk"])
        trades.append(active)
    return trades


def pivots_from(df):
    days = sorted(df["time"].dt.date.unique())
    if len(days) < 2:
        return None
    today = now_ist().date()
    prev_day = days[-2] if days[-1] == today else days[-1]
    p = df[df["time"].dt.date == prev_day]
    H, L, C = p["high"].max(), p["low"].min(), p["close"].iloc[-1]
    P = (H + L + C) / 3
    BC = (H + L) / 2
    TC = 2 * P - BC
    return {"date": prev_day, "PDH": H, "PDL": L, "PDC": C, "P": P, "TC": max(TC, BC), "BC": min(TC, BC),
            "R1": 2 * P - L, "S1": 2 * P - H, "R2": P + (H - L), "S2": P - (H - L),
            "R3": H + 2 * (P - L), "S3": L - 2 * (H - P), "cpr_width%": abs(TC - BC) / P * 100}


def signal_engine(spot, tech, piv, pcr, chg_pcr, mp, total_gex, flip, vix, ce_res, pe_sup):
    rows = []

    def add(name, score, note):
        rows.append({"Factor": name, "Score": score, "Reading": note})

    if tech is not None and len(tech) > 25:
        last = tech.iloc[-1]
        add("Price vs " + tech.attrs.get("vwap_kind", "VWAP"), 1 if last["close"] > last["vwap"] else -1,
            f"{fmt(last['close'])} vs {fmt(last['vwap'])}")
        add("EMA 9/21", 1 if last["ema9"] > last["ema21"] else -1, f"{fmt(last['ema9'])} / {fmt(last['ema21'])}")
        r = last["rsi"]
        add("RSI 14", 1 if r > 55 else (-1 if r < 45 else 0), f"{r:.1f}" + (" (overbought)" if r > 70 else " (oversold)" if r < 30 else ""))
        add("Supertrend 10,3", int(last["st_dir"]), "Up-trend" if last["st_dir"] == 1 else "Down-trend")
    if piv:
        if spot > piv["TC"]:
            add("CPR", 1, "Spot above CPR")
        elif spot < piv["BC"]:
            add("CPR", -1, "Spot below CPR")
        else:
            add("CPR", 0, "Spot inside CPR (range)")
    add("PCR (OI)", 1 if pcr > 1.2 else (-1 if pcr < 0.7 else 0), f"{pcr:.2f}")
    add("PCR (Change in OI)", 1 if chg_pcr > 1.2 else (-1 if chg_pcr < 0.8 else 0), f"{chg_pcr:.2f} — fresh writing side")
    add("Max Pain pull", 0, f"{fmt(mp, 0)} ({'above' if mp > spot else 'below'} spot, expiry-day magnet)")
    add("OI walls", 0, f"Support {fmt(pe_sup, 0)} | Resistance {fmt(ce_res, 0)}")
    score = sum(r["Score"] for r in rows)
    maxs = sum(1 for r in rows if r["Factor"] not in ("Max Pain pull", "OI walls"))
    regime = []
    if total_gex is not None:
        regime.append("Positive GEX → range/mean-reversion likely" if total_gex > 0 else "Negative GEX → trending/volatile moves likely")
    if flip:
        regime.append(f"Gamma flip ≈ {fmt(flip, 0)}")
    if vix:
        regime.append(f"India VIX {vix:.2f} → " + ("low vol, option selling favoured" if vix < 13 else "high vol, wide stops" if vix > 18 else "normal vol"))
    if score >= 3:
        bias, col = "BULLISH", "g"
    elif score <= -3:
        bias, col = "BEARISH", "r"
    else:
        bias, col = "SIDEWAYS / NO CLEAR EDGE", "y"
    return pd.DataFrame(rows), score, maxs, bias, col, regime


def _secret(name):
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


# ------------------------------------------------------------------ password lock (for cloud / phone use)
APP_PW = os.getenv("APP_PASSWORD", "") or _secret("APP_PASSWORD")
if APP_PW and not st.session_state.get("auth_ok"):
    st.markdown("## 🔒 Trading Dashboard")
    pw = st.text_input("Password daalo", type="password")
    if pw:
        if pw == APP_PW:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Galat password")
    st.stop()

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("## ⚙️ Setup")
demo = st.sidebar.toggle("Demo mode (no API)", value=False, help="API ke bina dummy data se dashboard dekhne ke liye")


cid = st.sidebar.text_input("Dhan Client ID", value=os.getenv("DHAN_CLIENT_ID", "") or _secret("DHAN_CLIENT_ID"), disabled=demo)
tok = st.sidebar.text_input("Dhan Access Token", type="password",
                            value=os.getenv("DHAN_ACCESS_TOKEN", "") or _secret("DHAN_ACCESS_TOKEN"), disabled=demo)
und_name = st.sidebar.selectbox("Underlying", list(UNDERLYINGS))
U = UNDERLYINGS[und_name]

if not demo and not (cid and tok):
    st.title("📈 Index Options Command Center")
    st.info("Sidebar mein Dhan Client ID aur Access Token daalo (ya Demo mode on karo). Token Dhan web → Profile → Access DhanHQ APIs se milta hai.")
    st.stop()

err_box = []
expiries = []
if demo:
    base = now_ist().date()
    expiries = [(base + dt.timedelta(days=(3 - base.weekday()) % 7 + 7 * i)).isoformat() for i in range(4)]
else:
    try:
        expiries = api_expiries(cid, tok, U["id"], U["seg"])
    except Exception as e:
        st.error(f"Expiry list fetch failed: {e}")
        st.stop()
expiry = st.sidebar.selectbox("Expiry", expiries)

fut = None if demo else nearest_future(und_name)
lot_default = fut[2] if fut else U["lot"]
lot = st.sidebar.number_input("Lot size", min_value=1, value=int(lot_default), help="Scrip master se auto; zarurat ho to badlo")
oi_in_lots = st.sidebar.toggle("OI is in contracts (lots)", value=False,
                               help="Dhan OI usually quantity mein deta hai. Agar contracts mein ho to on karo — GEX sahi scale hoga.")
interval = st.sidebar.selectbox("Candle interval (min)", [1, 3, 5, 15, 25, 60], index=2)
n_strikes = st.sidebar.slider("Strikes around ATM", 5, 30, 12)
st.sidebar.markdown("### 🎯 Trade settings")
capital = st.sidebar.number_input("Trading capital (₹)", min_value=1000, value=200000, step=10000)
risk_pct = st.sidebar.slider("Risk per trade (% of capital)", 0.25, 5.0, 1.0, 0.25)
rr = st.sidebar.slider("Target R:R", 1.0, 4.0, 2.0, 0.5)
chart_days = st.sidebar.slider("Chart days", 1, 5, 2)
strict = st.sidebar.slider("Signal strictness (candle score)", 4, 9, 6, help="Zyada = kam lekin strong signals")
st.sidebar.markdown("---")
auto = st.sidebar.toggle("Auto refresh", value=os.getenv("DASH_AUTO", "1") == "1")
refresh = st.sidebar.slider("Refresh every (sec)", 5, 120, 15)
st.sidebar.caption("Dhan Option Chain limit: 1 request / 3 sec.")

# ------------------------------------------------------------------ fetch
T = years_to_expiry(expiry)
spot, chain, vix, tech_src = None, pd.DataFrame(), None, ""
try:
    js = demo_chain(T=T, step=U["step"], spot={"NIFTY": 25000.0, "SENSEX": 82000.0, "BANKNIFTY": 56000.0}[und_name]) if demo \
        else api_chain(cid, tok, U["id"], U["seg"], expiry)
    spot, chain = parse_chain(js)
except Exception as e:
    st.error(f"Option chain fetch failed: {e}")
    st.stop()
if chain.empty:
    st.warning("Option chain khaali aaya — market band ho sakta hai ya expiry galat.")
    st.stop()
chain = fill_missing_greeks(chain, spot, T)

try:
    vix = 13.4 if demo else api_ltp(cid, tok, (("IDX_I", (VIX_ID,)),)).get(("IDX_I", VIX_ID))
except Exception as e:
    err_box.append(f"VIX: {e}")

tech = None
try:
    if demo:
        idx_c = demo_candles(spot, interval=interval, with_volume=False)
        fut_c = demo_candles(spot + 60, interval=interval, with_volume=True)
    else:
        idx_c = api_intraday(cid, tok, U["id"], U["seg"], "INDEX", interval, 7)
        fut_c = api_intraday(cid, tok, fut[0], fut[1], "FUTIDX", interval, 7, with_oi=True) if fut else pd.DataFrame()
    if not fut_c.empty:
        tech, tech_src = add_indicators(fut_c), "Current-month future (real volume VWAP)"
    elif not idx_c.empty:
        tech, tech_src = add_indicators(idx_c), "Spot index"
except Exception as e:
    err_box.append(f"Candles: {e}")
    idx_c = pd.DataFrame()

piv = pivots_from(idx_c) if idx_c is not None and not idx_c.empty else None

# ------------------------------------------------------------------ analytics
step = U["step"]
atm = float(chain.iloc[(chain["strike"] - spot).abs().argmin()]["strike"])
atm_row = chain[chain["strike"] == atm].iloc[0]
pcr = chain["pe_oi"].sum() / max(chain["ce_oi"].sum(), 1)
ce_chg_pos, pe_chg_pos = chain["ce_chg_oi"].clip(lower=0).sum(), chain["pe_chg_oi"].clip(lower=0).sum()
chg_pcr = pe_chg_pos / max(ce_chg_pos, 1)
mp, pains = max_pain(chain)
ce_res = float(chain.loc[chain["ce_oi"].idxmax(), "strike"])
pe_sup = float(chain.loc[chain["pe_oi"].idxmax(), "strike"])
ce_res_chg = float(chain.loc[chain["ce_chg_oi"].idxmax(), "strike"])
pe_sup_chg = float(chain.loc[chain["pe_chg_oi"].idxmax(), "strike"])
total_gex, flip = gex_profile(chain, spot, lot if oi_in_lots else 1, T)
atm_iv = np.nanmean([atm_row["ce_iv"], atm_row["pe_iv"]])
straddle = (atm_row["ce_ltp"] or 0) + (atm_row["pe_ltp"] or 0)
day_move = spot * (atm_iv / 100) * math.sqrt(1 / 365) if atm_iv == atm_iv else np.nan
sig_df, score, maxs, bias, bias_col, regime = signal_engine(spot, tech, piv, pcr, chg_pcr, mp, total_gex, flip, vix, ce_res, pe_sup)
# ---- session snapshots for intraday PCR / straddle / VIX trends
snap_key = f"snaps_{und_name}_{expiry}_{demo}"
snaps = st.session_state.setdefault(snap_key, [])
if not snaps or (t_now_ := now_ist().timestamp()) - snaps[-1]["ts"] >= 5:
    snaps.append({"ts": now_ist().timestamp(), "time": now_ist(), "spot": spot, "pcr": pcr, "chg_pcr": chg_pcr,
                  "straddle": straddle, "atm_iv": atm_iv, "vix": vix})
    del snaps[:-500]
scores = candle_scores(tech) if tech is not None and len(tech) > 30 else None
trades = generate_trades(tech, scores, th=strict, rr=rr) if scores is not None else []
of_rows, oc_score = option_flow(chain, spot, atm, step, snaps, pcr, chg_pcr)
# ---- composite table + bias (candle factors on latest candle + option-chain layer + CPR)
comp = []
if scores is not None:
    lastc = tech.iloc[-1]
    reads = {"VWAP": f"{fmt(lastc['close'])} vs {fmt(lastc['vwap'])}", "EMA 9/21": f"{fmt(lastc['ema9'])} / {fmt(lastc['ema21'])}",
             "Supertrend": "Up" if lastc["st_dir"] == 1 else "Down", "RSI": f"{lastc['rsi']:.1f}", "MACD": f"hist {lastc['macd_h']:+.2f}",
             "DI+ vs DI−": f"{lastc['pdi']:.1f} / {lastc['mdi']:.1f} · ADX {lastc['adx']:.1f}",
             "Futures OI buildup": str(lastc["buildup"]), "Volume spike": f"RVOL {float(lastc['rvol']):.2f}x"}
    for k_ in CANDLE_FACTORS:
        comp.append({"Layer": "Price / Flow", "Factor": k_, "Score": int(scores[k_].iloc[-1]), "Reading": reads[k_]})
for r_ in of_rows:
    comp.append({"Layer": "Option chain", **r_})
if piv:
    comp.append({"Layer": "Levels", "Factor": "CPR", "Score": 1 if spot > piv["TC"] else (-1 if spot < piv["BC"] else 0),
                 "Reading": "above CPR" if spot > piv["TC"] else ("below CPR" if spot < piv["BC"] else "inside CPR")})
sig_df = pd.DataFrame(comp)
score = int(sig_df["Score"].sum()) if len(sig_df) else 0
maxs = int(len(sig_df) + 1)
if score >= 6:
    bias, bias_col = "BULLISH", "g"
elif score <= -6:
    bias, bias_col = "BEARISH", "r"
else:
    bias, bias_col = "SIDEWAYS / NO CLEAR EDGE", "y"
if tech is not None and len(tech):
    adx_now = float(tech["adx"].iloc[-1])
    regime.insert(0, f"ADX {adx_now:.0f} → " + ("strong trend" if adx_now >= 25 else "weak trend" if adx_now >= 18 else "no trend / chop"))
if len(snaps) >= 6:
    d_st = snaps[-1]["straddle"] - snaps[-6]["straddle"]
    regime.append("ATM straddle " + ("expanding → buyers' market" if d_st > 0 else "decaying → sellers' market"))
basis = float(tech["close"].iloc[-1] - spot) if tech is not None and len(tech) else 0.0
live = trades[-1] if trades and trades[-1]["outcome"] == "OPEN" else None

# ------------------------------------------------------------------ header
t_now = now_ist()
mkt_open = t_now.weekday() < 5 and dt.time(9, 15) <= t_now.time() <= dt.time(15, 30)
st.markdown(f"# 📈 {und_name} Options Command Center")
st.caption(f"{'🟢 Market open' if mkt_open else '🔴 Market closed'} · Updated {t_now:%d %b %Y, %H:%M:%S} IST · Expiry {expiry} · "
           f"{'DEMO DATA' if demo else 'Live via Dhan API'}")

c = st.columns(6)
c[0].markdown(card("Spot", fmt(spot), f"ATM {fmt(atm, 0)}", "b"), unsafe_allow_html=True)
c[1].markdown(card("India VIX", fmt(vix), "volatility gauge", "p"), unsafe_allow_html=True)
c[2].markdown(card("PCR (OI)", fmt(pcr), f"Chg-OI PCR {fmt(chg_pcr)}", "g" if pcr > 1 else "r"), unsafe_allow_html=True)
c[3].markdown(card("Max Pain", fmt(mp, 0), f"{fmt(mp - spot, 0)} pts from spot", "y"), unsafe_allow_html=True)
c[4].markdown(card("ATM IV", fmt(atm_iv) + "%", f"1-day move ±{fmt(day_move, 0)}", "p"), unsafe_allow_html=True)
c[5].markdown(card("ATM Straddle", fmt(straddle), f"Expiry range {fmt(atm - straddle, 0)}–{fmt(atm + straddle, 0)}", "b"), unsafe_allow_html=True)

st.write("")
c = st.columns(4)
c[0].markdown(card("OI Support (PE)", fmt(pe_sup, 0), f"Fresh PE writing @ {fmt(pe_sup_chg, 0)}", "g"), unsafe_allow_html=True)
c[1].markdown(card("OI Resistance (CE)", fmt(ce_res, 0), f"Fresh CE writing @ {fmt(ce_res_chg, 0)}", "r"), unsafe_allow_html=True)
c[2].markdown(card("Net GEX (₹ Cr / 1%)", fmt(total_gex), "Positive = range · Negative = trend", "g" if total_gex > 0 else "r"), unsafe_allow_html=True)
c[3].markdown(card("Gamma Flip ≈", fmt(flip, 0) if flip else "—", "Iske neeche moves tez/volatile hote hain", "y"), unsafe_allow_html=True)

st.write("")
st.markdown(
    f"<div class='bias'>🧭 <b>Overall bias:</b> <span class='{bias_col}' style='font-weight:800'>{bias}</span> "
    f"(score {score:+d} / ±{maxs})<br><span style='color:#b9bde0'>{' · '.join(regime)}</span><br>"
    + (f"🎯 <b>Active signal:</b> <span class='{'g' if live['side'] == 'BUY' else 'r'}' style='font-weight:800'>"
       f"{'BUY CE' if live['side'] == 'BUY' else 'BUY PE'}</span> @ {live['time']:%H:%M} · Entry {fmt(live['entry'])} · "
       f"SL {fmt(live['sl'])} · Target {fmt(live['tg'])}" if live else "🎯 Abhi koi active signal nahi — wait karo")
    + "</div>",
    unsafe_allow_html=True,
)
if err_box:
    with st.expander("⚠️ Kuch data fetch nahi hua"):
        for e in err_box:
            st.write(e)

# ------------------------------------------------------------------ tabs
tabs = st.tabs(["🎯 Chart Signals", "🧭 Signals", "🔗 Option Chain", "📊 OI & GEX", "📉 Technicals", "💧 Liquidity", "🛡️ Risk & Trade Plan"])
DARK = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10))
win = chain[(chain["strike"] >= atm - n_strikes * step) & (chain["strike"] <= atm + n_strikes * step)].copy()

with tabs[0]:
    if tech is None or tech.empty:
        st.warning("Candle data nahi mila.")
    else:
        days_list = sorted(tech["date"].unique())[-chart_days:]
        tv = tech[tech["date"].isin(days_list)]
        t0 = tv["time"].iloc[0]
        f = go.Figure()
        f.add_candlestick(x=tv["time"], open=tv["open"], high=tv["high"], low=tv["low"], close=tv["close"], name="Price",
                          increasing_line_color="#34e5a0", decreasing_line_color="#ff6b8a")
        f.add_scatter(x=tv["time"], y=tv["ema9"], name="EMA 9", line=dict(color="#6cc6ff", width=1.2))
        f.add_scatter(x=tv["time"], y=tv["ema21"], name="EMA 21", line=dict(color="#c49bff", width=1.2))
        f.add_scatter(x=tv["time"], y=tv["vwap"], name=tech.attrs.get("vwap_kind", "VWAP"), line=dict(color="#ffd166", width=1.5, dash="dot"))
        for band, nm in (("vwap_u2", "VWAP +2σ"), ("vwap_l2", "VWAP −2σ")):
            f.add_scatter(x=tv["time"], y=tv[band], name=nm, line=dict(color="rgba(255,209,102,0.35)", width=1), showlegend=False)
        f.add_scatter(x=tv["time"], y=tv["st"], name="Supertrend", mode="markers",
                      marker=dict(size=3, color=np.where(tv["st_dir"] == 1, "#34e5a0", "#ff6b8a")))
        # option-chain levels on the chart (shifted by futures basis)
        lv_lines = [("PE Support (OI)", pe_sup, "#34e5a0"), ("CE Resistance (OI)", ce_res, "#ff6b8a"), ("Max Pain", mp, "#ffd166")]
        if flip:
            lv_lines.append(("Gamma Flip", flip, "#c49bff"))
        if piv:
            lv_lines += [("CPR TC", piv["TC"], "#9aa0c3"), ("CPR BC", piv["BC"], "#9aa0c3"), ("PDH", piv["PDH"], "#ff9f80"), ("PDL", piv["PDL"], "#7cf0c4")]
        lo, hi = tv["low"].min(), tv["high"].max()
        pad = (hi - lo) * 0.6
        for name, lvl, colr in lv_lines:
            y = lvl + basis
            if lo - pad <= y <= hi + pad:
                f.add_hline(y=y, line_dash="dash", line_width=1, line_color=colr, annotation_text=name, annotation_position="top left",
                            annotation_font_size=10, annotation_font_color=colr)
        vis = [t for t in trades if t["time"] >= t0]
        for t in vis:
            buy = t["side"] == "BUY"
            colr = "#34e5a0" if buy else "#ff6b8a"
            f.add_shape(type="line", x0=t["time"], x1=t["exit_time"], y0=t["sl"], y1=t["sl"], line=dict(color="#ff6b8a", width=2))
            f.add_shape(type="line", x0=t["time"], x1=t["exit_time"], y0=t["tg"], y1=t["tg"], line=dict(color="#34e5a0", width=2))
            f.add_shape(type="line", x0=t["time"], x1=t["exit_time"], y0=t["entry"], y1=t["entry"], line=dict(color="#6cc6ff", width=1, dash="dot"))
            row = tv[tv["time"] == t["time"]]
            yv = (row["low"].iloc[0] - row["atr"].iloc[0] * 0.6) if buy else (row["high"].iloc[0] + row["atr"].iloc[0] * 0.6)
            icon = {"TARGET": "✅", "SL": "❌", "DAY END": "⏹", "OPEN": "🔵"}[t["outcome"]]
            f.add_scatter(x=[t["time"]], y=[yv], mode="markers+text", showlegend=False,
                          marker=dict(symbol="triangle-up" if buy else "triangle-down", size=15, color=colr, line=dict(color="white", width=1)),
                          text=[f"{'BUY CE' if buy else 'BUY PE'} {icon}"], textposition="bottom center" if buy else "top center",
                          textfont=dict(color=colr, size=11),
                          hovertext=f"{'BUY CE' if buy else 'BUY PE'} @ {t['entry']:.1f}<br>SL {t['sl']:.1f} · Target {t['tg']:.1f}<br>Result: {t['outcome']} ({t['R']:+.2f}R)",
                          hoverinfo="text")
        f.update_layout(height=620, xaxis_rangeslider_visible=False, title="Buy/Sell signals with SL (red) & Target (green)",
                        legend=dict(orientation="h", y=-0.12), **DARK)
        f.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[15.5, 9.25], pattern="hour")])
        st.plotly_chart(f, width="stretch")

        # ---- live trade card with option conversion
        if live:
            buy = live["side"] == "BUY"
            oc_dir = oc_score if buy else -oc_score
            agree, against = oc_dir >= 2, oc_dir <= -1
            conf = "HIGH" if agree and live["score"] * (1 if buy else -1) >= strict + 1 else ("LOW" if against else "MEDIUM")
            conf_col = {"HIGH": "g", "MEDIUM": "y", "LOW": "r"}[conf]
            sd_ = "ce" if buy else "pe"
            k = atm - step if buy else atm + step   # 1 strike ITM
            rowk = chain[chain["strike"] == k]
            rowk = rowk.iloc[0] if len(rowk) else atm_row
            dlt = abs(rowk[f"{sd_}_delta"]) if rowk[f"{sd_}_delta"] == rowk[f"{sd_}_delta"] else 0.55
            prem = float(rowk[f"{sd_}_ask"] or rowk[f"{sd_}_ltp"] or 0)
            p_sl = max(prem - dlt * live["risk"], 0.05)
            p_tg = prem + dlt * live["risk"] * rr
            lots_sig = int((capital * risk_pct / 100) // max((prem - p_sl) * lot, 1))
            lots_sig = max(0, min(lots_sig, int(capital // max(prem * lot, 1))))
            st.markdown(
                f"<div class='bias'>🎯 <b>{'BUY CE' if buy else 'BUY PE'}</b> — signal candle {live['time']:%d %b %H:%M} · "
                f"Confidence <span class='{conf_col}' style='font-weight:800'>{conf}</span><br>"
                f"<b>Chart levels:</b> Entry {fmt(live['entry'])} · SL {fmt(live['sl'])} · Target {fmt(live['tg'])} · "
                f"abhi {live['R']:+.2f}R<br>"
                f"<b>Option ({int(rowk['strike'])} {sd_.upper()}, Δ {dlt:.2f}):</b> Entry ≈ ₹{fmt(prem)} · SL ≈ ₹{fmt(p_sl)} · "
                f"Target ≈ ₹{fmt(p_tg)} · Size {lots_sig} lot(s) ({lots_sig * lot} qty)</div>",
                unsafe_allow_html=True)
            notes = []
            wall = (ce_res + basis) if buy else (pe_sup + basis)
            if (buy and live["tg"] > wall > live["entry"]) or (not buy and live["tg"] < wall < live["entry"]):
                notes.append(f"Target OI wall ({fmt(wall - basis, 0)}) ke paar hai — wahan partial profit book karna behtar.")
            if against:
                notes.append("Option chain (PCR / fresh writing) is signal ke against hai — skip ya half size.")
            if total_gex > 0:
                notes.append("Positive GEX: range market — target jaldi book karo, trailing mat karo.")
            if lots_sig == 0:
                notes.append("Risk budget mein 1 lot fit nahi hota — skip karo ya sidebar mein risk badhao.")
            for n_ in notes:
                st.warning(n_)
            why = [{"Layer": "Price / Flow (signal candle)", "Factor": k_, "Score": int(scores[k_].iloc[live["i"]])} for k_ in CANDLE_FACTORS]
            why += [{"Layer": "Option chain (live)", "Factor": r_["Factor"], "Score": r_["Score"]} for r_ in of_rows]
            with st.expander(f"🔍 Ye signal kyun aaya? (candle score {live['score']:+d}, option score {oc_score:+d})", expanded=True):
                st.dataframe(pd.DataFrame(why), width="stretch", hide_index=True,
                             column_config={"Score": st.column_config.NumberColumn(format="%+d")})
        else:
            st.info("Abhi koi active signal nahi. Naya 🔺/🔻 aane ka wait karo — no signal = no trade.")

        closed = [t for t in trades if t["outcome"] != "OPEN"]
        if closed:
            wins = sum(1 for t in closed if t["R"] > 0)
            c = st.columns(4)
            c[0].markdown(card("Signals (loaded days)", f"{len(closed)}", "closed trades", "b"), unsafe_allow_html=True)
            c[1].markdown(card("Win rate", f"{wins / len(closed) * 100:.0f}%", f"{wins} jeete · {len(closed) - wins} haare", "g" if wins / len(closed) >= 0.4 else "r"), unsafe_allow_html=True)
            c[2].markdown(card("Total", f"{sum(t['R'] for t in closed):+.1f}R", "R = 1 stop-loss ka risk", "y"), unsafe_allow_html=True)
            c[3].markdown(card("Avg per trade", f"{np.mean([t['R'] for t in closed]):+.2f}R", "positive = edge", "p"), unsafe_allow_html=True)
            with st.expander("Saare signals ki list"):
                st.dataframe(pd.DataFrame([{"Time": f"{t['time']:%d %b %H:%M}", "Signal": "BUY CE" if t["side"] == "BUY" else "BUY PE",
                                            "Entry": round(t["entry"], 1), "SL": round(t["sl"], 1), "Target": round(t["tg"], 1),
                                            "Result": t["outcome"], "R": round(t["R"], 2)} for t in reversed(trades)]),
                             width="stretch", hide_index=True)
        st.caption(f"Signal rule: har candle par 8 factors (VWAP, EMA, Supertrend, RSI, MACD, DI+/DI−, Futures OI buildup ±2, Volume spike) ka score banta hai. "
                   f"Score +{strict} cross kare = BUY CE, −{strict} cross kare = BUY PE — sirf jab ADX ≥ 18 (trend ho), RSI extreme na ho, aur price VWAP ±2σ ke andar ho (chase nahi). "
                   "Phir live option-chain layer (PCR, fresh writing near ATM, order-book pressure, PCR/VIX trend) se Confidence tay hota hai. "
                   "9:30 se pehle / 2:45 ke baad naya signal nahi. SL = pichhli 4 candles ka swing (0.6–2 ATR), Target = SL × R:R. Past win-rate future ki guarantee nahi.")

with tabs[1]:
    st.dataframe(sig_df, width="stretch", hide_index=True,
                 column_config={"Score": st.column_config.NumberColumn(format="%+d")})
    st.markdown("**Kaise padhein:** +1 bullish, −1 bearish, 0 neutral (Futures OI buildup ±2 tak). Total ≥ +6 bullish, ≤ −6 bearish, beech mein no-trade/range zone. "
                "OI buildup: Long Buildup (price↑ OI↑) sabse bullish, Short Buildup (price↓ OI↑) sabse bearish, Short Covering / Long Unwinding kamzor moves. "
                "Positive GEX mein breakouts aksar fail hote hain (range trading), negative GEX mein trend tez chalta hai.")
    if piv:
        st.markdown(f"**Key levels (prev day {piv['date']}):**")
        lv = pd.DataFrame([{"Level": k, "Value": round(piv[k], 1)} for k in ["R3", "R2", "R1", "PDH", "TC", "P", "BC", "PDL", "S1", "S2", "S3"]])
        lv["Distance from spot"] = (lv["Value"] - spot).round(1)
        st.dataframe(lv, width="stretch", hide_index=True)
        st.caption(f"CPR width {piv['cpr_width%']:.2f}% — narrow CPR (<0.25%) = trending day ki sambhavna, wide = range day.")

with tabs[2]:
    show = win[["ce_oi", "ce_chg_oi", "ce_vol", "ce_iv", "ce_delta", "ce_theta", "ce_ltp", "strike",
                "pe_ltp", "pe_theta", "pe_delta", "pe_iv", "pe_vol", "pe_chg_oi", "pe_oi"]].copy()
    show.columns = ["CE OI", "CE ΔOI", "CE Vol", "CE IV", "CE Δ", "CE Θ", "CE LTP", "STRIKE",
                    "PE LTP", "PE Θ", "PE Δ", "PE IV", "PE Vol", "PE ΔOI", "PE OI"]
    mx = float(max(show["CE OI"].max(), show["PE OI"].max(), 1))
    st.dataframe(
        show.style.apply(lambda r: ["background-color: rgba(255,209,102,0.18)" if r["STRIKE"] == atm else "" for _ in r], axis=1)
        .format(precision=2, thousands=","),
        width="stretch", hide_index=True, height=min(38 * len(show) + 40, 900),
        column_config={"CE OI": st.column_config.ProgressColumn(min_value=0, max_value=mx, format="%.0f"),
                       "PE OI": st.column_config.ProgressColumn(min_value=0, max_value=mx, format="%.0f")},
    )
    st.caption("Peeli row = ATM. ΔOI = aaj ka change in OI. Θ = per-day time decay (₹ per unit).")

with tabs[3]:
    f = go.Figure()
    f.add_bar(x=win["strike"], y=win["ce_oi"], name="CE OI", marker_color="#ff6b8a")
    f.add_bar(x=win["strike"], y=win["pe_oi"], name="PE OI", marker_color="#34e5a0")
    f.add_vline(x=spot, line_dash="dash", line_color="#6cc6ff", annotation_text="Spot")
    f.add_vline(x=mp, line_dash="dot", line_color="#ffd166", annotation_text="Max Pain")
    f.update_layout(title="Open Interest by strike", barmode="group", **DARK)
    st.plotly_chart(f, width="stretch")

    f = go.Figure()
    f.add_bar(x=win["strike"], y=win["ce_chg_oi"], name="CE ΔOI", marker_color="#ff9f80")
    f.add_bar(x=win["strike"], y=win["pe_chg_oi"], name="PE ΔOI", marker_color="#7cf0c4")
    f.add_vline(x=spot, line_dash="dash", line_color="#6cc6ff")
    f.update_layout(title="Change in OI (fresh writing / unwinding)", barmode="group", **DARK)
    st.plotly_chart(f, width="stretch")

    f = go.Figure()
    f.add_bar(x=win["strike"], y=win["gex_cr"], marker_color=np.where(win["gex_cr"] >= 0, "#34e5a0", "#ff6b8a"), name="GEX")
    f.add_vline(x=spot, line_dash="dash", line_color="#6cc6ff", annotation_text="Spot")
    if flip:
        f.add_vline(x=flip, line_dash="dot", line_color="#ffd166", annotation_text="Flip")
    f.update_layout(title="Gamma Exposure by strike (₹ Cr per 1% move, dealer-long-call / short-put assumption)", **DARK)
    st.plotly_chart(f, width="stretch")

    if len(snaps) >= 3:
        sn = pd.DataFrame(snaps)
        f = make_subplots(specs=[[{"secondary_y": True}]])
        f.add_scatter(x=sn["time"], y=sn["pcr"], name="PCR", line=dict(color="#34e5a0"), secondary_y=False)
        f.add_scatter(x=sn["time"], y=sn["straddle"], name="ATM Straddle ₹", line=dict(color="#ffd166"), secondary_y=True)
        f.update_layout(title="Session trend: PCR & ATM straddle (app khulne ke baad se)", **DARK)
        st.plotly_chart(f, width="stretch")
    f = go.Figure()
    f.add_scatter(x=win["strike"], y=win["ce_iv"], name="CE IV", line=dict(color="#ff6b8a"))
    f.add_scatter(x=win["strike"], y=win["pe_iv"], name="PE IV", line=dict(color="#34e5a0"))
    f.add_vline(x=spot, line_dash="dash", line_color="#6cc6ff")
    f.update_layout(title="IV smile / skew", **DARK)
    st.plotly_chart(f, width="stretch")
    skew = win.loc[win["strike"] <= atm - 4 * step, "pe_iv"].mean() - win.loc[win["strike"] >= atm + 4 * step, "ce_iv"].mean()
    st.caption(f"Put–call skew (OTM PE IV − OTM CE IV): {fmt(skew)} — zyada positive = downside hedge ki demand / fear.")

with tabs[4]:
    if tech is None or tech.empty:
        st.warning("Candle data nahi mila.")
    else:
        st.caption(f"Source: {tech_src}")
        last_days = sorted(tech["date"].unique())[-2:]
        tv = tech[tech["date"].isin(last_days)]
        has_oi = tv["oi"].fillna(0).sum() > 0 if "oi" in tv else False
        f = make_subplots(rows=3 if has_oi else 2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2] if has_oi else [0.75, 0.25], vertical_spacing=0.03)
        f.add_candlestick(x=tv["time"], open=tv["open"], high=tv["high"], low=tv["low"], close=tv["close"], name="Price",
                          increasing_line_color="#34e5a0", decreasing_line_color="#ff6b8a", row=1, col=1)
        f.add_scatter(x=tv["time"], y=tv["ema9"], name="EMA 9", line=dict(color="#6cc6ff", width=1.3), row=1, col=1)
        f.add_scatter(x=tv["time"], y=tv["ema21"], name="EMA 21", line=dict(color="#c49bff", width=1.3), row=1, col=1)
        f.add_scatter(x=tv["time"], y=tv["vwap"], name=tech.attrs.get("vwap_kind", "VWAP"), line=dict(color="#ffd166", width=1.6, dash="dot"), row=1, col=1)
        f.add_scatter(x=tv["time"], y=tv["st"], name="Supertrend", mode="markers", marker=dict(size=3, color=np.where(tv["st_dir"] == 1, "#34e5a0", "#ff6b8a")), row=1, col=1)
        if piv and tech_src.startswith("Spot"):
            for k, colr in (("TC", "#9aa0c3"), ("BC", "#9aa0c3"), ("R1", "#ff6b8a"), ("S1", "#34e5a0"), ("PDH", "#ff9f80"), ("PDL", "#7cf0c4")):
                f.add_hline(y=piv[k], line_dash="dot", line_color=colr, annotation_text=k, row=1, col=1)
        f.add_scatter(x=tv["time"], y=tv["rsi"], name="RSI", line=dict(color="#c49bff"), row=2, col=1)
        f.add_hline(y=70, line_dash="dot", line_color="#ff6b8a", row=2, col=1)
        f.add_hline(y=30, line_dash="dot", line_color="#34e5a0", row=2, col=1)
        if has_oi:
            bcol = tv["buildup"].map({"Long Buildup": "#34e5a0", "Short Covering": "#7cf0c4", "Short Buildup": "#ff6b8a",
                                      "Long Unwinding": "#ff9f80"}).fillna("#9aa0c3")
            f.add_bar(x=tv["time"], y=tv["oi"].diff().fillna(0), marker_color=bcol, name="Fut ΔOI (buildup)", row=3, col=1)
        f.update_layout(height=760 if has_oi else 650, xaxis_rangeslider_visible=False,
                        title="Price · EMA · VWAP · Supertrend · RSI" + (" · Futures OI buildup" if has_oi else ""), **DARK)
        f.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[15.5, 9.25], pattern="hour")])
        st.plotly_chart(f, width="stretch")
        if has_oi:
            last_b = tv["buildup"].iloc[-1]
            st.markdown(f"**Abhi futures mein:** {last_b} · OI bars: 🟢 Long Buildup · 🩵 Short Covering · 🔴 Short Buildup · 🟠 Long Unwinding")
        if not tech_src.startswith("Spot"):
            st.caption("Chart futures ka hai (basis ki wajah se spot se thoda upar). CPR/pivot levels spot index par hain — Signals tab dekho.")

with tabs[5]:
    lq = liquidity_table(chain, atm, step, n_strikes)
    out = lq[["strike", "ce_bid", "ce_ask", "ce_spread%", "ce_vol", "CE liquidity", "pe_bid", "pe_ask", "pe_spread%", "pe_vol", "PE liquidity"]]
    out.columns = ["Strike", "CE Bid", "CE Ask", "CE Spread %", "CE Vol", "CE", "PE Bid", "PE Ask", "PE Spread %", "PE Vol", "PE"]
    st.dataframe(out.style.format(precision=2, thousands=","), width="stretch", hide_index=True)
    st.markdown("**Rule:** sirf 🟢/🟡 strikes trade karo. Wide spread mein entry-exit dono pe slippage lagta hai — "
                "deep OTM sasta dikhta hai par nikalna mushkil hota hai.")

with tabs[6]:
    a, b = st.columns(2)
    with a:
        st.caption(f"Capital ₹{capital:,.0f} · Risk {risk_pct}% · R:R {rr} (sidebar se badlo)")
        side = st.radio("Option", ["CE", "PE"], horizontal=True)
        k_opts = list(win["strike"].astype(int))
        strike = st.selectbox("Strike", k_opts, index=k_opts.index(int(atm)) if int(atm) in k_opts else 0)
    r = chain[chain["strike"] == strike].iloc[0]
    s = side.lower()
    entry_default = float(r[f"{s}_ask"]) if r[f"{s}_ask"] == r[f"{s}_ask"] and r[f"{s}_ask"] > 0 else float(r[f"{s}_ltp"] or 1)
    with b:
        entry = st.number_input("Entry premium (₹)", min_value=0.05, value=round(max(entry_default, 0.05), 2))
        sl_pct = st.slider("Stop-loss on premium (%)", 5, 60, 25)
    sl_pts = entry * sl_pct / 100
    risk_amt = capital * risk_pct / 100
    lots = int(risk_amt // (sl_pts * lot)) if sl_pts > 0 else 0
    afford = int(capital // (entry * lot)) if entry > 0 else 0
    lots_final = max(0, min(lots, afford))
    qty = lots_final * lot
    theta_day = (r[f"{s}_theta"] or 0) * qty
    delta_pos = (r[f"{s}_delta"] or 0) * qty
    spot_move_for_target = (sl_pts * rr) / abs(r[f"{s}_delta"]) if r[f"{s}_delta"] else np.nan
    c = st.columns(4)
    c[0].markdown(card("Lots", f"{lots_final}", f"Qty {qty} · capital limit {afford} lots", "b"), unsafe_allow_html=True)
    c[1].markdown(card("Stop-loss", fmt(entry - sl_pts), f"Max loss ₹{fmt(sl_pts * qty, 0)}", "r"), unsafe_allow_html=True)
    c[2].markdown(card("Target", fmt(entry + sl_pts * rr), f"Profit ₹{fmt(sl_pts * rr * qty, 0)}", "g"), unsafe_allow_html=True)
    c[3].markdown(card("Theta / day", f"₹{fmt(theta_day, 0)}", f"Position Δ {fmt(delta_pos, 0)}", "y"), unsafe_allow_html=True)
    st.write("")
    st.markdown(f"Target ke liye {und_name} ko lagbhag **{fmt(spot_move_for_target, 0)} points** "
                f"{'upar' if side == 'CE' else 'neeche'} jaana hoga (delta ke hisaab se, gamma/IV change ignore). "
                f"Aaj ka IV-based expected move ±{fmt(day_move, 0)} hai — agar zaroori move isse bada hai to setup kamzor hai.")
    warn = []
    if r[f"{s}_ask"] and r[f"{s}_bid"] and (r[f"{s}_ask"] - r[f"{s}_bid"]) / max(entry, 0.05) > 0.03:
        warn.append("Spread wide hai (>3%) — limit order use karo.")
    if T * 365 < 1:
        warn.append("Expiry day: theta aur gamma dono bahut tez — chhota size rakho.")
    if score * (1 if side == "CE" else -1) < 0:
        warn.append(f"Overall bias ({bias}) aapke trade ke against hai.")
    if lots_final == 0:
        warn.insert(0, f"Risk budget ₹{fmt(risk_amt, 0)} mein 1 lot bhi fit nahi hota (1 lot ka risk ₹{fmt(sl_pts * lot, 0)}). Risk % badhao, SL tight karo, ya trade skip karo.")
    for w in warn:
        st.warning(w)

st.markdown("---")
st.caption("⚠️ Sirf analysis tool — investment advice nahi. Data delays / API errors possible. Hamesha stop-loss ke saath trade karo.")

if auto:
    time.sleep(refresh)
    st.rerun()
