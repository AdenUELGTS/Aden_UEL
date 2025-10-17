# disaster_forecast_app.py
# Forecast monthly counts of natural disasters (EONET + optional USGS)
# Models: Seasonal-Naive, SARIMAX, SARIMAX+Fourier, ETS, RandomForest(lags),
#         Ridge(lags), Poisson(lags), NegativeBinomial(lags), Prophet*
# Metrics: RMSE, MAE, R²  (selection metric configurable)
# History default: from year 2000
# *Prophet is optional: pip install prophet

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from datetime import date
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# Statsmodels
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Optional Prophet
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(
    page_title="🌏 Natural Disasters Forecast — Multi-Model (Improved)",
    page_icon="🌋",
    layout="wide",
)

# ---------- Global Dark Theme (CSS) ----------
st.markdown("""
<style>
:root{
  --bg: #0b0f19;
  --surface: #0f172a;
  --surface-2: #111827;
  --text: #e5e7eb;
  --muted: #94a3b8;
  --accent: #22c55e;      /* green */
  --accent-2: #60a5fa;    /* blue */
  --danger: #ef4444;
  --border: #1f2937;
  --radius: 12px;
  --fs-base: 14px;
}

/* App background + typography */
html, body, [data-testid="stAppViewContainer"]{
  background: var(--bg) !important;
  color: var(--text) !important;
  font-size: var(--fs-base) !important;
}
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

/* Sidebar */
[data-testid="stSidebar"]{
  background: var(--surface-2) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebarNav"] { color: var(--text) !important; }

/* Headers & hr */
h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }
hr, .stDivider, [role="separator"]{
  border-color: var(--border) !important;
  background: var(--border) !important;
}

/* Buttons */
.stButton>button{
  background: var(--accent) !important;
  color: #0b0f19 !important;
  border: none !important;
  border-radius: var(--radius) !important;
  padding: 0.6rem 1rem !important;
  font-weight: 600 !important;
}
.stButton>button:hover{ filter: brightness(1.08); }

/* Download buttons */
.stDownloadButton>button{
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 0.55rem 0.9rem !important;
}
.stDownloadButton>button:hover{
  border-color: var(--accent) !important; color: var(--accent) !important;
}

/* Inputs (selects, text, date inputs, sliders) */
[data-baseweb="select"]>div, [data-baseweb="input"]>div{
  background: var(--surface) !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
}
input, textarea{
  background: var(--surface) !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
}
.css-10trblm, .stDateInput { color: var(--text) !important; }

/* Metrics */
[data-testid="stMetric"]{
  background: linear-gradient(180deg, rgba(34,197,94,0.09), rgba(96,165,250,0.08));
  padding: 14px !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"]{
  color: var(--text) !important;
}

/* Dataframe/table shells */
[data-testid="stDataFrame"]{
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}
[data-testid="stDataFrame"] * { color: var(--text) !important; }

/* Expander */
[data-testid="stExpander"]{
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}
[data-testid="stExpander"] details>summary { color: var(--text) !important; }

/* Captions */
.small-font, .caption, .stCaption, p, span { color: var(--muted); }

/* Plotly modebar icons */
.js-plotly-plot .plotly .modebar-btn svg { fill: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

st.title("🌏 Natural Disasters — Forecasting Dashboard (from 2000)")
st.caption("Dark theme • Smoothing, spike clipping, SARIMAX+Fourier seasonality, and Negative Binomial for over-dispersed counts.")

TODAY = date.today()

# Reusable Plotly dark layout
PLOTLY_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0b0f19",
    plot_bgcolor="#0f172a",
    font=dict(color="#e5e7eb"),
    xaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
    yaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
)

# =========================
# Region presets — (min_lon, min_lat, max_lon, max_lat)
# =========================
REGIONS: Dict[str, Optional[Tuple[float, float, float, float]]] = {
    "Global": None,
    "Asia": (25.0, -12.0, 180.0, 82.0),
    "Southeast Asia": (92.0, -15.0, 141.0, 25.0),
    "South Asia": (60.0, 0.0, 100.0, 35.0),
    "East Asia": (100.0, 15.0, 180.0, 55.0),
    "Middle East": (25.0, 12.0, 60.0, 42.0),
}

# =========================
# Sidebar — Controls
# =========================
with st.sidebar:
    st.header("Filters & Settings")

    dr = st.date_input(
        "History window (UTC)",
        value=(date(2000, 1, 1), TODAY),
        min_value=date(2000, 1, 1),
        max_value=TODAY,
    )
    start_date, end_date = (
        dr if isinstance(dr, (list, tuple)) and len(dr) == 2 else (date(2000, 1, 1), TODAY)
    )

    region_name = st.selectbox("Region preset", options=list(REGIONS.keys()), index=1)
    bbox = REGIONS[region_name]

    st.markdown("**Categories (EONET)**")
    use_eq = st.checkbox("Earthquakes (EONET)", value=True)
    use_fl = st.checkbox("Floods", value=True)
    use_ss = st.checkbox("Typhoons / Severe Storms", value=True)

    st.divider()
    st.markdown("**USGS Earthquakes (optional, more complete)**")
    include_usgs = st.checkbox("Include USGS Earthquakes", value=True)
    usgs_minmag = st.slider("USGS min magnitude", 3.0, 8.5, 4.5, 0.1)

    st.divider()
    target_choice = st.selectbox(
        "Target series to forecast",
        [
            "All selected categories (combined)",
            "Earthquakes",
            "Floods",
            "Severe Storms",
        ],
        index=0,
    )

    horizon_years = st.slider("Forecast horizon (years)", min_value=1, max_value=5, value=3, step=1)
    backtest_months = st.slider("Backtest window (months)", min_value=6, max_value=36, value=18, step=6)

    st.divider()
    st.markdown("**Make the target easier to predict**")
    target_smoothing = st.checkbox("Use 3-month rolling mean", value=True)
    clip_outliers = st.checkbox("Clip extreme spikes (99.5th pct)", value=True)

    st.divider()
    st.markdown("**Which models to try**")
    use_sn = st.checkbox("Seasonal-Naive(12)", True)
    use_sarimax = st.checkbox("SARIMAX", True)
    use_sarimax_fourier = st.checkbox("SARIMAX + Fourier (K=2)", True)
    use_ets = st.checkbox("Exponential Smoothing (ETS)", True)
    use_rf = st.checkbox("RandomForest(lags)", True)
    use_ridge = st.checkbox("Ridge Regression (lags+seasonal)", True)
    use_poisson = st.checkbox("Poisson Regression (lags+seasonal)", True)
    use_negbin = st.checkbox("Negative Binomial (lags+seasonal)", True)
    use_prophet = st.checkbox("Prophet", HAS_PROPHET)
    if not HAS_PROPHET and use_prophet:
        st.warning("Prophet not installed. Run: pip install prophet")

    st.divider()
    select_metric = st.selectbox("Pick best model by", ["RMSE", "R²"], index=0)

    run_btn = st.button("🚀 Run training & forecasting")

# =========================
# Helpers
# =========================
def _flatten_coords(coords) -> List[List[float]]:
    if not isinstance(coords, list):
        return []
    if len(coords) == 2 and all(isinstance(x, (int, float)) for x in coords):
        return [coords]
    flat, stack = [], [coords]
    while stack:
        cur = stack.pop()
        if isinstance(cur, list):
            if len(cur) == 2 and all(isinstance(x, (int, float)) for x in cur):
                flat.append(cur)
            else:
                stack.extend(cur)
    return flat

def _centroid_lonlat(pairs: List[List[float]]) -> Tuple[Optional[float], Optional[float]]:
    if not pairs:
        return None, None
    df = pd.DataFrame(pairs, columns=["lon", "lat"])
    return float(df["lon"].mean()), float(df["lat"].mean())

def _apply_bbox_filter(df: pd.DataFrame, bbox_tuple: Optional[Tuple[float, float, float, float]]) -> pd.DataFrame:
    if not bbox_tuple or df.empty:
        return df
    min_lon, min_lat, max_lon, max_lat = bbox_tuple
    return df[(df["longitude"].between(min_lon, max_lon)) & (df["latitude"].between(min_lat, max_lat))]

def _canonical_category(eonet_category_titles: List[str]) -> str:
    joined = " | ".join(eonet_category_titles or [])
    if "Earthquake" in joined: return "Earthquakes"
    if "Flood" in joined: return "Floods"
    if ("Severe Storm" in joined) or ("Tropical Cyclone" in joined): return "Severe Storms"
    return "Other"

# =========================
# Data fetchers (cached)
# =========================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_eonet(categories: List[str], start_d: date, end_d: date,
                bbox_tuple: Optional[Tuple[float, float, float, float]]) -> pd.DataFrame:
    params = {"status": "all", "start": start_d.isoformat(), "end": end_d.isoformat()}
    if categories: params["category"] = ",".join(categories)
    if bbox_tuple:
        min_lon, min_lat, max_lon, max_lat = bbox_tuple
        params["bbox"] = f"{min_lon},{max_lat},{max_lon},{min_lat}"  # EONET expects (minlon, maxlat, maxlon, minlat)
    url = "https://eonet.gsfc.nasa.gov/api/v3/events"
    r = requests.get(url, params=params, timeout=180)
    r.raise_for_status()
    data = r.json()

    rows = []
    for e in data.get("events", []):
        cats = [c.get("title") for c in (e.get("categories") or []) if isinstance(c, dict)]
        canon = _canonical_category(cats)
        for g in (e.get("geometry") or []):
            dt = pd.to_datetime(g.get("date"), errors="coerce", utc=True)
            pairs = _flatten_coords(g.get("coordinates"))
            if pairs:
                if len(pairs) == 1: lon, lat = pairs[0][0], pairs[0][1]
                else: lon, lat = _centroid_lonlat(pairs)
            else:
                lon, lat = None, None
            rows.append({
                "source_system": "EONET",
                "date": dt,
                "category": canon,
                "latitude": lat,
                "longitude": lon,
                "title": e.get("title"),
                "magnitude": None,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_usgs_earthquakes(start_d: date, end_d: date,
                           bbox_tuple: Optional[Tuple[float, float, float, float]],
                           min_mag: float) -> pd.DataFrame:
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_d.isoformat(),
        "endtime": end_d.isoformat(),
        "minmagnitude": min_mag,
        "orderby": "time-asc",
        "limit": 20000
    }
    if bbox_tuple:
        min_lon, min_lat, max_lon, max_lat = bbox_tuple
        params["minlatitude"] = min_lat; params["maxlatitude"] = max_lat
        params["minlongitude"] = min_lon; params["maxlongitude"] = max_lon
    r = requests.get(url, params=params, timeout=180)
    r.raise_for_status()
    j = r.json()

    rows = []
    for f in (j.get("features") or []):
        p = f.get("properties") or {}; g = f.get("geometry") or {}
        coords = (g.get("coordinates") or [None, None, None])
        rows.append({
            "source_system": "USGS",
            "date": pd.to_datetime(p.get("time"), unit="ms", utc=True),
            "category": "Earthquakes",
            "latitude": coords[1],
            "longitude": coords[0],
            "title": p.get("title"),
            "magnitude": p.get("mag"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df

# =========================
# Aggregation → Monthly target series
# =========================
def build_monthly_series(df: pd.DataFrame, target_choice: str) -> pd.Series:
    if df.empty: return pd.Series(dtype=float)
    df = df.dropna(subset=["date"]).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    grp = df.groupby("month").size() if target_choice == "All selected categories (combined)" \
          else df[df["category"] == target_choice].groupby("month").size()
    if grp.empty: return pd.Series(dtype=float)
    idx = pd.date_range(grp.index.min(), grp.index.max(), freq="MS")
    grp = grp.reindex(idx, fill_value=0); grp.name = "y"
    return grp.astype(float)

# =========================
# Feature builders
# =========================
def month_fourier(idx, m=12, K=2):
    # K harmonics: sin/cos(k*2πt/m)
    t = np.arange(len(idx), dtype=float)
    out = {}
    for k in range(1, K+1):
        out[f"sin_{k}"] = np.sin(2*np.pi*k*t/m)
        out[f"cos_{k}"] = np.cos(2*np.pi*k*t/m)
    return pd.DataFrame(out, index=idx)

def build_lag_feature_frame(series: pd.Series, n_lags=12):
    df = pd.DataFrame({"y": series})
    for L in range(1, n_lags+1):
        df[f"lag_{L}"] = df["y"].shift(L)
    df = df.join(month_fourier(df.index))
    return df.dropna()

# =========================
# Modeling utilities
# =========================
def seasonal_naive_forecast(series: pd.Series, steps: int, season: int = 12) -> pd.Series:
    if len(series) < season:
        idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
        return pd.Series([series.iloc[-1]] * steps, index=idx)
    fc = [series.iloc[-season + (i % season)] for i in range(steps)]
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    return pd.Series(fc, index=idx)

def sarimax_fit(series: pd.Series):
    for (p,d,q,P,D,Q) in [(1,1,1,1,1,1),(1,1,0,0,1,1)]:
        try:
            model = SARIMAX(series, order=(p,d,q), seasonal_order=(P,D,Q,12),
                            enforce_stationarity=False, enforce_invertibility=False)
            return model.fit(disp=False)
        except Exception:
            continue
    return None

def sarimax_fourier_one_step(train: pd.Series, K: int = 2):
    X = month_fourier(train.index, K=K)
    try:
        m = SARIMAX(train, exog=X, order=(1,0,0), seasonal_order=(0,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)
        res = m.fit(disp=False)
        nxt = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=1, freq="MS")
        Xf = month_fourier(nxt, K=K)
        return float(res.get_forecast(steps=1, exog=Xf).predicted_mean.iloc[0])
    except Exception:
        return float(train.iloc[-1])

def ets_one_step(train: pd.Series):
    try:
        fit = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit(optimized=True)
        return float(fit.forecast(1).iloc[0])
    except Exception:
        return float(train.iloc[-1])

def ridge_one_step(train: pd.Series):
    df = build_lag_feature_frame(train)
    X, y = df.drop(columns=["y"]), df["y"]
    if len(X) == 0: return float(train.iloc[-1])
    m = Ridge(alpha=1.0, random_state=42).fit(X, y)
    lastX = build_lag_feature_frame(train).drop(columns=["y"]).iloc[[-1]]
    return float(m.predict(lastX)[0])

def poisson_one_step(train: pd.Series):
    df = build_lag_feature_frame(train)
    if df.empty: return float(train.iloc[-1])
    X = sm.add_constant(df.drop(columns=["y"]))
    y = df["y"]
    try:
        model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        yhat = model.predict(sm.add_constant(build_lag_feature_frame(train).drop(columns=["y"]).iloc[[-1]]))[0]
        return float(max(yhat, 0.0))
    except Exception:
        return float(train.iloc[-1])

def negbin_one_step(train: pd.Series):
    df = build_lag_feature_frame(train)
    if df.empty: return float(train.iloc[-1])
    X = sm.add_constant(df.drop(columns=["y"]))
    y = df["y"]
    try:
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
        yhat = model.predict(sm.add_constant(build_lag_feature_frame(train).drop(columns=["y"]).iloc[[-1]]))[0]
        return float(max(yhat, 0.0))
    except Exception:
        return float(train.iloc[-1])

def rf_fit(series: pd.Series):
    df = build_lag_feature_frame(series)
    if df.empty:
        rf = RandomForestRegressor(random_state=42); feats = []
        return rf, df, feats
    X, y = df.drop(columns=["y"]), df["y"]
    rf = RandomForestRegressor(n_estimators=600, min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    return rf, df, list(X.columns)

def rf_forecast(rf, series: pd.Series, steps: int, feature_cols: List[str]) -> pd.Series:
    history = series.copy(); preds = []
    for _ in range(steps):
        df = build_lag_feature_frame(history)
        X = df[feature_cols].iloc[[-1]]
        yhat = float(rf.predict(X)[0])
        next_idx = history.index[-1] + pd.offsets.MonthBegin(1)
        history.loc[next_idx] = yhat; preds.append((next_idx, yhat))
    return pd.Series([p[1] for p in preds], index=[p[0] for p in preds])

def sarimax_fourier_fit_forecast(series: pd.Series, steps: int, K: int = 2) -> pd.Series:
    X = month_fourier(series.index, K=K)
    try:
        m = SARIMAX(series, exog=X, order=(1,0,0), seasonal_order=(0,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)
        res = m.fit(disp=False)
        future_idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
        Xf = month_fourier(future_idx, K=K)
        fc = res.get_forecast(steps=steps, exog=Xf).predicted_mean
        fc.index = future_idx
        return fc
    except Exception:
        return seasonal_naive_forecast(series, steps)

def negbin_iterative_forecast(series: pd.Series, steps: int) -> pd.Series:
    history = series.copy(); preds = []
    for _ in range(steps):
        yhat = negbin_one_step(history)
        nxt = history.index[-1] + pd.offsets.MonthBegin(1)
        history.loc[nxt] = yhat; preds.append((nxt, yhat))
    return pd.Series([p[1] for p in preds], index=[p[0] for p in preds])

def prophet_fit_forecast(series: pd.Series, steps: int) -> pd.Series:
    if not HAS_PROPHET:
        return seasonal_naive_forecast(series, steps)
    try:
        dfp = pd.DataFrame({"ds": pd.DatetimeIndex(series.index), "y": series.values})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(dfp)
        future = m.make_future_dataframe(periods=steps, freq="MS", include_history=False)
        fcst = m.predict(future)[["ds", "yhat"]]
        fcst.index = pd.DatetimeIndex(fcst["ds"]).to_period("M").to_timestamp()
        return pd.Series(np.maximum(fcst["yhat"].values, 0.0), index=fcst.index)
    except Exception:
        return seasonal_naive_forecast(series, steps)

# =========================
# Metrics helper
# =========================
def r2_score_series(y: pd.Series, yhat: pd.Series) -> float:
    y = y.astype(float); yhat = yhat.astype(float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)

# =========================
# Backtesting
# =========================
@dataclass
class ModelResult:
    name: str
    rmse: float
    mae: float
    r2: float
    preds: pd.Series

def backtest_models(series: pd.Series, backtest_months: int,
                    flags: Dict[str, bool]) -> List[ModelResult]:
    enabled = [name for name, on in flags.items() if on]
    if len(series) < 24:  # simple split
        cut = max(12, int(len(series) * 0.8))
        train, test = series.iloc[:cut], series.iloc[cut:]
        idx = test.index
        results = []

        for name in enabled:
            if name == "Seasonal-Naive(12)":
                fc = seasonal_naive_forecast(train, steps=len(idx)).reindex(idx)
            elif name == "SARIMAX":
                sar = sarimax_fit(train)
                fc = sar.get_forecast(steps=len(idx)).predicted_mean if sar is not None else seasonal_naive_forecast(train, len(idx)).reindex(idx)
                fc.index = idx
            elif name == "SARIMAX+Fourier":
                fc = pd.Series([sarimax_fourier_one_step(train.iloc[:i]) for i in range(len(train), len(train)+len(idx))], index=idx)
            elif name == "ETS":
                fc = pd.Series([ets_one_step(train.iloc[:i]) for i in range(len(train), len(train)+len(idx))], index=idx)
            elif name == "RandomForest(lags)":
                rf, df_rf, feats = rf_fit(train)
                fc = rf_forecast(rf, train, steps=len(idx), feature_cols=feats); fc.index = idx
            elif name == "Ridge(lags)":
                fc = pd.Series([ridge_one_step(train.iloc[:i]) for i in range(len(train), len(train)+len(idx))], index=idx)
            elif name == "Poisson(lags)":
                fc = pd.Series([poisson_one_step(train.iloc[:i]) for i in range(len(train), len(train)+len(idx))], index=idx)
            elif name == "NegBin(lags)":
                fc = pd.Series([negbin_one_step(train.iloc[:i]) for i in range(len(train), len(train)+len(idx))], index=idx)
            elif name == "Prophet":
                fc = prophet_fit_forecast(train, steps=len(idx)).reindex(idx)
            else:
                continue

            rmse = float(np.sqrt(mean_squared_error(test, fc)))
            mae = float(mean_absolute_error(test, fc))
            r2 = r2_score_series(test, fc)
            results.append(ModelResult(name, rmse, mae, r2, fc))
        return results

    # Rolling 1-step ahead over last backtest_months
    test_start = series.index[-backtest_months]
    truth = series[test_start:]
    results_map = {name: [] for name in enabled}

    for t in truth.index:
        train = series[:t - pd.offsets.MonthBegin(1)]
        for name in enabled:
            try:
                if name == "Seasonal-Naive(12)":
                    yhat = seasonal_naive_forecast(train, steps=1).iloc[0]
                elif name == "SARIMAX":
                    sar = sarimax_fit(train)
                    yhat = sar.get_forecast(steps=1).predicted_mean.iloc[0] if sar is not None else train.iloc[-1]
                elif name == "SARIMAX+Fourier":
                    yhat = sarimax_fourier_one_step(train, K=2)
                elif name == "ETS":
                    yhat = ets_one_step(train)
                elif name == "RandomForest(lags)":
                    rf, df_rf, feats = rf_fit(train)
                    yhat = rf_forecast(rf, train, steps=1, feature_cols=feats).iloc[0]
                elif name == "Ridge(lags)":
                    yhat = ridge_one_step(train)
                elif name == "Poisson(lags)":
                    yhat = poisson_one_step(train)
                elif name == "NegBin(lags)":
                    yhat = negbin_one_step(train)
                elif name == "Prophet":
                    yhat = float(prophet_fit_forecast(train, steps=1).iloc[0])
                else:
                    continue
                results_map[name].append(yhat)
            except Exception:
                results_map[name].append(float(train.iloc[-1]))

    results: List[ModelResult] = []
    for name, preds in results_map.items():
        yhat = pd.Series(preds, index=truth.index)
        rmse = float(np.sqrt(mean_squared_error(truth, yhat)))
        mae = float(mean_absolute_error(truth, yhat))
        r2 = r2_score_series(truth, yhat)
        results.append(ModelResult(name, rmse, mae, r2, yhat))
    return results

def pick_best_model(results: List[ModelResult], priority: str = "rmse") -> ModelResult:
    key = "rmse" if priority.lower() == "rmse" else "r2"
    reverse = False if key == "rmse" else True  # R² higher is better
    return sorted(results, key=lambda r: getattr(r, key), reverse=reverse)[0]

def fit_and_forecast_best(series: pd.Series, steps: int, best_name: str):
    """Return (forecast_series, conf_df|None) for the chosen model."""
    if best_name == "Seasonal-Naive(12)":
        return seasonal_naive_forecast(series, steps), None

    if best_name == "SARIMAX":
        sar = sarimax_fit(series)
        if sar is None:
            return seasonal_naive_forecast(series, steps), None
        try:
            res = sar.get_forecast(steps=steps)
            fc = res.predicted_mean
            conf = res.conf_int(alpha=0.2)
            if conf.shape[1] >= 2:
                conf.columns = ["lower", "upper"] + list(conf.columns[2:])
            conf.index = fc.index
            conf = conf[["lower", "upper"]] if {"lower","upper"}.issubset(conf.columns) else None
            return fc, conf
        except Exception:
            return seasonal_naive_forecast(series, steps), None

    if best_name == "SARIMAX+Fourier":
        fc = sarimax_fourier_fit_forecast(series, steps, K=2)
        return fc, None

    if best_name == "ETS":
        try:
            fit = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=12).fit(optimized=True)
            fc = fit.forecast(steps)
            return fc, None
        except Exception:
            return seasonal_naive_forecast(series, steps), None

    if best_name == "RandomForest(lags)":
        rf, _, feats = rf_fit(series)
        fc = rf_forecast(rf, series, steps, feats)
        return fc, None

    if best_name == "Ridge(lags)":
        history = series.copy(); preds = []
        for _ in range(steps):
            yhat = ridge_one_step(history)
            nxt = history.index[-1] + pd.offsets.MonthBegin(1)
            history.loc[nxt] = yhat; preds.append((nxt, yhat))
        return pd.Series([p[1] for p in preds], index=[p[0] for p in preds]), None

    if best_name == "Poisson(lags)":
        history = series.copy(); preds = []
        for _ in range(steps):
            yhat = poisson_one_step(history)
            nxt = history.index[-1] + pd.offsets.MonthBegin(1)
            history.loc[nxt] = yhat; preds.append((nxt, yhat))
        return pd.Series([p[1] for p in preds], index=[p[0] for p in preds]), None

    if best_name == "NegBin(lags)":
        return negbin_iterative_forecast(series, steps), None

    if best_name == "Prophet":
        return prophet_fit_forecast(series, steps), None

    return seasonal_naive_forecast(series, steps), None

# =========================
# Fetch & prepare data
# =========================
selected_slugs = []
if use_eq: selected_slugs.append("earthquakes")
if use_fl: selected_slugs.append("floods")
if use_ss: selected_slugs.append("severeStorms")

with st.spinner("Fetching EONET…"):
    eonet_df = fetch_eonet(selected_slugs, start_date, end_date, bbox)

if include_usgs and use_eq:
    with st.spinner("Fetching USGS earthquakes…"):
        usgs_df = fetch_usgs_earthquakes(start_date, end_date, bbox, usgs_minmag)
else:
    usgs_df = pd.DataFrame(columns=eonet_df.columns)

combined = pd.concat([eonet_df, usgs_df], ignore_index=True, sort=False)
combined = _apply_bbox_filter(combined, bbox)

if combined.empty:
    st.warning("No events returned. Try broadening dates, enabling more categories, or switching region to Global.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Total rows", f"{len(combined):,}")
with c2: st.metric("Date span", f"{combined['date'].min().date()} → {combined['date'].max().date()}")
with c3: st.metric("Region", region_name)
with c4: st.metric("Categories", ", ".join(sorted(combined["category"].dropna().unique())))

# =========================
# Build target series (monthly counts) + make it easier to predict
# =========================
y = build_monthly_series(combined, target_choice)

if y.empty or y.sum() == 0:
    st.warning("No monthly counts for the selected target. Try another category or region.")
    st.stop()

orig_y = y.copy()
if clip_outliers and len(y) > 24:
    hi = y.quantile(0.995)
    y = y.clip(upper=hi)
if target_smoothing and len(y) > 6:
    y = y.rolling(3, min_periods=1).mean()
y = y.dropna()

st.subheader("📈 Monthly event counts (historical)")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=y.index, y=y.values, mode="lines", name="Monthly count"))
fig_hist.update_layout(
    height=260,
    title="Monthly event counts (historical)",
    xaxis_title="Month",
    yaxis_title="Event count",
    hovermode="x unified",
    margin=dict(l=40, r=20, t=50, b=40),
    **PLOTLY_DARK_LAYOUT
)
st.plotly_chart(fig_hist, use_container_width=True, theme=None)

if target_smoothing or clip_outliers:
    st.caption("Applied preprocessing: " +
               ("3-month rolling mean; " if target_smoothing else "") +
               ("99.5th-pct spike clipping" if clip_outliers else ""))

# =========================
# Train, backtest, select best, forecast
# =========================
if run_btn:
    flags = {
        "Seasonal-Naive(12)": use_sn,
        "SARIMAX": use_sarimax,
        "SARIMAX+Fourier": use_sarimax_fourier,
        "ETS": use_ets,
        "RandomForest(lags)": use_rf,
        "Ridge(lags)": use_ridge,
        "Poisson(lags)": use_poisson,
        "NegBin(lags)": use_negbin,
        "Prophet": use_prophet and HAS_PROPHET,
    }
    flags = {k: v for k, v in flags.items() if v}

    with st.spinner("Training models & running backtests…"):
        results = backtest_models(y, backtest_months=backtest_months, flags=flags)

    if not results:
        st.error("No model could be evaluated. Try enabling at least one model or extend the history window.")
        st.stop()

    # Metrics table (sort by chosen selection metric)
    mt = pd.DataFrame([{
        "Model": r.name,
        "RMSE": round(r.rmse, 3),
        "MAE": round(r.mae, 3),
        "R²": (None if pd.isna(r.r2) else round(r.r2, 3)),
    } for r in results])

    sort_col = "RMSE" if select_metric == "RMSE" else "R²"
    ascending = True if sort_col == "RMSE" else False
    st.subheader("🧪 Backtest metrics")
    st.dataframe(mt.sort_values(sort_col, ascending=ascending), height=260, use_container_width=True)
    st.caption("Lower is better for RMSE/MAE; higher is better for R².")

    best = pick_best_model(results, priority=("rmse" if select_metric == "RMSE" else "r2"))
    st.success(f"🏆 Best model by {select_metric}: **{best.name}**")

    horizon_months = int(horizon_years * 12)
    fc, conf = fit_and_forecast_best(y, steps=horizon_months, best_name=best.name)

    # Plot (union of history + forecast indexes)
    hist = y.copy()
    df_plot = pd.concat([hist.rename("History"), fc.rename("Forecast")], axis=1).sort_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["History"], mode="lines", name="History"))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Forecast"], mode="lines", name="Forecast"))

    if conf is not None and not conf.empty:
        conf = conf.copy().reindex(fc.index)
        fig.add_trace(go.Scatter(
            x=conf.index, y=conf["upper"], mode="lines", name="Upper (≈80%)",
            line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=conf.index, y=conf["lower"], mode="lines", name="Lower (≈80%)",
            fill="tonexty", line=dict(width=0), fillcolor="rgba(96,165,250,0.25)", showlegend=True
        ))

    fig.update_layout(
        height=420,
        title=f"Forecast: {target_choice} — {region_name} | Horizon: {horizon_years} year(s)",
        xaxis_title="Month", yaxis_title="Event count",
        hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40),
        **PLOTLY_DARK_LAYOUT
    )
    st.subheader("🔮 Forecast")
    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.subheader("📊 Forecast info")
    st.write(f"**Best model:** {best.name}")
    st.write(f"**Horizon:** {horizon_years} year(s) ({horizon_months} months)")
    st.write(f"**History window (after preprocessing):** {y.index.min().date()} → {y.index.max().date()}")

    # Downloads
    hist_csv = y.reset_index().rename(columns={"index": "month", "y": "count"}).to_csv(index=False).encode("utf-8")
    st.download_button("Download history (CSV)", data=hist_csv, file_name="history_monthly_counts.csv", mime="text/csv")

    fc_out = pd.DataFrame({"month": fc.index, "forecast_count": fc.values})
    if conf is not None and not conf.empty:
        fc_out = fc_out.join(conf[["lower", "upper"]], how="left")
    fc_csv = fc_out.to_csv(index=False).encode("utf-8")
    st.download_button("Download forecast (CSV)", data=fc_csv, file_name="forecast_counts.csv", mime="text/csv")

else:
    st.info("Choose models and click **Run training & forecasting** to evaluate and forecast.")

# =========================
# Tips
# =========================
with st.expander("💡 Why scores can look low & how to boost them"):
    st.markdown("""
- **Data regime changes**: EONET coverage is curated and evolves; very early years can be sparse. Try limiting to **2010+** or a **single hazard**.
- **Heterogeneous mix**: Earthquakes + Floods + Typhoons behave differently. Modeling **one category at a time** often raises R².
- **Noise**: Monthly counts are spiky. The **3-month rolling mean** and **spike clipping** options make the signal more forecastable.
- **Over-dispersion**: Counts often have variance » mean. **Negative Binomial** handles this better than Poisson.
- **Seasonality**: **SARIMAX+Fourier** captures multiple seasonal harmonics more flexibly than plain SARIMAX.
""")
