# streamlit_frontend.py
# pip install streamlit pandas openpyxl xlsxwriter st-aggrid altair

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os, io, itertools, json, datetime, sys, subprocess
from pathlib import Path
from typing import Optional, Tuple, List

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

# ========= CONFIG =========
DEFAULT_CSV = r"C:\Users\aden.chong\OneDrive - United Engineers Limited\Aden\Week5\Extracted Hotel Data\hotel_prices_provider_rows.csv"
CLEANING_SCRIPT_NAME = "CleaningHotelData.py"  # assumes same folder as this file

st.set_page_config(page_title="Hotel Prices (Pivot Viewer)", page_icon="🏨", layout="wide")
st.title("🏨 Hotel Prices — OTAs × Hotels")

# ========= GLOBAL DARK THEME + STANDARDIZED TYPOGRAPHY =========
st.markdown("""
<style>
:root{
  /* Typography */
  --font-sans: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol";
  --fs-1: 28px; --fs-2: 22px; --fs-3: 18px; --fs-base: 14px;
  --lh-tight: 1.2; --lh-base: 1.55; --radius: 10px;

  /* Palette */
  --bg:#0b0f19;           /* app background */
  --surface:#0f172a;      /* cards/containers */
  --surface-2:#111827;    /* hovers/active */
  --border:#1f2937;
  --text:#e5e7eb;
  --muted:#9ca3af;

  /* Signals */
  --brand:#22c55e; --info:#3b82f6; --warn:#f59e0b; --danger:#ef4444; --ok:#10b981;

  /* Δ heat */
  --grid-best:#103d27; --grid-good:#0b5130; --grid-mid:#4a3a10; --grid-bad:#5a1e1e;

  --focus:#2563eb;
}

/* ==== Global app background & typography (MAIN AREA TOO) ==== */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] > .main .block-container{
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-sans) !important;
  font-size: var(--fs-base) !important;
  line-height: var(--lh-base) !important;
  -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
}

/* Sidebar */
[data-testid="stSidebar"]{
  background-color: var(--surface) !important;
  color: var(--text) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] *{ color: var(--text) !important; }

/* Headings */
h1{ font-size:var(--fs-1) !important; line-height:var(--lh-tight) !important; font-weight:700 !important; }
h2{ font-size:var(--fs-2) !important; line-height:var(--lh-tight) !important; font-weight:700 !important; }
h3, .stMarkdown h3{ font-size:var(--fs-3) !important; line-height:var(--lh-tight) !important; font-weight:600 !important; }

/* Inputs */
input, textarea, select, .stTextInput > div > div > input,
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea{
  background-color:#0b1220 !important; color:var(--text) !important; border:1px solid var(--border) !important;
  border-radius:8px !important;
}
input:focus, textarea:focus, select:focus{ outline:2px solid var(--focus) !important; }

/* Buttons */
button[kind="primary"], .stDownloadButton > button, .stButton > button{
  background-color:#1f2937 !important; color:var(--text) !important; border:1px solid #374151 !important; border-radius:8px !important;
  font-weight:600 !important;
}
button[kind="primary"]:hover, .stDownloadButton > button:hover, .stButton > button:hover{
  background-color:var(--surface-2) !important; border-color:#4b5563 !important;
}

/* Alerts */
[data-testid="stAlert"]{ filter:brightness(0.98); border-radius:var(--radius) !important; }

/* Scrollbars */
*::-webkit-scrollbar{ height:10px; width:10px; }
*::-webkit-scrollbar-thumb{ background:#374151; border-radius:8px; }
*::-webkit-scrollbar-track{ background:#0d1220; }

/* DataFrame (st.dataframe) wrapper so it doesn't sit on white */
[data-testid="stDataFrame"]{
  background:var(--surface) !important;
  border:1px solid var(--border) !important;
  border-radius:var(--radius) !important;
}

/* ===== AG Grid — dark everywhere ===== */
[class^="ag-theme-"], [class*=" ag-theme-"]{
  --ag-foreground-color: var(--text);
  --ag-background-color: var(--surface);
  --ag-header-background-color: var(--surface);
  --ag-header-foreground-color: var(--text);
  --ag-row-hover-color: var(--surface-2);
  --ag-selected-row-background-color: var(--surface-2);
  --ag-border-color: var(--border);
  font-family: var(--font-sans) !important;
  font-size: var(--fs-base) !important;
}
.ag-root-wrapper{ border-radius:var(--radius) !important; }
.ag-header-viewport{ overflow:visible !important; }
.ag-theme-alpine-dark .ag-header-cell-label,
.ag-theme-alpine .ag-header-cell-label,
.ag-theme-balham .ag-header-cell-label,
.ag-theme-balham-dark .ag-header-cell-label{ font-weight:600 !important; letter-spacing:.2px; }
.ag-cell, .ag-header-cell, .ag-header-group-cell{ color:var(--text) !important; line-height:1.4 !important; }
.ag-theme-alpine-dark .null-black, .ag-theme-balham-dark .null-black, .ag-theme-alpine .null-black, .ag-theme-balham .null-black{ color:#000 !important; }

/* Δ% heat colors */
.ag-delta-best{ background: var(--grid-best) !important; }
.ag-delta-good{ background: var(--grid-good) !important; }
.ag-delta-mid { background: var(--grid-mid)  !important; }
.ag-delta-bad { background: var(--grid-bad)  !important; }

/* ===== DARK selects in the sidebar (Missing values / Currency) ===== */
:root{
  --select-bg:#0b1220; --select-border:#1f2937; --select-text:#e5e7eb; --select-muted:#9ca3af;
  --menu-bg:#0f172a; --menu-border:#1f2937; --menu-hover:#111827; --menu-selected:#1f2937;
}
[data-testid="stSidebar"] div[data-baseweb="select"] > div{
  background:var(--select-bg) !important; border:1px solid var(--select-border) !important; border-radius:8px !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"],
[data-testid="stSidebar"] div[data-baseweb="select"] *,
[data-testid="stSidebar"] div[data-baseweb="select"] [role="combobox"],
[data-testid="stSidebar"] div[data-baseweb="select"] [role="combobox"] *{
  color:var(--select-text) !important; -webkit-text-fill-color:var(--select-text) !important; opacity:1 !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] input::placeholder{ color:var(--select-muted) !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] svg{ color:var(--select-text) !important; fill:var(--select-text) !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within{
  box-shadow:0 0 0 2px var(--focus) inset !important; border-color:var(--focus) !important;
}
/* Dropdown menu (portal) */
[data-baseweb="menu"]{
  background:var(--menu-bg) !important; color:var(--select-text) !important; border:1px solid var(--menu-border) !important;
}
[data-baseweb="menu"] [role="option"], [data-baseweb="menu"] [role="option"] *{ color:var(--select-text) !important; }
[data-baseweb="menu"] [role="option"]:hover{ background:var(--menu-hover) !important; }
[data-baseweb="menu"] [aria-selected="true"]{ background:var(--menu-selected) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Radio group label (e.g., "Choose a simple view") */
div[data-testid="stWidgetLabel"] p {
  color: var(--text) !important;
}

/* Radio option text (horizontal or vertical) */
div[role="radiogroup"] label p,
div[role="radiogroup"] label span {
  color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Radio group label + options above charts (keep) */
div[data-testid="stWidgetLabel"] p,
div[role="radiogroup"] label p,
div[role="radiogroup"] label span {
  color: var(--text) !important;
}

/* KPI metrics: label + value in white */
[data-testid="stMetric"] { color: var(--text) !important; }
[data-testid="stMetricLabel"] div { color: var(--text) !important; }
[data-testid="stMetricValue"] div { color: var(--text) !important; }

/* Section headings (e.g., 'Cheapest by hotel (summary)', 'Analysis') */
h2, h3, .stMarkdown h3 {
  color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)



# ========= CLEANER LAUNCH HELPER (runs the external script) =========
def run_cleaner(script_name: str, output_csv: Optional[str] = None, extra_args: Optional[List[str]] = None) -> None:
    """
    Execute the cleaning script with the current Python (same venv as Streamlit).
    If your cleaner accepts --output, we pass it.
    """
    script_path = Path(__file__).with_name(script_name)
    if not script_path.exists():
        st.warning(f"Cleaning script not found: {script_path}")
        return
    cmd = [sys.executable, str(script_path)]
    if output_csv:
        # Remove these 2 args if your cleaner doesn't accept --output
        cmd += ["--output", output_csv]
    if extra_args:
        cmd += list(extra_args)
    subprocess.run(cmd, check=True)

# ========= HELPERS =========
def try_read_csv(csv_path: str) -> Tuple[Optional[pd.DataFrame], str]:
    encodings = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
    seps = (None, ",", ";", "\t")
    last_err_1 = ""
    for enc, sep in itertools.product(encodings, seps):
        try:
            df = pd.read_csv(csv_path, encoding=enc, sep=sep, engine="python", index_col=0)
            return df, f"Loaded with encoding={enc}, sep={repr(sep)}, index_col=0"
        except Exception as e:
            last_err_1 = str(e)
    last_err_2 = ""
    for enc, sep in itertools.product(encodings, seps):
        try:
            df = pd.read_csv(csv_path, encoding=enc, sep=sep, engine="python")
            if "Provider" in df.columns:
                df = df.set_index("Provider")
            return df, f"Loaded with encoding={enc}, sep={repr(sep)}, post-set index='Provider' (if present)"
        except Exception as e:
            last_err_2 = str(e)
    return None, f"Failed to read file. Last errors: index_col attempt: {last_err_1} | no-index attempt: {last_err_2}"

@st.cache_data(show_spinner=False)
def load_pivot(csv_path: str) -> pd.DataFrame:
    df, dbg = try_read_csv(csv_path)
    if df is None:
        raise RuntimeError(dbg)
    if df.index.name is None or str(df.index.name).strip() == "":
        df.index.name = "Provider"
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Pivot") -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=True)
    return bio.getvalue()

def make_rates_pp(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Rates ++ = price × 1.19 (price + 10% + 9%)."""
    return df * 1.19

def interleave_prices_and_rates(df_prices: pd.DataFrame, df_rates: pd.DataFrame) -> pd.DataFrame:
    cols = []
    out = pd.DataFrame(index=df_prices.index)
    for col in df_prices.columns:
        out[col] = df_prices[col]
        out[f"{col} rates ++"] = df_rates[col]
        cols += [col, f"{col} rates ++"]
    out = out[cols]
    out.index.name = df_prices.index.name
    return out

def aggrid_from_df(
    df: pd.DataFrame,
    decimals: int = 2,
    pinned_provider: bool = True,
    quick_text: str = "",
    height: int = 540,
    empty_label: Optional[str] = None,
    currency_prefix: Optional[str] = None,
    highlight_row_min: bool = True
):
    """
    Render an AG Grid with standardized fonts, horizontal scroll, pinned Provider, formatting, quick filter.
    """
    df_show = df.reset_index()
    gb = GridOptionsBuilder.from_dataframe(df_show)
    gb.configure_default_column(resizable=True, sortable=True, filter=True, wrapText=False, autoHeight=False)
    if "Provider" in df_show.columns and pinned_provider:
        gb.configure_column("Provider", pinned="left")

    # value formatting
    fill_text_js = json.dumps(empty_label) if empty_label is not None else "null"
    cur_js = json.dumps(currency_prefix) if currency_prefix is not None else "null"
    value_formatter = JsCode(f"""
        function(params) {{
            const FILL = {fill_text_js};
            const CUR  = {cur_js};
            if (params.value === null || params.value === undefined || params.value === "") {{
                return (FILL !== null) ? FILL : "";
            }}
            let v = Number(params.value);
            if (isNaN(v)) return params.value;
            let s = v.toFixed({decimals});
            return (CUR !== null) ? (CUR + s) : s;
        }}
    """)

    # rule: if value is null/blank, force black text (so "No Price Available" shows in black)
    null_rule_js = JsCode("""
        function(params) {
            return (params.value === null || params.value === undefined || params.value === "");
        }
    """)

    for col in df_show.columns:
        if col != "Provider":
            gb.configure_column(
                col,
                type=["numericColumn"],
                valueFormatter=value_formatter,
                cellClassRules={"null-black": null_rule_js}
            )

    # Dark-friendly Δ heat
    delta_cell_style = JsCode("""
        function(params) {
            if (params.value === null || params.value === undefined || params.value === "") { return {}; }
            let v = Number(params.value);
            if (isNaN(v)) return {};
            if (v <= 0)     return {'backgroundColor': '#103d27'};  // best
            if (v <= 5)     return {'backgroundColor': '#0b5130'};
            if (v <= 15)    return {'backgroundColor': '#4a3a10'};
            return {'backgroundColor': '#5a1e1e'};
        }
    """)

    # Row-min highlight for price-like tables
    highlight_min_js = JsCode("""
    function(params) {
        if (params.value === null || params.value === undefined || params.value === "" || isNaN(Number(params.value))) { return {}; }
        const row = params.node.data;
        let minVal = Infinity;
        for (const k in row) {
            if (k === "Provider") continue;
            const v = Number(row[k]);
            if (!isNaN(v)) { minVal = Math.min(minVal, v); }
        }
        if (Number(params.value) === minVal) {
            return { 'boxShadow': 'inset 0 0 0 2px #10b981', 'borderRadius': '6px' };
        }
        return {};
    }
    """)

    # Detect Δ/% view vs price-like view (pure Python)
    is_delta_view = any(("Δ" in str(c)) or ("delta" in str(c).lower()) for c in df_show.columns)
    price_like = (currency_prefix is not None)

    gb.configure_grid_options(
        rowHeight=34, enableRangeSelection=True, suppressMovableColumns=False,
        ensureDomOrder=True, animateRows=False, domLayout="normal", quickFilterText=quick_text,
    )
    grid_options = gb.build()

    # attach cell styles (delta heat OR row-min ring)
    for col in df_show.columns:
        if col == "Provider":
            continue
        for i, cdef in enumerate(list(grid_options.get("columnDefs", []))):
            if cdef.get("field") == col:
                if is_delta_view:
                    grid_options["columnDefs"][i] = {**cdef, **{"cellStyle": delta_cell_style}}
                elif price_like and highlight_row_min:
                    grid_options["columnDefs"][i] = {**cdef, **{"cellStyle": highlight_min_js}}

    AgGrid(
        df_show,
        gridOptions=grid_options,
        theme="balham-dark",
        height=height,
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.NO_UPDATE,
        fit_columns_on_grid_load=False,
    )

# ========= SIDEBAR =========
with st.sidebar:
    st.subheader("Settings")
    csv_path = st.text_input("Finalised CSV path", DEFAULT_CSV)

    # Default set to "No Price Available"
    fill_mode = st.selectbox("Missing values", ["blank", "0", "No Price Available"], index=2)
    decimals  = st.number_input("Decimal places", min_value=0, max_value=6, value=2, step=1)
    view_mode = st.radio(
        "View",
        ["Prices", "Rates ++ (price × 1.19)", "Prices & Rates ++ (side-by-side)", "Δ vs Row Min (%)"],
        index=0
    )

    pin_provider = st.checkbox("Freeze 'Provider' column", value=True)
    quick_filter = st.text_input("Quick search (filters all columns)", value="")
    grid_height = st.slider("Grid height (px)", min_value=360, max_value=900, value=540, step=30)

    # ===== A) Currency & Rates++ toggle =====
    st.markdown("---")
    st.markdown("**Currency & Tax**")
    currencies = {
        "SGD $": 1.00,
        "USD $": 0.73,   # update if needed
        "EUR €": 0.67,
        "AUD $": 1.02
    }
    sel_ccy = st.selectbox("Currency", list(currencies.keys()), index=0)
    fx_default = currencies[sel_ccy]
    fx = st.number_input(
        "FX multiplier (Selected = FX × SGD)",
        min_value=0.0001, value=float(fx_default), step=0.0001, format="%.4f",
        help="All prices are converted: Price_selected = Price_SGD × FX."
    )
    show_rates_pp_toggle = st.checkbox("Apply Rates ++ to displayed prices (10% svc + 9% GST)", value=False)
    ccy_symbol = "€" if "€" in sel_ccy else "$"

    st.markdown("---")
    debug_show = st.checkbox("Show debug info (raw preview, shapes)", value=False)
    refresh = st.button("🔄 Reload")

# ===== Disclaimer config (no sidebar controls) =====
DEFAULT_DISCLAIMER = (
    "Some cells show “No Price Available”. This may occur when a provider is offering "
    "discounted, member-only, or time-limited rates that aren’t captured by the automation. "
    "Please check the provider’s website for the most accurate and up-to-date price."
)
show_disclaimer = (fill_mode == "No Price Available")
disclaimer_text = DEFAULT_DISCLAIMER

# ========= RUN CLEANER BEFORE LOADING CSV =========
if "cleaned_once" not in st.session_state:
    st.session_state.cleaned_once = False

should_run_cleaner = refresh or (not st.session_state.cleaned_once)
if should_run_cleaner:
    with st.spinner("Preparing latest data…"):
        try:
            run_cleaner(CLEANING_SCRIPT_NAME, output_csv=csv_path)
            load_pivot.clear()
            st.session_state.cleaned_once = True
        except subprocess.CalledProcessError as e:
            st.error(f"Cleaning script failed (exit code {e.returncode}). Check CleaningHotelData.py.")
        except Exception as e:
            st.error(f"Could not execute cleaning script: {e}")

# ========= LOAD DATA / CHECKS =========
if not os.path.exists(csv_path):
    st.error(f"File not found:\n{csv_path}")
    st.stop()

size_bytes = os.path.getsize(csv_path)
st.caption(f"Reading: `{csv_path}`  •  Size: {size_bytes:,} bytes")

try:
    pivot = load_pivot(csv_path)
except Exception as e:
    st.error(f"Could not load CSV: {e}")
    with st.expander("Raw file preview (first 30 lines)"):
        try:
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                raw = "".join([next(f) for _ in range(30)])
            st.code(raw, language="text")
        except Exception as e2:
            st.write(f"(Could not preview raw file: {e2})")
    st.stop()

# Filters
with st.sidebar:
    providers = list(pivot.index.astype(str))
    hotels = list(pivot.columns.astype(str))
    sel_providers = st.multiselect("Providers", options=providers, default=providers)
    sel_hotels = st.multiselect("Hotels", options=hotels, default=hotels)

# Apply filters
base_df = pivot.copy()
if sel_providers:
    base_df = base_df.loc[base_df.index.intersection(sel_providers)]
else:
    base_df = base_df.iloc[0:0]
if sel_hotels:
    keep_cols = [c for c in sel_hotels if c in base_df.columns]
    base_df = base_df[keep_cols]
else:
    base_df = base_df.iloc[:, 0:0]

# ===== Currency & tax application to the working frame =====
_work_df = base_df.copy()
if show_rates_pp_toggle:
    _work_df = make_rates_pp(_work_df)
if fx and fx != 1.0:
    _work_df = _work_df * fx

# ===== Build view =====
if view_mode == "Prices":
    view_df = _work_df.copy()

elif view_mode == "Rates ++ (price × 1.19)":
    rates_df = make_rates_pp(base_df)
    if fx and fx != 1.0:
        rates_df = rates_df * fx
    view_df = rates_df.rename(columns=lambda c: f"{c} rates ++")

elif view_mode == "Prices & Rates ++ (side-by-side)":
    rates_df = make_rates_pp(base_df)
    if fx and fx != 1.0:
        rates_df = rates_df * fx
    view_df = interleave_prices_and_rates(_work_df, rates_df)

else:  # Δ vs Row Min (%)
    if _work_df.empty:
        view_df = _work_df.copy()
    else:
        base = _work_df.min(axis=1).replace(0, np.nan)
        view_df = ((_work_df.T / base).T - 1.0) * 100.0
        view_df.columns = [f"{c} (Δ% vs row min)" for c in view_df.columns]

# Fill & format
if fill_mode == "0":
    view_df = view_df.fillna(0)
if isinstance(decimals, int) and not view_df.empty and view_mode != "Δ vs Row Min (%)":
    view_df = view_df.round(decimals)

# ===== WARNINGS =====
if pivot.empty:
    st.warning("Loaded table has 0 rows/columns. Check the CSV structure.")
elif view_df.empty:
    st.info("No data to display (check your Provider/Hotel filters).")

# ===== OPTIONAL DISCLAIMER (auto, no sidebar controls) =====
if (
    fill_mode == "No Price Available"
    and show_disclaimer
    and not view_df.empty
    and view_df.isna().sum().sum() > 0
):
    st.warning(disclaimer_text)

# ===== TABLE (AG GRID) =====
st.subheader("Table")
is_price_like_view = view_mode in [
    "Prices", "Rates ++ (price × 1.19)", "Prices & Rates ++ (side-by-side)"
]
aggrid_from_df(
    view_df,
    decimals=decimals,
    pinned_provider=True,
    quick_text=quick_filter,
    height=grid_height,
    empty_label=("No Price Available" if fill_mode == "No Price Available" else None),
    currency_prefix=(ccy_symbol if is_price_like_view else None),
    highlight_row_min=True
)

# ===== Cheapest by hotel (summary) =====
st.markdown("### Cheapest by hotel (summary)")
if not view_df.empty and is_price_like_view:
    summary_rows = []
    for hotel in [c for c in view_df.columns if c != "Provider"]:
        col_vals = pd.to_numeric(view_df[hotel], errors="coerce")
        min_price = col_vals.min()
        if pd.isna(min_price):
            summary_rows.append({"Hotel": hotel, "Cheapest Provider(s)": "—", "Price": np.nan})
        else:
            winners = view_df.index[col_vals == min_price].tolist()
            summary_rows.append({
                "Hotel": hotel,
                "Cheapest Provider(s)": ", ".join(map(str, winners))[:200],
                "Price": min_price
            })
    summary_df = pd.DataFrame(summary_rows).sort_values("Hotel").reset_index(drop=True)
    if isinstance(decimals, int):
        summary_df["Price"] = summary_df["Price"].round(decimals)
    summary_df["Price"] = summary_df["Price"].apply(lambda v: (f"{ccy_symbol}{v:.{decimals}f}") if pd.notna(v) else "")
    st.dataframe(summary_df, use_container_width=True)

# ===== Analysis (simple, non-analyst friendly) =====
# ===== Analysis (simple, non-analyst friendly) =====
st.markdown("### Analysis")

if view_df.empty:
    st.caption("Add filters or load a CSV to enable charts.")
else:
    try:
        import altair as alt

        # Axes/legend black on white chart background
        def dark_cfg(ch):
            return (
                ch.configure_axis(
                    labelFontSize=12,
                    titleFontSize=12,
                    labelColor='#111827',   # near-black
                    titleColor='#111827',
                    domainColor='#111827',
                    tickColor='#111827',
                    gridColor='#e5e7eb'
                )
                .configure_legend(
                    labelFontSize=12,
                    titleFontSize=12,
                    labelColor='#111827',
                    titleColor='#111827',
                )
                .configure_view(stroke=None, fill='#ffffff')
            )

        # Safe slider: shows a fixed caption if range collapses (e.g., max == min)
        def slider_or_fixed(label, min_v, max_v, default_v, step=1, key=None):
            if max_v <= min_v:
                v = max_v
                st.caption(f"{label}: {v} (auto)")
                return v
            default_v = min(max(default_v, min_v), max_v)
            return st.slider(label, min_v, max_v, default_v, step=step, key=key)

        # ============== Shared transforms ==============
        df_num = view_df.apply(pd.to_numeric, errors="coerce")
        long = (
            df_num.reset_index()
            .rename(columns={"index": "Provider"})
            .melt(id_vars="Provider", var_name="Hotel", value_name="Value")
        ).dropna(subset=["Value"])

        # ============== High-level KPIs (easy to read) ==============
        c1, c2, c3 = st.columns(3)
        if is_price_like_view and not long.empty:
            mins = long.groupby("Hotel", as_index=False).agg(MinPrice=("Value", "min"))
            winners = long.merge(mins, on="Hotel").query("Value == MinPrice")
            c1.metric("Hotels shown", f"{mins['Hotel'].nunique():,}")
            avg_min = mins["MinPrice"].mean()
            c2.metric(f"Avg cheapest price ({sel_ccy})", f"{ccy_symbol}{avg_min:.{decimals}f}")
            top = (
                winners.groupby("Provider", as_index=False).size()
                .sort_values("size", ascending=False)
                .head(1)
            )
            if not top.empty:
                c3.metric("Top cheapest provider", f"{top.iloc[0]['Provider']}  ({int(top.iloc[0]['size'])} wins)")
            else:
                c3.metric("Top cheapest provider", "—")
        elif not is_price_like_view and not long.empty:
            c1.metric("Hotels shown", f"{long['Hotel'].nunique():,}")
            avg_delta = long["Value"].mean()
            c2.metric("Avg % above cheapest", f"{avg_delta:.1f}%")
            near_share = (long["Value"] <= 5).mean() * 100 if not long.empty else 0
            c3.metric("Within 5% of cheapest", f"{near_share:.1f}%")

        st.markdown("---")

        # ============== Simple chart selector ==============
        if is_price_like_view:
            chart_choice = st.radio(
                "Choose a simple view",
                [
                    "Cheapest price by hotel (bar)",
                    "Who is cheapest most often (provider wins)",
                    "Price range by hotel (min–avg–max)",
                    "Quick comparison grid (heatmap)",
                ],
                index=0,
                horizontal=True,
            )

            # ---- 1) Cheapest price by hotel (with labels) ----
            if chart_choice == "Cheapest price by hotel (bar)":
                mins = long.groupby("Hotel", as_index=False).agg(MinPrice=("Value", "min"))
                mins["Label"] = mins["MinPrice"].round(decimals).apply(lambda v: f"{ccy_symbol}{v:.{decimals}f}")
                n_hotels = len(mins)
                topn = slider_or_fixed("Show top N (cheapest first)", 1, n_hotels, min(20, n_hotels), step=1)
                mins = mins.sort_values("MinPrice", ascending=True).head(topn)

                bars = alt.Chart(mins).mark_bar().encode(
                    y=alt.Y("Hotel:N", sort='-x', title="Hotel"),
                    x=alt.X("MinPrice:Q", title=f"Cheapest price ({sel_ccy})"),
                    tooltip=[
                        alt.Tooltip("Hotel:N"),
                        alt.Tooltip("MinPrice:Q", title=f"Cheapest price ({sel_ccy})", format=",.2f"),
                    ],
                ).properties(height=max(240, 18 * len(mins)))
                labels = alt.Chart(mins).mark_text(align="left", baseline="middle", dx=4, color='#111827').encode(
                    text="Label:N"
                )
                st.altair_chart(dark_cfg(bars + labels), use_container_width=True)

                with st.expander("How to read this"):
                    st.write(
                        f"- Each bar shows the **lowest available price** found for that hotel across all providers.\n"
                        f"- Shorter bar = cheaper. Values are shown in **{sel_ccy}**."
                    )

            # ---- 2) Provider wins (simple ranking) ----
            elif chart_choice == "Who is cheapest most often (provider wins)":
                mins = long.groupby("Hotel", as_index=False).agg(MinPrice=("Value", "min"))
                winners = long.merge(mins, on="Hotel").query("Value == MinPrice")
                win_counts = winners.groupby("Provider", as_index=False).size().rename(columns={"size": "Wins"})
                win_counts = win_counts.sort_values("Wins", ascending=False)

                bars = alt.Chart(win_counts).mark_bar().encode(
                    x=alt.X("Wins:Q", title="Number of times the provider is the cheapest"),
                    y=alt.Y("Provider:N", sort='-x', title="Provider"),
                    tooltip=["Provider:N", "Wins:Q"],
                ).properties(height=max(240, 20 * len(win_counts)))
                st.altair_chart(dark_cfg(bars), use_container_width=True)

                with st.expander("How to read this"):
                    st.write(
                        "- Shows **how often** each provider offered the **cheapest price**.\n"
                        "- Higher bar = more frequent cheapest provider."
                    )

            # ---- 3) Price range by hotel (min–avg–max) ----
            elif chart_choice == "Price range by hotel (min–avg–max)":
                stats = long.groupby("Hotel", as_index=False).agg(
                    Min=("Value", "min"),
                    Avg=("Value", "mean"),
                    Max=("Value", "max"),
                )
                n_hotels = len(stats)
                topn = slider_or_fixed("Show top N hotels (by lowest average price)", 1, n_hotels, min(20, n_hotels), step=1)
                stats = stats.sort_values("Avg", ascending=True).head(topn)

                rules = alt.Chart(stats).mark_rule().encode(
                    y=alt.Y("Hotel:N", sort='-x', title="Hotel"),
                    x=alt.X("Min:Q", title=f"Price range ({sel_ccy})"),
                    x2="Max:Q",
                    tooltip=[
                        "Hotel:N",
                        alt.Tooltip("Min:Q", title=f"Min ({sel_ccy})", format=",.2f"),
                        alt.Tooltip("Avg:Q", title=f"Avg ({sel_ccy})", format=",.2f"),
                        alt.Tooltip("Max:Q", title=f"Max ({sel_ccy})", format=",.2f"),
                    ],
                ).properties(height=max(260, 20 * len(stats)))
                points = alt.Chart(stats).mark_point(size=60, color='#111827').encode(x="Avg:Q", y="Hotel:N")
                st.altair_chart(dark_cfg(rules + points), use_container_width=True)

                with st.expander("How to read this"):
                    st.write(
                        "- Each line shows the **full price range** (min to max) for a hotel.\n"
                        "- The dot shows the **average** price across providers.\n"
                        "- **Shorter lines** and **lower dots** mean more consistently low prices."
                    )

            # ---- 4) Heatmap (renamed + simplified) ----
            else:
                n_hotels = len(df_num.columns)
                max_hotels = slider_or_fixed(
                    "Limit hotels in grid (for readability)",
                    1,                       # safe minimum
                    n_hotels,                # true max = number of hotel columns
                    min(30, n_hotels),       # sensible default
                    step=1
                )
                use_hotels = list(df_num.columns)[:max_hotels]
                heat_long = (
                    df_num[use_hotels]
                    .reset_index().rename(columns={"index": "Provider"})
                    .melt(id_vars="Provider", var_name="Hotel", value_name="Price")
                    .dropna()
                )
                heat = alt.Chart(heat_long).mark_rect().encode(
                    x=alt.X("Hotel:N", title="Hotel"),
                    y=alt.Y("Provider:N", title="Provider"),
                    color=alt.Color("Price:Q", title=f"Price ({sel_ccy})", scale=alt.Scale(scheme="greenblue")),
                    tooltip=[
                        "Provider:N", "Hotel:N",
                        alt.Tooltip("Price:Q", title=f"Price ({sel_ccy})", format=",.2f"),
                    ],
                ).properties(height=max(260, 16 * heat_long["Provider"].nunique()))
                st.altair_chart(dark_cfg(heat), use_container_width=True)

                with st.expander("How to read this"):
                    st.write(
                        "- Darker tiles indicate **lower prices**; lighter tiles indicate higher.\n"
                        "- Read across a row to compare a provider **across hotels**."
                    )

        else:
            # ============== Δ% view (simple, one-click summary) ==============
            chart_choice = st.radio(
                "Choose a simple view",
                [
                    "Average % above cheapest by provider (bar)",
                    "How far from cheapest by hotel (stacked distribution)",
                ],
                index=0,
                horizontal=True,
            )

            # ---- 1) Avg % above cheapest by provider ----
            if chart_choice == "Average % above cheapest by provider (bar)":
                prov = long.groupby("Provider", as_index=False).agg(AvgDelta=("Value", "mean"))
                prov = prov.sort_values("AvgDelta", ascending=True)  # lower is better
                bars = alt.Chart(prov).mark_bar().encode(
                    x=alt.X("AvgDelta:Q", title="Average % above the cheapest (lower is better)"),
                    y=alt.Y("Provider:N", sort='x', title="Provider"),
                    tooltip=["Provider:N", alt.Tooltip("AvgDelta:Q", title="Average Δ%", format=",.1f")],
                ).properties(height=max(240, 18 * len(prov)))
                st.altair_chart(dark_cfg(bars), use_container_width=True)

                with st.expander("How to read this"):
                    st.write(
                        "- Shows, on average, **how far each provider's prices are from the cheapest**.\n"
                        "- **Lower bars** are better (closer to the cheapest)."
                    )

            # ---- 2) Stacked distribution by hotel (0–5%, 5–10%, 10–20%, >20%) ----
            else:
                bins = pd.cut(
                    long["Value"],
                    bins=[-0.001, 5, 10, 20, long["Value"].max() + 0.001],
                    labels=["0–5%", "5–10%", "10–20%", ">20%"],
                )
                dist = (
                    long.assign(Bucket=bins)
                    .groupby(["Hotel", "Bucket"], as_index=False).size()
                )
                base = dist.groupby("Hotel", as_index=False).agg(Total=("size", "sum"))
                dist = dist.merge(base, on="Hotel")
                dist["Share"] = dist["size"] / dist["Total"] * 100

                stacked = alt.Chart(dist).mark_bar().encode(
                    x=alt.X("Share:Q", title="% of providers in bucket", stack="normalize"),
                    y=alt.Y("Hotel:N", sort='-x', title="Hotel"),
                    color=alt.Color("Bucket:N", title="% above cheapest",
                                    scale=alt.Scale(scheme="redyellowgreen")),
                    tooltip=[
                        "Hotel:N", "Bucket:N",
                        alt.Tooltip("Share:Q", title="Share", format=",.1f"),
                    ],
                ).properties(height=max(260, 18 * dist["Hotel"].nunique()))
                st.altair_chart(dark_cfg(stacked), use_container_width=True)

                with st.expander("How to read this"):
                    st.write(
                        "- For each hotel, shows **what share of providers** are within **0–5%, 5–10%, 10–20% or >20%** of the cheapest.\n"
                        "- More green (0–5%) = **more competition near the best price**."
                    )

    except ModuleNotFoundError as e:
        if 'altair' in str(e).lower():
            st.info("Altair is not installed. Run: pip install altair")
        else:
            st.error(f"Missing module: {e}")
    except Exception as e:
        st.error(f"Chart error: {e}")

# ===== DOWNLOADS =====
def choose_names(mode: str):
    if mode == "Rates ++ (price × 1.19)":
        base = "hotel_prices_rates_pp"
    elif mode == "Prices & Rates ++ (side-by-side)":
        base = "hotel_prices_prices_and_rates_pp"
    elif mode == "Δ vs Row Min (%)":
        base = "hotel_prices_delta_vs_row_min_pct"
    else:
        base = "hotel_prices_view"
    return f"{base}.csv", f"{base}.xlsx"

col1, col2 = st.columns(2)
csv_name, xlsx_name = choose_names(view_mode)
with col1:
    st.download_button(
        "⬇️ Download CSV",
        data=view_df.to_csv(index=True, encoding="utf-8-sig"),
        file_name=csv_name,
        mime="text/csv",
        disabled=view_df.empty
    )
with col2:
    st.download_button(
        "⬇️ Download Excel",
        data=to_excel_bytes(view_df),
        file_name=xlsx_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=view_df.empty
    )

# ===== DEBUG =====
if debug_show:
    st.markdown("---")
    st.subheader("Debug")
    st.write(f"Original pivot shape: {pivot.shape} (rows=providers, cols=hotels)")
    st.write(f"Filtered view shape: {view_df.shape}")
    with st.expander("Raw file preview (first 30 lines)"):
        try:
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                raw = "".join([next(f) for _ in range(30)])
            st.code(raw, language="text")
        except Exception as e:
            st.write(f"(Could not preview raw file: {e})")
    with st.expander("DataFrame head (first 10 rows)"):
        st.dataframe(pivot.head(10), use_container_width=True)
    with st.expander("dtypes"):
        st.write(pivot.dtypes)

# ===== SHAREABLE URL (modern API) =====
try:
    st.query_params.update({
        "providers": json.dumps(sel_providers),
        "hotels": json.dumps(sel_hotels),
        "view": view_mode,
        "ccy": sel_ccy,
        "decimals": str(decimals),
    })
except Exception:
    pass

# Manual reload
if refresh:
    load_pivot.clear()
    st.rerun()
