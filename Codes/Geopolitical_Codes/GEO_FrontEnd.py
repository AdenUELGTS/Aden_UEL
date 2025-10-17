# app_geo_news.py — v5
# Live RSS fetch (CNN/BBC/CNA) integrated into the frontend
# - Simple table renderer forced
# - Filters bypassed
# - Global table search (table-only)
# - Auto-refresh + manual refresh
from __future__ import annotations

import io, re, time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.express as px
from dateutil import tz

# Live fetch deps
import feedparser
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CSV = "geopolitics_news.csv"

FEEDS: Dict[str, List[str]] = {
    "CNN": ["https://rss.cnn.com/rss/edition_world.rss"],
    "BBC": ["https://feeds.bbci.co.uk/news/world/rss.xml"],
    "CNA": ["https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6311"],
}

COUNTRY_MAP = {
    "united states": ["united states", "u.s.", "us", "usa", "america"],
    "china": ["china", "prc"],
    "india": ["india"],
    "russia": ["russia"],
    "japan": ["japan"],
    "germany": ["germany"],
    "france": ["france"],
    "united kingdom": ["united kingdom", "u.k.", "uk", "britain", "british"],
    "italy": ["italy"],
    "canada": ["canada"],
    "south korea": ["south korea", "republic of korea", "rok"],
    "australia": ["australia"],
    "mexico": ["mexico"],
    "brazil": ["brazil"],
    "south africa": ["south africa"],
    "turkey": ["turkey", "t\u00fcrkiye"],
    "saudi arabia": ["saudi arabia", "saudi"],
    "argentina": ["argentina"],
    "european union": ["european union", "eu"],
    # hotspots
    "ukraine": ["ukraine"],
    "iran": ["iran"],
    "israel": ["israel"],
    "palestine": ["palestine", "palestinian"],
    "taiwan": ["taiwan"],
    "north korea": ["north korea", "dprk"],
    "pakistan": ["pakistan"],
    "nigeria": ["nigeria"],
    "egypt": ["egypt"],
    "ethiopia": ["ethiopia"],
    "philippines": ["philippines"],
    "vietnam": ["vietnam"],
}

KW_CORE  = ["sanction","sanctions","tariff","embargo","blockade","election","vote","ballot",
            "referendum","coup","treaty","accord","agreement","ceasefire","truce","talks",
            "summit","negotiation","border","territory","annex","occupation","mobilization",
            "conscription","nato","aukus","quad","asean","brics","opec","g20","un"]
KW_HEAVY = ["war","invasion","offensive","strike","airstrike","missile","drone",
            "nuclear","icbm","coup","ceasefire","blockade","embargo"]

# ─────────────────────────────────────────────────────────────────────────────
# STYLES / HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🌍 Geopolitics Monitor", page_icon="🌍", layout="wide")
st.title("🌍 Geopolitics Monitor — CNN • BBC • CNA")
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
.kpi-card { background: var(--secondary-background-color, #161a22); padding: 1rem;
            border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); }
.kpi-val { font-size: 1.6rem; font-weight: 700; line-height: 1.2; }
.kpi-sub { font-size: 0.9rem; opacity: 0.8; }
a { text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# UTILS (ported from your Geopolitical_Coding.py)
# ─────────────────────────────────────────────────────────────────────────────
def _compile_country_regex() -> Dict[str, re.Pattern]:
    res = {}
    for canon, forms in COUNTRY_MAP.items():
        parts = [r"\b" + re.escape(f).replace(r"\.", r"\.?") + r"\b" for f in forms]
        res[canon] = re.compile("|".join(parts), re.IGNORECASE)
    return res
COUNTRY_RE  = _compile_country_regex()
KW_CORE_RE  = re.compile(r"\b(" + "|".join(map(re.escape, KW_CORE )) + r")\b", re.IGNORECASE)
KW_HEAVY_RE = re.compile(r"\b(" + "|".join(map(re.escape, KW_HEAVY)) + r")\b", re.IGNORECASE)

def strip_html(x: Optional[str]) -> str:
    if not x: return ""
    return BeautifulSoup(x, "html.parser").get_text(" ", strip=True)

def detect_countries(text: str) -> List[str]:
    return sorted({canon for canon, pat in COUNTRY_RE.items() if pat.search(text)})

def score_item(title: str, summary: str) -> Tuple[int, List[str], Dict[str,int]]:
    content = f"{title} {summary}"
    countries = detect_countries(content)
    core_hits  = len(KW_CORE_RE.findall(content))
    heavy_hits = len(KW_HEAVY_RE.findall(content))
    score = 2*len(countries) + core_hits + 2*heavy_hits
    if len(countries) >= 2:
        score += 2
    return score, countries, {"core": core_hits, "heavy": heavy_hits}

def parse_pub(entry: Any) -> Optional[datetime]:
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None) or entry.get(attr)
        if t:
            try:
                ts = time.mktime(t)
                return datetime.utcfromtimestamp(ts).replace(tzinfo=tz.tzutc())
            except Exception:
                pass
    return None

def is_recent(dt: Optional[datetime], days: int) -> bool:
    if dt is None:
        return True
    cutoff = datetime.utcnow().replace(tzinfo=tz.tzutc()) - timedelta(days=days)
    return dt >= cutoff

@st.cache_data(show_spinner=True)
def fetch_rows_live(days: int, min_score: int, require_big_country: bool,
                    selected_sources: List[str], _seed: int) -> pd.DataFrame:
    """Fetch & score RSS items live. Cached; bust with `_seed` when pressing Refresh."""
    rows: List[Dict[str, Any]] = []
    total_raw = 0
    for source in selected_sources:
        urls = FEEDS.get(source, [])
        for url in urls:
            feed = feedparser.parse(url)
            items = feed.entries or []
            kept_here = 0
            for e in items:
                total_raw += 1
                title   = strip_html(getattr(e, "title", "") or e.get("title") or "")
                summary = strip_html(getattr(e, "summary", "") or e.get("description") or "")
                link    = getattr(e, "link", "") or e.get("link") or ""
                pub     = parse_pub(e)
                if not is_recent(pub, days):
                    continue
                score, countries, kw = score_item(title, summary)
                if score < min_score:
                    continue
                if require_big_country and not countries:
                    continue
                rows.append({
                    "source": source,
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "published_utc": pub.isoformat() if pub else "",
                    "impact_score": score,
                    "countries": ", ".join(countries),
                    "kw_core": kw["core"],
                    "kw_heavy": kw["heavy"],
                })
                kept_here += 1

    # de-dup & sort
    seen, out = set(), []
    for r in rows:
        key = r["link"] or r["title"]
        if key in seen:
            continue
        seen.add(key); out.append(r)
    out.sort(key=lambda r: (r["impact_score"], r["published_utc"]), reverse=True)

    df = pd.DataFrame(out, columns=[
        "source","title","summary","link","published_utc",
        "impact_score","countries","kw_core","kw_heavy"
    ])
    return df

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    try:
        if file is not None:
            return pd.read_csv(file)
        p = Path(DEFAULT_CSV)
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Source + Controls
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Source")
mode = st.sidebar.radio("Data Source", ["Live RSS", "CSV file"], index=0, help="Live mode fetches CNN/BBC/CNA on the fly.")
table_search = st.sidebar.text_input("🔎 Search table (all columns)", placeholder="Type to filter rows…").strip()

if mode == "CSV file":
    uploaded = st.sidebar.file_uploader("Upload CSV (optional). If empty, the app reads geopolitics_news.csv", type=["csv"])
else:
    uploaded = None

# Live fetch controls
enabled_sources = st.sidebar.multiselect(
    "Feeds to include",
    options=list(FEEDS.keys()),
    default=list(FEEDS.keys()),
    help="Uncheck a source to exclude it from live fetch."
)

col_l, col_r = st.sidebar.columns(2)
with col_l:
    days = st.number_input("Lookback days", min_value=1, max_value=60, value=7, step=1)
with col_r:
    min_score = st.number_input("Min impact score", min_value=0, max_value=50, value=3, step=1)

require_big = st.sidebar.checkbox("Require country mention", value=True,
                                  help="Keep only stories that mention at least one mapped country.")

save_csv = st.sidebar.checkbox("Save fetched data to CSV", value=True)
csv_path = st.sidebar.text_input("Output CSV path", value=DEFAULT_CSV)

auto = st.sidebar.checkbox("Auto-refresh", value=False, help="Periodically rerun the app to fetch new items.")
interval_min = st.sidebar.number_input("Every (minutes)", min_value=1, max_value=60, value=5, step=1)
refresh_now = st.sidebar.button("🔁 Refresh Now")

# Small session-state seed to bust cache on manual refresh
if "refresh_seed" not in st.session_state:
    st.session_state.refresh_seed = 0
if refresh_now:
    st.session_state.refresh_seed += 1

# Auto-refresh (does not spam; only if enabled)
if auto:
    st.experimental_rerun  # for type hints
    st.autorefresh(interval=interval_min * 60 * 1000, key="auto_refresh_counter")

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🔧 Diagnostics"):
    cwd = Path.cwd()
    st.write("**Working directory**:", str(cwd))
    st.write("**CSV files here**:", [p.name for p in cwd.glob("*.csv")] or "— none —")
    st.write("**Default CSV**:", DEFAULT_CSV)

# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
# Hidden defaults (kept from v4)
force_simple = True   # always use simple table renderer
bypass = True         # always bypass filters (show everything)

# Load data either live or from CSV
if mode == "Live RSS":
    df = fetch_rows_live(days=days,
                         min_score=int(min_score),
                         require_big_country=require_big,
                         selected_sources=enabled_sources,
                         _seed=st.session_state.refresh_seed)
    fetched_at = datetime.now(tz.gettz("Asia/Singapore")).strftime("%Y-%m-%d %H:%M:%S %Z")
    st.caption(f"Live fetch at **{fetched_at}** | Rows: **{len(df)}** from {', '.join(enabled_sources) or '—'}")
    if save_csv and len(df) > 0:
        try:
            out_path = Path(csv_path).resolve()
            df.to_csv(out_path, index=False)
            st.caption(f"Saved **{len(df)}** rows → `{out_path}`")
        except Exception as e:
            st.warning(f"Could not save CSV: {e}")
else:
    df = load_csv(uploaded)

# Sample data fallback (if still empty)
if df.empty:
    st.warning("No data loaded. Upload a CSV or switch to Live RSS. You can also load sample rows.")
    if st.button("Load sample data"):
        sample = io.StringIO("""source,title,summary,link,published_utc,impact_score,countries,kw_core,kw_heavy
BBC,Ceasefire talks resume,Nations push for deal,https://example.com/1,2025-10-10T03:00:00Z,6,"united states, russia",2,1
CNN,Border tensions rise,Forces mobilize at frontier,https://example.com/2,2025-10-11T12:30:00Z,5,"china, india",1,1
CNA,Sanctions expanded,Trade ministry announces new measures,https://example.com/3,2025-10-12T09:15:00Z,4,"european union",2,0
""")
        df = pd.read_csv(sample)
        st.success("Loaded sample data.")
    else:
        st.stop()

# Normalize columns
df.columns = df.columns.str.strip()
NEEDED_STR = ["source","title","summary","link","countries","published_utc"]
NEEDED_NUM = ["impact_score","kw_core","kw_heavy"]
for c in NEEDED_STR:
    if c not in df.columns: df[c] = ""
for c in NEEDED_NUM:
    if c not in df.columns: df[c] = 0

df["impact_score"] = pd.to_numeric(df["impact_score"], errors="coerce").fillna(0)
df["kw_core"] = pd.to_numeric(df["kw_core"], errors="coerce").fillna(0).astype(int)
df["kw_heavy"] = pd.to_numeric(df["kw_heavy"], errors="coerce").fillna(0).astype(int)

df["published_dt_utc"] = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
sgt = tz.gettz("Asia/Singapore")
df["published_sgt"] = df["published_dt_utc"].dt.tz_convert(sgt)
df["date_sgt"] = df["published_sgt"].dt.date
df["time_sgt"] = df["published_sgt"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
for col in ("source","title","summary","countries","link"):
    df[col] = df[col].fillna("").astype(str)

# Apply mask (bypassed)
filtered = df.copy() if bypass else df.iloc[0:0].copy()  # (keeping bypass=True behavior)

# Quick peek
st.caption(f"Loaded rows: **{len(df)}** | Columns: **{len(df.columns)}** | After filters: **{len(filtered)}**")
with st.expander("👀 First 5 rows (after filters)"):
    st.dataframe(filtered.head(5), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-val">{len(filtered)}</div><div class="kpi-sub">Articles</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-val">{filtered["source"].nunique()}</div><div class="kpi-sub">Sources</div></div>', unsafe_allow_html=True)
with c3:
    avg_imp = filtered["impact_score"].mean() if not filtered.empty else 0
    st.markdown(f'<div class="kpi-card"><div class="kpi-val">{avg_imp:.1f}</div><div class="kpi-sub">Avg Impact Score</div></div>', unsafe_allow_html=True)
with c4:
    exploded = filtered.assign(_cl=filtered["countries"].str.split(r"\s*,\s*")).explode("_cl")
    uniq_countries = exploded["_cl"].replace("", pd.NA).dropna().nunique()
    st.markdown(f'<div class="kpi-card"><div class="kpi-val">{uniq_countries}</div><div class="kpi-sub">Countries Mentioned</div></div>', unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────
col_a, col_b = st.columns([3, 2])

with col_a:
    by_day = filtered.groupby(filtered["published_sgt"].dt.date).size().reset_index(name="articles")
    by_day.rename(columns={"published_sgt": "date"}, inplace=True)
    if not by_day.empty:
        fig = px.line(by_day, x="date", y="articles", markers=True, title="Articles per day (SGT)")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for the selected filters.")

with col_b:
    top_countries = (
        exploded.query("_cl.notna() and _cl != ''")
        .groupby("_cl").size().reset_index(name="count")
        .sort_values("count", ascending=False).head(15)
    )
    if not top_countries.empty:
        fig2 = px.bar(top_countries, x="count", y="_cl", orientation="h", title="Top countries mentioned")
        fig2.update_layout(margin=dict(l=0, r=0, t=40, b=0), yaxis_title="", xaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No country mentions for current selection.")

# ─────────────────────────────────────────────────────────────────────────────
# Stories table (with table-only search)
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Stories")
show = filtered.copy()
show["Published (SGT)"] = show["published_sgt"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
show["Impact"] = show["impact_score"].astype(int)
show["Summary"] = show["summary"].str.slice(0, 220) + show["summary"].apply(
    lambda s: "…" if isinstance(s, str) and len(s) > 220 else ""
)

if table_search:
    q = table_search.lower()
    searchable = (
        show[["Published (SGT)", "source", "Impact", "countries", "title", "Summary", "link"]]
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    show = show[searchable.str.contains(q, na=False)].copy()

if show.empty:
    st.warning("Zero rows to display. (Filtering is currently bypassed.)")
else:
    show_simple = show.copy()
    show_simple["Open"] = show_simple["link"].apply(lambda u: f"[🔗 Open]({u})" if isinstance(u, str) and u else "")
    st.dataframe(
        show_simple[["Published (SGT)", "source", "Impact", "countries", "title", "Summary", "Open"]],
        use_container_width=True,
        height=540
    )

# Raw data peek + download
with st.expander("👀 Raw data (first 50 rows)"):
    st.dataframe(df.head(50), use_container_width=True)

st.download_button(
    "⬇️ Download filtered CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="geopolitics_news_filtered.csv",
    mime="text/csv"
)
