# -*- coding: utf-8 -*-
# Clean_LiveNation.py
# Run directly (no arguments). Creates/overwrites exactly ONE file:
#   C:\Users\aden.chong\OneDrive - United Engineers Limited\Yim Jun Teck's files - PAG_A1\livenation_events_long.csv

import os
import re
import csv
import time
import datetime as dt
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
from calendar import month_abbr
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ========= hard-coded paths =========
INPUT_CSV  = Path(r"C:\Users\aden.chong\OneDrive - United Engineers Limited\Yim Jun Teck's files - PAG_A1\LiveNation_Unclean_Data.csv")
OUTPUT_CSV = Path(r"C:\Users\aden.chong\OneDrive - United Engineers Limited\Yim Jun Teck's files - PAG_A1\livenation_events_long.csv")

# Geocoding preferences (no side files)
DEFAULT_COUNTRY = "Singapore"  # append to queries to help disambiguate
COUNTRY_CODE    = "sg"         # restrict to SG; set to None for global
SLEEP_SECONDS   = 1.1          # Nominatim politeness

# ========= safe overwrite (no extra files left behind) =========
def safe_overwrite_csv(df: pd.DataFrame, out_path: Path, attempts: int = 10, base_sleep: float = 0.6):
    out_path = Path(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass

    last_err = None
    for i in range(1, attempts + 1):
        try:
            df.to_csv(tmp, index=False, encoding="utf-8")
            os.replace(tmp, out_path)  # atomic replace
            return
        except PermissionError as e:
            last_err = e
            time.sleep(base_sleep * i)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            raise
    # As a last resort, write directly (still only one final file if replace fails)
    df.to_csv(out_path, index=False, encoding="utf-8")
    if last_err:
        print(f"Warning: saved directly due to lock. Last error: {last_err}")

# ========= parsing (unclean -> long) =========
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
month_header_re = re.compile(rf"^(?:{MONTHS})\s+\d{{4}}$", re.I)
location_re = re.compile(r"^\w.*\s\|\s.+$")   # "City | Venue"

month_map = {m.lower(): i for i, m in enumerate(month_abbr) if m}
month_map.update({"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12})

def parse_events_from_text(text: str) -> List[Dict[str, Any]]:
    blocks = [b.strip() for b in text.split("Find Tickets") if b.strip()]
    events = []
    current_year = None
    for b in blocks:
        lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        filtered = []
        for ln in lines:
            low = ln.lower()
            if low.startswith("search by artist") or low.startswith("filters"):
                continue
            if low in ("location", "date", "genre"):
                continue
            if month_header_re.match(ln):
                m = re.search(r"(\d{4})", ln)
                if m:
                    current_year = int(m.group(1))
                continue
            filtered.append(ln)
        loc_idx = next((i for i in range(len(filtered)-1, -1, -1)
                        if location_re.match(filtered[i])), None)
        if loc_idx is None or loc_idx < 2:
            continue
        location_line = filtered[loc_idx]
        artist_line   = filtered[loc_idx-1] if loc_idx-1 >= 0 else ""
        title_line    = filtered[loc_idx-2] if loc_idx-2 >= 0 else ""
        date_line     = filtered[loc_idx-3] if loc_idx-3 >= 0 else ""
        try:
            city, venue = [s.strip() for s in location_line.split("|", 1)]
        except ValueError:
            city, venue = location_line.strip(), ""
        events.append({
            "DateRaw": date_line,
            "Title": title_line,
            "Artist": artist_line,
            "City": city,
            "Venue": venue,
            "YearHint": current_year,
        })
    return events

def parse_date_tokens(date_raw: str, year_hint: Optional[int]) -> Tuple[Optional[str], Optional[str]]:
    tok = (date_raw or "").strip()
    pairs = re.findall(r"(\d{1,2})([A-Za-z]{3})", tok)
    if not pairs:
        return None, None
    def to_date(day: str, mon3: str, year: Optional[int]):
        mi = month_map.get(mon3.lower())
        if not (mi and year):
            return None
        try:
            return dt.date(int(year), int(mi), int(day))
        except Exception:
            return None
    dates = [d for d in (to_date(d, m, year_hint) for d, m in pairs) if d]
    if not dates:
        return None, None
    return dates[0].isoformat(), dates[-1].isoformat()

def slug(title: str, idx: int) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(title)).strip("_")
    if len(s) > 40:
        s = s[:40].rstrip("_")
    return f"{idx+1:02d}_{s or 'Event'}"

def build_long_from_unclean(unclean_csv: Path) -> pd.DataFrame:
    try:
        df_raw = pd.read_csv(unclean_csv, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df_raw = pd.read_csv(unclean_csv, encoding="cp1252")
    if df_raw.shape[1] == 1:
        text = str(df_raw.iloc[0, 0])
    else:
        text = "\n".join(df_raw.astype(str).fillna("").agg(" ".join, axis=1).tolist())
    events = parse_events_from_text(text)
    df = pd.DataFrame(events)
    if df.empty:
        return pd.DataFrame(columns=["EventID","Title","Artist","City","Venue","DateRaw","StartDate","EndDate","YearHint"])
    start_end = [parse_date_tokens(r, y) for r, y in zip(df["DateRaw"], df["YearHint"])]
    df["StartDate"] = [a for a, _ in start_end]
    df["EndDate"]   = [b for _, b in start_end]
    df["EventID"]   = [slug(t, i) for i, t in enumerate(df["Title"])]
    return df[["EventID","Title","Artist","City","Venue","DateRaw","StartDate","EndDate","YearHint"]]

# ========= geocoding (memory-only; no extra files) =========
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())

def clean_venue_name(s: str) -> str:
    s = norm(s)
    s = re.sub(r"\bSG\b", "Singapore", s, flags=re.I)
    if re.search(r"\bIndoor Stadium\b", s, re.I) and "Singapore Indoor Stadium" not in s:
        s = s.replace("Indoor Stadium", "Singapore Indoor Stadium")
    s = s.replace("Star Theatre", "The Star Theatre")
    s = re.sub(r"\b(Find Tickets|Presented by Live Nation|Live Nation)\b", "", s, flags=re.I)
    return norm(s)

def build_query(venue: str, city: str, country: Optional[str]) -> str:
    parts = [clean_venue_name(venue), norm(city)]
    if country:
        parts.append(country)
    return ", ".join([p for p in parts if p])

_geolocator = Nominatim(user_agent="ue-pag-livenation-onefile")
_geocode = RateLimiter(_geolocator.geocode, min_delay_seconds=SLEEP_SECONDS, swallow_exceptions=True)

def geocode_one(query: str):
    if not query:
        return None, None, None
    try:
        if COUNTRY_CODE:
            loc = _geocode(query, exactly_one=True, addressdetails=False, country_codes=COUNTRY_CODE, timeout=20)
        else:
            loc = _geocode(query, exactly_one=True, addressdetails=False, timeout=20)
        if loc:
            return float(loc.latitude), float(loc.longitude), getattr(loc, "address", None)
    except Exception:
        pass
    return None, None, None

# ========= main =========
def main():
    # 1) Clean to long
    df = build_long_from_unclean(INPUT_CSV)

    # 2) Geocode unique queries (memory cache only)
    df["_q"] = df.apply(lambda r: build_query(r.get("Venue",""), r.get("City",""), DEFAULT_COUNTRY), axis=1)
    uniques = sorted(set([q for q in df["_q"].astype(str) if q]))
    mem_cache: Dict[str, Tuple[Optional[float], Optional[float], Optional[str], str]] = {}
    print(f"Unique venues to geocode: {len(uniques)}")

    for q in uniques:
        lat, lon, addr = geocode_one(q)
        prov = "nominatim" if lat is not None else "not_found"
        mem_cache[q] = (lat, lon, addr, prov)

    df["Latitude"]        = df["_q"].map(lambda q: mem_cache.get(q, (None,None,None,None))[0])
    df["Longitude"]       = df["_q"].map(lambda q: mem_cache.get(q, (None,None,None,None))[1])
    df["GeocodedAddress"] = df["_q"].map(lambda q: mem_cache.get(q, (None,None,None,None))[2])
    df["GeocodedBy"]      = df["_q"].map(lambda q: mem_cache.get(q, (None,None,None,None))[3])
    df.drop(columns=["_q"], inplace=True)

    # 3) Save (only one file)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    safe_overwrite_csv(df, OUTPUT_CSV)
    print(f"Saved -> {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
