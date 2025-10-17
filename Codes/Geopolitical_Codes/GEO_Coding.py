# Geopolitical_Coding.py
# CNN, BBC, CNA geopolitics via RSS -> CSV, with loud prints so you always see output

import re, time, csv, sys, argparse, logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any

import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import tz
from pathlib import Path

FEEDS = {
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
    if dt is None:  # keep if unknown date
        return True
    cutoff = datetime.utcnow().replace(tzinfo=tz.tzutc()) - timedelta(days=days)
    return dt >= cutoff

def fetch_rows(days: int, min_score: int, require_big_country: bool, logger: logging.Logger) -> List[Dict[str, Any]]:
    rows, total_raw = [], 0
    for source, urls in FEEDS.items():
        for url in urls:
            logger.info(f"[{source}] Fetching: {url}")
            feed = feedparser.parse(url)
            items = feed.entries or []
            logger.info(f"[{source}] Items: {len(items)}")
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
            logger.info(f"[{source}] Kept: {kept_here}")
    # de-dup & sort
    seen, out = set(), []
    for r in rows:
        key = r["link"] or r["title"]
        if key in seen: 
            continue
        seen.add(key); out.append(r)
    out.sort(key=lambda r: (r["impact_score"], r["published_utc"]), reverse=True)
    logger.info(f"[ALL] Parsed={total_raw} Kept(after filters)={len(out)}")
    return out

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export geopolitics-heavy news to CSV (CNN/BBC/CNA)")
    ap.add_argument("--days", type=int, default=7, help="Lookback days (UTC)")
    ap.add_argument("--min-score", type=int, default=3, help="Minimum impact score")
    ap.add_argument("--out", default="geopolitics_news.csv", help="Output CSV path")
    ap.add_argument("--no-big-country", action="store_true", help="Do not require a big-country mention")
    ap.add_argument("--verbose", "-v", action="store_true", help="Verbose logs")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    print(f"Running with: days={args.days}, min_score={args.min_score}, require_big_country={not args.no_big_country}")

    rows = fetch_rows(args.days, args.min_score, not args.no_big_country, logging.getLogger("geo"))
    df = pd.DataFrame(rows, columns=[
        "source","title","summary","link","published_utc",
        "impact_score","countries","kw_core","kw_heavy"
    ])
    out_path = Path(args.out).resolve()
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Saved {len(df)} rows -> {out_path}")
    if len(df) == 0:
        print("No rows matched. Try lowering --min-score, adding --no-big-country, or increasing --days.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
