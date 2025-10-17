# pivot_hotel_prices.py
# pip install pandas

import pandas as pd
import glob, os, re, subprocess, sys, shutil
from typing import Optional, List
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────
# Input folders to search (check both styles)
INPUT_DIRS = [
    r"C:\Users\aden.chong\OneDrive - United Engineers Limited\Aden\Week5\Extracted_Hotel_Data",
    r"C:\Users\aden.chong\OneDrive - United Engineers Limited\Aden\Week5\Extracted Hotel Data",
]
# Output folder (where we will SAVE the final CSV)
OUTPUT_DIR = r"C:\Users\aden.chong\OneDrive - United Engineers Limited\Aden\Week5\Extracted_Hotel_Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Git repo target
REPO_PATH      = r"C:\PAG\pa-rochester-rate-monitor"
TARGET_BRANCH  = "aden-branch"
TARGET_FILENAME = "hotel_prices_provider_rows.csv"

# Order of providers (rows) in the pivot
PROVIDER_ORDER = ["Agoda", "Booking.com", "Expedia", "Traveloka", "Trip.com", "Klook"]

BAD_TEXT_RE = re.compile(
    r"(?:no\s*price|no\s*rate|try\s*other\s*date|no\s*availability|sold\s*out|n/?a|not\s*found|no\s*data)",
    re.I
)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def find_input_files() -> List[str]:
    """Find CSVs under either folder with space/underscore file naming."""
    patterns = []
    for base in INPUT_DIRS:
        patterns.append(os.path.join(base, "Extracted_Hotel_Data_*.csv"))
        patterns.append(os.path.join(base, "Extracted Hotel Data_*.csv"))
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    # de-duplicate and sort
    return sorted(set(files))

def hotel_name_from_filename(path: str) -> str:
    base = os.path.basename(path)
    name = re.sub(r"^Extracted[_\s]*Hotel[_\s]*Data_", "", base, flags=re.IGNORECASE)
    name = re.sub(r"\.csv$", "", name, flags=re.IGNORECASE)
    return name.strip()

def normalize_provider(raw: str) -> Optional[str]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    v = raw.strip().lower()
    v = re.sub(r"^https?://(www\.)?", "", v)
    if "tripadvisor" in v: return None
    if "agoda" in v: return "Agoda"
    if "booking" in v: return "Booking.com"
    if "expedia" in v: return "Expedia"
    if "traveloka" in v: return "Traveloka"
    if "trip.com" in v or "tripcom" in v or "ctrip" in v: return "Trip.com"
    if "klook" in v or "kloook" in v: return "Klook"
    return None

def clean_price(val) -> Optional[float]:
    if pd.isna(val): return None
    s = str(val)
    if BAD_TEXT_RE.search(s): return None
    s = s.replace(",", "")
    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else None

def read_two_col_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, header=None, engine="python", on_bad_lines="skip", encoding=enc)
            df = df.iloc[:, :2].copy()
            df.columns = ["raw_provider", "raw_price"]
            df = df.dropna(how="all")
            df["hotel"] = hotel_name_from_filename(path)
            return df
        except Exception:
            continue
    raise RuntimeError(f"Failed to read {path}")

# ─────────────────────────────────────────────────────────────
# CORE
# ─────────────────────────────────────────────────────────────
def build_pivot() -> pd.DataFrame:
    files = find_input_files()
    if not files:
        looked = "\n".join([
            f"  - {os.path.join(d, 'Extracted_Hotel_Data_*.csv')}" for d in INPUT_DIRS
        ] + [
            f"  - {os.path.join(d, 'Extracted Hotel Data_*.csv')}" for d in INPUT_DIRS
        ])
        raise SystemExit(
            "No files found.\nLooked in:\n" + looked +
            "\n\nTip: ensure your CSVs are named like 'Extracted_Hotel_Data_<Hotel>.csv' or 'Extracted Hotel Data_<Hotel>.csv'."
        )

    all_hotels: List[str] = sorted({hotel_name_from_filename(p) for p in files})
    frames: List[pd.DataFrame] = []

    for p in files:
        try:
            df0 = read_two_col_csv(p)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
            continue

        hotel = hotel_name_from_filename(p)
        m_placeholder_any = (
            df0["raw_price"].astype(str).str.contains(BAD_TEXT_RE, na=False)
            | df0["raw_provider"].astype(str).str.contains(BAD_TEXT_RE, na=False)
        )
        m_provider_is_hotel = (
            df0["raw_provider"].astype(str).str.strip().str.casefold()
            == hotel.strip().casefold()
        )
        m_price_has_digit = df0["raw_price"].astype(str).str.contains(r"\d", na=False)

        df = df0[~m_placeholder_any & ~m_provider_is_hotel & m_price_has_digit].copy()
        if df.empty:
            continue

        df = df.dropna(subset=["raw_provider"])
        df["provider_norm"] = df["raw_provider"].astype(str).apply(normalize_provider)
        df["price"] = df["raw_price"].apply(clean_price)
        df = df[df["provider_norm"].isin(PROVIDER_ORDER)]
        df = df[df["price"].notna()]
        if df.empty:
            continue

        df = df.drop_duplicates(subset=["hotel", "provider_norm", "price"])
        frames.append(df[["hotel", "provider_norm", "price"]])

    if not frames:
        # No usable rows, return an all-NaN pivot with all providers × hotels
        empty = pd.DataFrame(
            index=pd.Index(PROVIDER_ORDER, name="provider_norm"),
            columns=all_hotels,
            dtype=float,
        )
        return empty

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["hotel", "provider_norm", "price"])
    combined = combined.sort_values(["hotel", "provider_norm", "price"]).reset_index(drop=True)

    pivot = combined.pivot_table(
        index="provider_norm", columns="hotel", values="price", aggfunc="min", dropna=False
    )
    pivot = pivot.reindex(PROVIDER_ORDER)
    pivot = pivot.reindex(columns=all_hotels)
    return pivot

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pivot = build_pivot()

    # Save to OUTPUT_DIR (underscore folder)
    out_csv = os.path.join(OUTPUT_DIR, "hotel_prices_provider_rows.csv")
    pivot.to_csv(out_csv, encoding="utf-8-sig", float_format="%.2f", index=True, index_label="Provider")
    print(f"✅ Saved: {out_csv}")

    # Copy CSV into repo
    dest = os.path.join(REPO_PATH, TARGET_FILENAME)
    os.makedirs(REPO_PATH, exist_ok=True)
    shutil.copy2(out_csv, dest)
    print(f"✅ Copied to repo: {dest}")

    # Optional: pull-first to avoid conflicts
    subprocess.run(["git", "-C", REPO_PATH, "fetch", "--prune"], check=False)
    subprocess.run(["git", "-C", REPO_PATH, "checkout", TARGET_BRANCH], check=False)
    subprocess.run(["git", "-C", REPO_PATH, "pull", "--rebase", "origin", TARGET_BRANCH], check=False)

    # Commit & push
    cmds = [
        ["git", "-C", REPO_PATH, "add", TARGET_FILENAME],
        ["git", "-C", REPO_PATH, "commit", "-m", f"Auto-update on {datetime.now():%Y-%m-%d %H:%M}"],
        ["git", "-C", REPO_PATH, "push", "origin", TARGET_BRANCH],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[WARN] {' '.join(cmd)}\n{result.stderr}")
        else:
            print(result.stdout.strip())
    print("🚀 Git push completed.")
