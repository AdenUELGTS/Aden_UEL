# bot_multi_hotels.py — multi-hotel selection + prices + pretty ALL-HOTEL COMPARISON
# deps: python-telegram-bot==21.4, python-dotenv
# run : .\.venv\Scripts\python.exe -u bot.py

import os
import csv
import logging
from typing import Optional, List, Tuple, Set, Dict
import re

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    ContextTypes, MessageHandler, filters
)
from telegram.request import HTTPXRequest

# ───────── env ─────────
load_dotenv()
TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
CSV_PATH = (os.getenv("CSV_PATH") or "").strip()
CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()  # optional startup DM
PIVOT_HOTEL = (os.getenv("PIVOT_HOTEL") or "").strip()   # single or CSV/semicolon list
PROXY_URL = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")

# Parity thresholds (percent spread between max and min)
PARITY_WARN = float(os.getenv("PARITY_WARNING", "5.0"))      # >= -> WARN
PARITY_CRIT = float(os.getenv("PARITY_CRITICAL", "10.0"))    # >= -> CRITICAL

# ───────── state ─────────
def _parse_initial_selection(val: str) -> Set[str]:
    if not val:
        return set()
    parts = [p.strip() for p in re.split(r"[;,]", val) if p.strip()]
    return set(parts)

selected_hotels: Set[str] = _parse_initial_selection(PIVOT_HOTEL)

# ───────── logging ─────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("par_competitor_bot")

MAX_CHUNK = 3800   # Telegram ~4096 hard limit; keep margin
PAGE_SIZE = 9      # picker grid: 3x3

# ───────── file helpers ─────────
def _open_csv_reader(path: str):
    """Open CSV with robust encoding fallback and return (reader, file_handle)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    encodings = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            fh = open(path, "r", encoding=enc, errors="replace", newline="")
            rdr = csv.reader(fh)
            return rdr, fh
        except Exception as e:
            last_err = e
    raise last_err if last_err else RuntimeError("Failed to read CSV")

def read_header(path: str) -> List[str]:
    rdr, fh = _open_csv_reader(path)
    try:
        for row in rdr:
            if row and any(cell.strip() for cell in row):
                return [cell.strip() for cell in row]
        return []
    finally:
        fh.close()

def list_hotels_from_header(header: List[str]) -> List[str]:
    """Pivot layout: first column is typically 'Provider'. Return all others."""
    if not header:
        return []
    if "Provider" in header:
        return [h for h in header if h != "Provider"]
    return header[1:] if len(header) > 1 else []

def provider_col_index(header: List[str]) -> int:
    if not header:
        return 0
    return header.index("Provider") if "Provider" in header else 0

def read_hotel_column(path: str, hotel_name: str, limit: Optional[int] = None) -> Tuple[List[Tuple[str, str]], int]:
    """Return (rows, total_rows) for one hotel column."""
    header = read_header(path)
    if not header:
        raise RuntimeError("CSV seems empty or header is missing.")
    name_lower = hotel_name.lower().strip()
    by_lower = {h.lower(): h for h in header}
    if name_lower not in by_lower:
        raise KeyError(f"Hotel column not found: {hotel_name}")
    hotel_col = by_lower[name_lower]
    hotel_idx = header.index(hotel_col)
    prov_idx = provider_col_index(header)

    rdr, fh = _open_csv_reader(path)
    out: List[Tuple[str, str]] = []
    total = 0
    try:
        first = True
        for row in rdr:
            if first:
                first = False
                continue  # skip header
            if not row:
                continue
            if hotel_idx >= len(row) and prov_idx >= len(row):
                continue
            provider = (row[prov_idx] if prov_idx < len(row) else "").strip()
            value = (row[hotel_idx] if hotel_idx < len(row) else "").strip()
            total += 1
            if limit is None or len(out) < limit:
                out.append((provider, value))
    finally:
        fh.close()
    return out, total

# ───────── parsing helpers ─────────
_num_re = re.compile(r"(\d+(?:\.\d+)?)")

def parse_price(cell: str) -> Optional[float]:
    """Extract a numeric price from a cell (handles '$200', 'S$ 199.50', 'No price', '', etc.)."""
    if not cell:
        return None
    s = str(cell).replace(",", "").strip()
    m = _num_re.search(s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

# ───────── UI helpers ─────────
def _short(s: Optional[str], n: int = 24) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[: n - 1] + "…"

def _selection_summary() -> str:
    if not selected_hotels:
        return "no current selections"
    items = sorted(selected_hotels)
    if len(items) <= 5:
        return ", ".join(items)
    return ", ".join(items[:5]) + f"  (+{len(items)-5} more)"

def build_start_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📜 Show all hotels", callback_data="menu:show_hotels")],
        [InlineKeyboardButton("➕ Add Hotel(s)", callback_data="menu:add_hotels"),
         InlineKeyboardButton("➖ Remove Hotel(s)", callback_data="menu:remove_hotels")],
        [InlineKeyboardButton("🔎 Find Prices", callback_data="menu:find_prices")],
        [InlineKeyboardButton("📊 Compare (All Hotels)", callback_data="menu:compare_all")],
    ])

async def send_menu_below(update: Update, ctx: ContextTypes.DEFAULT_TYPE, note: Optional[str] = None):
    text = (note + "\n\n" if note else "") + \
           "Welcome to PAR_Competitor_Pricing_Bot\n" \
           "Get Started:\n\n" \
           f"Currently selected hotels: {_selection_summary()}"
    await ctx.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text,
        reply_markup=build_start_menu()
    )

def _build_picker(hotels: List[str], page: int, mode: str) -> Tuple[str, InlineKeyboardMarkup]:
    """mode: 'add' (show all) or 'rem' (only selected)."""
    src = hotels if mode == "add" else [h for h in hotels if h in selected_hotels]
    total = len(src)
    if total == 0:
        title = "No hotels to display." if mode == "add" else "No selected hotels to remove."
        return (title, InlineKeyboardMarkup([[InlineKeyboardButton("⬅ Back", callback_data="picker:done")]]))

    pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    page = max(0, min(page, pages - 1))
    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    page_items = src[start:end]

    rows = []
    for i, name in enumerate(page_items):
        abs_idx = hotels.index(name)
        checked = "☑ " if name in selected_hotels else "☐ "
        label = checked + _short(name, 20)
        btn = InlineKeyboardButton(label, callback_data=f"picker:{mode}:toggle:{abs_idx}")
        if i % 3 == 0:
            rows.append([btn])
        else:
            rows[-1].append(btn)

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("◀ Prev", callback_data=f"picker:{mode}:page:{page-1}"))
    nav.append(InlineKeyboardButton("✅ Done", callback_data="picker:done"))
    if page < pages - 1:
        nav.append(InlineKeyboardButton("Next ▶", callback_data=f"picker:{mode}:page:{page+1}"))
    rows.append(nav)

    rows.append([InlineKeyboardButton("⛔ Clear all selections", callback_data="picker:clear_all")])

    title = f"{'Add' if mode=='add' else 'Remove'} hotels ({page+1}/{pages}) — selected: {len(selected_hotels)}"
    return title, InlineKeyboardMarkup(rows)

async def _send_or_edit_picker(update: Update, ctx: ContextTypes.DEFAULT_TYPE, mode: str, page: int = 0):
    header = read_header(CSV_PATH)
    hotels = list_hotels_from_header(header)
    title, kb = _build_picker(hotels, page, mode)
    q = update.callback_query
    if q and q.message:
        await q.edit_message_text(title, reply_markup=kb)
    else:
        await ctx.bot.send_message(chat_id=update.effective_chat.id, text=title, reply_markup=kb)

async def _send_chunked(chat_id: int, text: str, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Split long text by lines to keep under Telegram limits."""
    lines = text.splitlines()
    buf = ""
    for ln in lines:
        if len(buf) + len(ln) + 1 > MAX_CHUNK:
            await ctx.bot.send_message(chat_id, buf or " ")
            buf = ""
        buf += ("" if not buf else "\n") + ln
    if buf:
        await ctx.bot.send_message(chat_id, buf)

# ───────── price printing for selected ─────────
async def _send_prices_for_selected(update: Update, ctx: ContextTypes.DEFAULT_TYPE, limit: int = 30):
    if not selected_hotels:
        await ctx.bot.send_message(update.effective_chat.id, "⚠️ No hotels selected. Tap “Add Hotel(s)” to choose some.")
        return

    header = read_header(CSV_PATH)
    hotels = list_hotels_from_header(header)

    valid = [h for h in sorted(selected_hotels) if h in hotels]
    invalid = [h for h in sorted(selected_hotels) if h not in hotels]
    if invalid:
        await ctx.bot.send_message(update.effective_chat.id, "Some selections no longer exist in CSV:\n• " + "\n• ".join(invalid))

    for hotel in valid:
        try:
            rows, total = read_hotel_column(CSV_PATH, hotel, limit=limit)
            lines = [f"🏨 {hotel} — Provider → Price (first {len(rows)}/{total})", "—"]
            for p, v in rows:
                lines.append(f"{p or '(blank provider)'} : {v or ''}")
            await _send_chunked(update.effective_chat.id, "\n".join(lines), ctx)
        except Exception as e:
            await ctx.bot.send_message(update.effective_chat.id, f"❌ {hotel}: {e}")

# ───────── ALL-HOTELS COMPARISON (pretty print) ─────────
def _read_all_prices(path: str) -> Dict[str, List[Tuple[str, float]]]:
    """Return {hotel: [(provider, price_float), ...]} for all numeric cells."""
    header = read_header(path)
    hotels = list_hotels_from_header(header)
    prov_idx = provider_col_index(header)
    idxs = [(h, header.index(h)) for h in hotels]

    data: Dict[str, List[Tuple[str, float]]] = {h: [] for h in hotels}
    rdr, fh = _open_csv_reader(path)
    try:
        first = True
        for row in rdr:
            if first:
                first = False
                continue
            if not row:
                continue
            provider = (row[prov_idx] if prov_idx < len(row) else "").strip()
            for h, j in idxs:
                if j < len(row):
                    price = parse_price(row[j])
                    if price is not None:
                        data[h].append((provider, price))
    finally:
        fh.close()
    return data

def _parity_label(spread_pct: float) -> str:
    if spread_pct >= PARITY_CRIT:
        return "CRITICAL"
    if spread_pct >= PARITY_WARN:
        return "WARN"
    return "OK"

def _bar(pct: float, width: int = 20) -> str:
    """Unicode bar for quick visual (0–100%)."""
    pct = max(0.0, min(100.0, pct))
    filled = int(round(pct / 100 * width))
    return "█" * filled + "░" * (width - filled)

def _format_hotel_block(r: Dict, rank: int) -> str:
    """Readable multi-line block for a single hotel."""
    return (
        f"{rank}. 🏨 {r['hotel']}\n"
        f"   • Providers: {r['n']}\n"
        f"   • Lowest : ${r['min_price']:.2f} via {r['min_prov']}\n"
        f"   • Highest: ${r['max_price']:.2f} via {r['max_prov']}\n"
        f"   • Average: ${r['avg_price']:.2f}\n"
        f"   • Spread : {r['spread_pct']:.1f}%  {_bar(r['spread_pct'])}  → {r['parity']}"
    )

async def _compare_all_prices(update: Update, ctx: ContextTypes.DEFAULT_TYPE, top_n: int = 60, batch: int = 10):
    """
    Pretty-print comparison:
      - Sorted by worst spread first
      - One clearly spaced block per hotel
      - Sent in batches (default 10 hotels/message) to avoid clumping
    """
    data = _read_all_prices(CSV_PATH)

    results = []
    for hotel, items in data.items():
        if not items:
            continue
        n = len(items)
        min_prov, min_price = min(items, key=lambda x: x[1])
        max_prov, max_price = max(items, key=lambda x: x[1])
        avg_price = sum(p for _, p in items) / n
        spread_pct = ((max_price - min_price) / min_price * 100.0) if min_price > 0 else 0.0
        parity = _parity_label(spread_pct)
        results.append({
            "hotel": hotel,
            "n": n,
            "min_price": min_price,
            "min_prov": min_prov,
            "max_price": max_price,
            "max_prov": max_prov,
            "avg_price": avg_price,
            "spread_pct": spread_pct,
            "parity": parity,
        })

    results.sort(key=lambda r: r["spread_pct"], reverse=True)
    results = results[:top_n]

    if not results:
        await ctx.bot.send_message(update.effective_chat.id, "📊 Comparison: No numeric prices were found in the CSV.")
        return

    # Header
    header = (
        f"📊 All-Hotels Comparison\n"
        f"   Thresholds: WARN ≥ {PARITY_WARN:.1f}% · CRITICAL ≥ {PARITY_CRIT:.1f}%\n"
        f"   Sorted by widest spread (worst) first.\n"
        f"—"
    )
    await ctx.bot.send_message(update.effective_chat.id, header)

    # Send in neat batches (each block separated by a blank line)
    for i in range(0, len(results), batch):
        chunk = results[i:i + batch]
        blocks = []
        for j, r in enumerate(chunk, start=i + 1):
            blocks.append(_format_hotel_block(r, j))
        text = "\n\n".join(blocks)  # blank line between hotels
        await _send_chunked(update.effective_chat.id, text, ctx)

# ───────── commands ─────────
HELP = (
    "Welcome to PAR_Competitor_Pricing_Bot\n\n"
    "Commands:\n"
    "/start — open the start menu\n"
    "/menu — open the start menu\n"
    "/help — show this help\n"
    "/compare — compare prices across all hotels\n"
)

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await send_menu_below(update, ctx)

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP)

async def cmd_compare(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _compare_all_prices(update, ctx)
    await send_menu_below(update, ctx, note="📦 Comparison complete.")

# ───────── button callbacks ─────────
async def on_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global selected_hotels
    q = update.callback_query
    data = (q.data or "")
    await q.answer()

    try:
        if data == "menu:show_hotels" or data == "menu:add_hotels":
            await _send_or_edit_picker(update, ctx, mode="add", page=0)
            return

        if data == "menu:remove_hotels":
            await _send_or_edit_picker(update, ctx, mode="rem", page=0)
            return

        if data == "menu:find_prices":
            await _send_prices_for_selected(update, ctx, limit=30)
            await send_menu_below(update, ctx, note="📦 Results are above.")
            return

        if data == "menu:compare_all":
            await _compare_all_prices(update, ctx)
            await send_menu_below(update, ctx, note="📦 Comparison complete.")
            return

        # picker events
        if data == "picker:done":
            await send_menu_below(update, ctx, note="✅ Selection updated.")
            return

        if data == "picker:clear_all":
            selected_hotels.clear()
            await send_menu_below(update, ctx, note="🧹 Cleared all selections.")
            return

        if data.startswith("picker:add:page:"):
            page = int(data.split(":")[3])
            await _send_or_edit_picker(update, ctx, mode="add", page=page)
            return

        if data.startswith("picker:rem:page:"):
            page = int(data.split(":")[3])
            await _send_or_edit_picker(update, ctx, mode="rem", page=page)
            return

        if data.startswith("picker:add:toggle:"):
            idx = int(data.split(":")[3])
            header = read_header(CSV_PATH)
            hotels = list_hotels_from_header(header)
            if 0 <= idx < len(hotels):
                name = hotels[idx]
                if name in selected_hotels:
                    selected_hotels.remove(name)
                else:
                    selected_hotels.add(name)
            await _send_or_edit_picker(update, ctx, mode="add", page=0)
            return

        if data.startswith("picker:rem:toggle:"):
            idx = int(data.split(":")[3])
            header = read_header(CSV_PATH)
            hotels = list_hotels_from_header(header)
            if 0 <= idx < len(hotels):
                name = hotels[idx]
                if name in selected_hotels:
                    selected_hotels.remove(name)
            await _send_or_edit_picker(update, ctx, mode="rem", page=0)
            return

    except Exception as e:
        await ctx.bot.send_message(update.effective_chat.id, f"Error: {e}")
        await send_menu_below(update, ctx, note="⚠️ An error occurred.")

# ───────── app wiring ─────────
async def _post_init(app: Application):
    try:
        me = await app.bot.get_me()
        print(f"✅ connected: @{me.username} ({me.id})")
    except Exception as e:
        print(f"❌ cannot reach Telegram API: {e}")
        return
    try:
        if CHAT_ID:
            await app.bot.send_message(
                chat_id=CHAT_ID,
                text="Welcome to PAR_Competitor_Pricing_Bot\n"
                     "Get Started:\n\n"
                     f"Currently selected hotels: {', '.join(sorted(selected_hotels)) or 'no current selections'}",
                reply_markup=build_start_menu()
            )
        else:
            print("ℹ️  set TELEGRAM_CHAT_ID in .env if you want a startup DM.")
    except Exception as e:
        print(f"warn: couldn't send startup message: {e}")

def build_app() -> Application:
    if not TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")
    builder = Application.builder().token(TOKEN).post_init(_post_init)
    if PROXY_URL:
        builder = builder.request(HTTPXRequest(proxy_url=PROXY_URL))
        print(f"🌐 using proxy: {PROXY_URL}")
    app = builder.build()

    # commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("menu", cmd_start))   # alias
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("compare", cmd_compare))

    # callbacks
    app.add_handler(CallbackQueryHandler(on_button))

    # debug sink
    async def _dbg(update: Update, context: ContextTypes.DEFAULT_TYPE):
        log.info("DBG text: %r", getattr(update.message, "text", None))
    app.add_handler(MessageHandler(filters.ALL, _dbg), group=-1)

    return app

def main():                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    if not CSV_PATH:
        print("ERROR: CSV_PATH missing from .env")
        return
    folder = os.path.dirname(CSV_PATH) or "."
    if not os.path.exists(folder):
        print(f"ERROR: folder not found: {folder}")
        return

    app = build_app()
    print("🤖 PAR_Competitor_Pricing_Bot starting…")
    print(f"📄 CSV: {CSV_PATH}")
    print(f"🧩 Starting selections: {', '.join(sorted(selected_hotels)) or '(none)'}")
    print(f"💬 Startup chat: {CHAT_ID or '(none)'}")
    app.run_polling(drop_pending_updates=True, close_loop=False)

if __name__ == "__main__":
    main()
