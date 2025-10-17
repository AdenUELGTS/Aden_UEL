# tools/auto_requirements.py
# Scan project files for imports, map to pip packages, and optionally install them.
# Usage (from project root, with venv activated):
#   python tools/auto_requirements.py
#   python tools/auto_requirements.py --apply
#   python tools/auto_requirements.py --export requirements.txt
#   python tools/auto_requirements.py --root Aden_UEL --apply --export requirements.txt

import os, sys, ast, json, argparse, subprocess, re
from pathlib import Path
from typing import Set, Dict, Iterable, Tuple, List, Optional

# --- Pip package mapping for common import->package differences
MODULE_TO_PIP: Dict[str, str] = {
    # core data/vis
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "bs4": "beautifulsoup4",
    "Crypto": "pycryptodome",
    "win32com": "pywin32",
    "OpenSSL": "pyOpenSSL",
    "telegram": "python-telegram-bot",
    "pymysql": "PyMySQL",
    "mysqlclient": "mysqlclient",
    "psycopg2": "psycopg2-binary",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "argcomplete": "argcomplete",
    # Aden’s stack (add your usual suspects)
    "feedparser": "feedparser",
    "watchdog": "watchdog",
    "streamlit": "streamlit",
    "plotly": "plotly",
    "altair": "altair",
    "lxml": "lxml",
    "requests": "requests",
    "python_dotenv": "python-dotenv",  # just in case someone imports like this
}

DEFAULT_IGNORE_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__", ".vscode",
    "build", "dist", "node_modules", ".mypy_cache", ".pytest_cache",
}

PY_EXTS = {".py"}
NB_EXTS = {".ipynb"}

# Best-effort stdlib detection (Python 3.10+)
try:
    import sys as _sys
    STDLIB_NAMES = set(getattr(_sys, "stdlib_module_names", ()))
except Exception:
    STDLIB_NAMES = set()

# Some obvious built-ins not always in stdlib listing (defensive)
BUILTIN_LIKE = {
    "os", "sys", "re", "json", "math", "time", "datetime", "pathlib", "subprocess",
    "typing", "itertools", "functools", "collections", "random", "statistics",
    "hashlib", "hmac", "logging", "argparse", "shutil", "glob", "enum",
    "threading", "asyncio", "concurrent", "dataclasses", "http", "urllib",
    "zipfile", "tarfile", "csv", "getpass", "platform", "traceback",
    "base64", "email", "inspect", "importlib", "site", "copy", "pprint",
}

def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_dir():
            if p.name in DEFAULT_IGNORE_DIRS:
                # Skip whole subtree
                dirs = []  # just for clarity
            continue
        if p.suffix in PY_EXTS or p.suffix in NB_EXTS:
            # skip ignored dir components in the path
            if any(part in DEFAULT_IGNORE_DIRS for part in p.parts):
                continue
            yield p

def extract_imports_from_code(code: str) -> Set[str]:
    """Return top-level imported module names from Python code using AST."""
    mods: Set[str] = set()
    try:
        tree = ast.parse(code)
    except Exception:
        return mods

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top:
                    mods.add(top)
        elif isinstance(node, ast.ImportFrom):
            # Ignore relative imports (level > 0)
            if node.level and node.level > 0:
                continue
            if node.module:
                top = node.module.split(".")[0]
                if top:
                    mods.add(top)
    # Also catch simple importlib.import_module("pkg") patterns
    for m in re.findall(r"importlib\.import_module\(\s*['\"]([a-zA-Z0-9_.-]+)['\"]\s*\)", code):
        top = m.split(".")[0]
        if top:
            mods.add(top)
    return mods

def extract_imports_from_notebook(nb_path: Path) -> Set[str]:
    mods: Set[str] = set()
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception:
        return mods
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            mods |= extract_imports_from_code(src)
    return mods

def is_third_party(mod: str) -> bool:
    # Treat anything not clearly stdlib/builtin as third-party
    if mod in BUILTIN_LIKE:
        return False
    if mod in STDLIB_NAMES:
        return False
    return True

def module_to_pip_name(mod: str) -> str:
    return MODULE_TO_PIP.get(mod, mod)

def is_module_installed(mod: str) -> bool:
    """Check if the importable module exists in current interpreter."""
    try:
        import importlib.util
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False

def install_package(pip_name: str) -> int:
    print(f"  -> pip install {pip_name}")
    try:
        return subprocess.call([sys.executable, "-m", "pip", "install", pip_name])
    except Exception as e:
        print(f"     ! install failed for {pip_name}: {e}")
        return 1

def main():
    ap = argparse.ArgumentParser(description="Scan project for imports and install missing pip packages.")
    ap.add_argument("--root", default=".", help="Project root to scan (default: current directory).")
    ap.add_argument("--apply", action="store_true", help="Install missing packages into the active environment.")
    ap.add_argument("--export", default=None, help="Write pip freeze to this path after install (e.g. requirements.txt).")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    print(f"[scan] root = {root}")

    all_imports: Set[str] = set()
    for p in iter_files(root):
        try:
            if p.suffix in PY_EXTS:
                code = p.read_text(encoding="utf-8", errors="ignore")
                mods = extract_imports_from_code(code)
            else:
                mods = extract_imports_from_notebook(p)
            if mods:
                all_imports |= mods
        except Exception as e:
            print(f"[warn] could not read {p}: {e}")

    # Keep only top-level third-party names
    third_party = sorted(m for m in all_imports if is_third_party(m))

    print("\n[imports] detected third-party candidates:")
    for m in third_party:
        print("  -", m)

    # Map to pip names and test installation status
    mapping: List[Tuple[str, str, bool]] = []
    for mod in third_party:
        pip_name = module_to_pip_name(mod)
        installed = is_module_installed(mod)
        mapping.append((mod, pip_name, installed))

    print("\n[status] module -> pip, installed?")
    missing: List[str] = []
    for mod, pip_name, installed in mapping:
        flag = "OK " if installed else "MISS"
        print(f"  [{flag}] {mod:20s} -> {pip_name}")
        if not installed:
            missing.append(pip_name)

    if not missing:
        print("\n[done] No missing packages detected 🎉")
    else:
        print("\n[need] Missing pip packages:")
        for pip_name in missing:
            print("  -", pip_name)

        if args.apply:
            print("\n[apply] Installing missing packages into current environment...")
            failed = []
            for pip_name in missing:
                rc = install_package(pip_name)
                if rc != 0:
                    failed.append(pip_name)
            if failed:
                print("\n[fail] These packages failed to install:")
                for p in failed:
                    print("  -", p)
            else:
                print("\n[ok] All missing packages installed.")

    if args.export:
        # Export full, pinned env (after installs) for reproducibility
        try:
            print(f"\n[export] Writing pip freeze to {args.export}")
            with open(args.export, "w", encoding="utf-8") as f:
                out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
                f.write(out)
            print("[export] Done.")
        except Exception as e:
            print(f"[export] Failed to write freeze: {e}")

if __name__ == "__main__":
    main()
