#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repo quality gate.

1. English-named docs (.md/.html) must contain zero CJK characters.
2. Chinese-named docs (.zh-CN suffix or CJK filename) must contain Chinese.
3. Every asset referenced by README/index must exist.
4. Bilingual README + index must all be present.

Exits 0 on success, 1 on any failure.
"""
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
cjk = re.compile(r"[\u4e00-\u9fff]")
ASSET_RE = re.compile(r"assets/[\w\-.]+\.(?:png|jpg|jpeg|gif|svg)")
REQUIRED = ["README.md", "README.zh-CN.md", "index.html", "index.zh-CN.html"]


def count_cjk(s: str) -> int:
    return len(cjk.findall(s))


def is_zh_name(name: str) -> bool:
    b = Path(name).name
    return bool(cjk.search(b)) or ".zh-CN" in b


def main() -> int:
    errors = []

    # --- 1/2. document language consistency -------------------------------
    # Scan only git-tracked .md/.html (untracked local docs like OPERATIONS.md are
    # intentionally ignored) so the gate reflects exactly what gets published.
    r = subprocess.run(
        ["git", "ls-files", "*.md", "*.html"],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8",
    )
    doc_files = [root / f for f in r.stdout.splitlines() if f]
    seen = set()
    for fp in doc_files:
        p = fp.resolve()
        if p in seen or fp.name.startswith("."):
            continue
        seen.add(p)
        try:
            text = fp.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        n = count_cjk(text)
        if is_zh_name(fp.name):
            if n == 0:
                errors.append(f"[{fp.name}] Chinese-named doc has no Chinese content")
        else:
            if n > 0:
                errors.append(f"[{fp.name}] English-named doc contains {n} CJK chars")

    # --- 3. asset references exist ----------------------------------------
    for fname in REQUIRED:
        fp = root / fname
        if not fp.is_file():
            errors.append(f"missing {fname}")
            continue
        text = fp.read_text(encoding="utf-8", errors="replace")
        for ref in ASSET_RE.findall(text):
            if not (root / ref).is_file():
                errors.append(f"missing asset referenced in {fname}: {ref}")

    # --- 4. bilingual files present ---------------------------------------
    for fname in REQUIRED:
        if not (root / fname).is_file():
            errors.append(f"missing {fname}")

    if errors:
        print("QUALITY FAIL")
        for e in sorted(set(errors)):
            print(" -", e)
        return 1
    print("QUALITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
