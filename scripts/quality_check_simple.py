# -*- coding: utf-8 -*-
"""仓库质量门禁 —— 基础可读版（quality_check_simple.py）
=====================================================
这个文件做的事，和原版 quality_check.py 完全一样：
  1. 英文名的文档（.md / .html）里不能出现任何中文字符
  2. 中文名的文档（文件名带中文 或 带 .zh-CN）里必须有中文
  3. README / index 里引用的 assets/xxx.png 等图片资源必须真实存在
  4. 双语的 README.md / README.zh-CN.md / index.html / index.zh-CN.html 都得齐

全部通过就打印 "QUALITY OK" 并退出 0；有任何一条不过就打印
"QUALITY FAIL" 和问题清单，退出 1。

区别只是：本文件故意用"最朴素的写法"（for 循环、if 判断、普通函数），
把原版用正则表达式做的"找中文字"和"找图片引用"改用普通循环实现，
每一步都写清楚了在做什么、为什么。

运行方式（和原版一样，可不带参数，默认检查当前目录）：
    python scripts/quality_check_simple.py [项目根目录]
"""
import subprocess
import sys
from pathlib import Path

# ------------------------------------------------------------------
# 0. 配置
# ------------------------------------------------------------------
# 命令行可以传一个目录当要检查的项目根，不传就用当前目录
root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

# 双语文件必须四个都在
REQUIRED = ["README.md", "README.zh-CN.md", "index.html", "index.zh-CN.html"]

# 图片扩展名白名单（和原版正则里的 png/jpg/jpeg/gif/svg 一致）
ASSET_EXTS = [".png", ".jpg", ".jpeg", ".gif", ".svg"]


# ------------------------------------------------------------------
# 1. 两个小工具：数中文字 / 找图片引用
# ------------------------------------------------------------------
def is_cjk(ch):
    """判断一个字符是不是中日韩统一表意文字（Unicode 区间 U+4E00 ~ U+9FFF）。"""
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF


def count_cjk(text):
    """数一段文本里有多少个中文字符（等价于原版 [\\u4e00-\\u9fff] 正则匹配）。"""
    n = 0
    for ch in text:
        if is_cjk(ch):
            n += 1
    return n


def has_cjk(name):
    """判断文件名里有没有中文字符。"""
    for ch in name:
        if is_cjk(ch):
            return True
    return False


def is_zh_name(filename):
    """判断一个文件名算不算"中文名文档"：要么文件名带中文，要么带 .zh-CN。"""
    base = Path(filename).name
    if has_cjk(base):
        return True
    if ".zh-CN" in base:
        return True
    return False


def find_asset_refs(text):
    """找出文本里所有 assets/xxx.png 这类图片引用（等价于原版正则）。

    原版正则：assets/[\\w\\-.]+\\.(?:png|jpg|jpeg|gif|svg)
    也就是：以 "assets/" 开头，后面跟一串字母/数字/下划线/连字符/点，
    最后以 .png / .jpg / .jpeg / .gif / .svg 结尾。
    这里用普通循环逐位置扫描实现。
    """
    refs = []
    n = len(text)
    i = 0
    while i < n:
        # 找 "assets/" 这个前缀
        if text[i:i + 7] == "assets/":
            # 从 "assets/" 后面开始，一直收字母/数字/下划线/连字符/点
            j = i + 7
            name_start = j
            while j < n:
                ch = text[j]
                if ch.isalnum() or ch == "_" or ch == "-" or ch == ".":
                    j += 1
                else:
                    break
            body = text[name_start:j]     # 收上来的文件名部分
            # 检查它是不是以某个图片扩展名结尾，并且前面至少有一个字符
            for ext in ASSET_EXTS:
                if body.endswith(ext) and len(body) > len(ext):
                    refs.append("assets/" + body)
                    break
            i = j                          # 从文件名结束处继续往后扫
        else:
            i += 1
    return refs


# ------------------------------------------------------------------
# 2. 主流程
# ------------------------------------------------------------------
def main():
    errors = []

    # ---- 2.1 / 2.2 文档语言一致性 ----
    # 只扫 git 跟踪的 .md / .html 文件（本地未跟踪的文档故意忽略），
    # 这样门禁反映的就是"真正会发布出去"的那部分内容。
    r = subprocess.run(
        ["git", "ls-files", "*.md", "*.html"],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8",
    )
    doc_files = [root / f for f in r.stdout.splitlines() if f]

    seen = set()
    for fp in doc_files:
        p = fp.resolve()
        # 同一个文件只检查一次；以 . 开头的隐藏文件跳过
        if p in seen or fp.name.startswith("."):
            continue
        seen.add(p)
        try:
            text = fp.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        n = count_cjk(text)
        if is_zh_name(fp.name):
            # 中文名的文档：里面一个中文字都没有 → 有问题
            if n == 0:
                errors.append(f"[{fp.name}] Chinese-named doc has no Chinese content")
        else:
            # 英文名的文档：出现了中文字符 → 有问题
            if n > 0:
                errors.append(f"[{fp.name}] English-named doc contains {n} CJK chars")

    # ---- 2.3 README / index 引用的图片资源必须存在 ----
    for fname in REQUIRED:
        fp = root / fname
        if not fp.is_file():
            errors.append(f"missing {fname}")
            continue
        text = fp.read_text(encoding="utf-8", errors="replace")
        for ref in find_asset_refs(text):
            if not (root / ref).is_file():
                errors.append(f"missing asset referenced in {fname}: {ref}")

    # ---- 2.4 双语文件都得齐 ----
    for fname in REQUIRED:
        if not (root / fname).is_file():
            errors.append(f"missing {fname}")

    # ---- 2.5 输出结论 ----
    if errors:
        print("QUALITY FAIL")
        for e in sorted(set(errors)):
            print(" -", e)
        return 1
    print("QUALITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
