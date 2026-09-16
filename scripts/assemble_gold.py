#!/usr/bin/env python3
"""Assemble the gold SRT from source structure + my Korean lines.

gold_ko file format: one block per line, `<idx>\t<korean>`. Order-independent
(mapped by idx). Missing idx falls back to the source English text.

Usage: python assemble_gold.py <src.srt> <gold_ko.txt> <out.srt>
"""
import sys, re

def parse_srt(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read().replace("\r\n", "\n")
    out = []
    for ch in raw.strip().split("\n\n"):
        l = ch.split("\n")
        if len(l) < 2 or not l[0].strip().isdigit():
            continue
        out.append((l[0].strip(), l[1].strip(), " ".join(x.strip() for x in l[2:]).strip()))
    return out

def main():
    src, ko_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    blocks = parse_srt(src)
    ko = {}
    with open(ko_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            m = re.match(r"(\d+)\t(.*)$", line)
            if m:
                ko[m.group(1)] = m.group(2).strip()
    missing = [idx for idx, _, _ in blocks if idx not in ko]
    with open(out, "w", encoding="utf-8") as f:
        for idx, ts, en in blocks:
            f.write(f"{idx}\n{ts}\n{ko.get(idx, en)}\n\n")
    print(f"assembled {len(blocks)} blocks -> {out}; missing={len(missing)}: {missing[:20]}")

if __name__ == "__main__":
    main()
