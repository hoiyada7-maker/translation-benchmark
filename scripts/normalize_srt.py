#!/usr/bin/env python3
"""Normalize an engine SRT to the source's exact block set (idx + timestamp).

Handles engine outputs that leaked blank lines / multi-line text: a block starts
at a bare-integer line immediately followed by an SRT timestamp line; everything
until the next such start is the block text (joined to one line). Re-emits using
the SOURCE idx+timestamp so every engine file has exactly the source's blocks.

Usage: python normalize_srt.py <src.srt> <engine.srt> [out.srt]
       (out defaults to overwriting engine.srt)
"""
import sys, re

TS = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}\s*-->")

def parse_source(path):
    raw = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    out = []
    for ch in raw.strip().split("\n\n"):
        l = ch.split("\n")
        if len(l) >= 2 and l[0].strip().isdigit():
            out.append((l[0].strip(), l[1].strip()))
    return out

def parse_engine(path):
    lines = open(path, encoding="utf-8").read().replace("\r\n", "\n").split("\n")
    blocks = {}
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip().isdigit() and i + 1 < n and TS.match(lines[i+1].strip()):
            idx = lines[i].strip()
            i += 2
            buf = []
            while i < n and not (lines[i].strip().isdigit() and i + 1 < n and TS.match(lines[i+1].strip())):
                buf.append(lines[i])
                i += 1
            text = " ".join(x.strip() for x in buf if x.strip())
            blocks[idx] = re.sub(r"\s+", " ", text).strip()
        else:
            i += 1
    return blocks

def main():
    src, eng = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else eng
    source = parse_source(src)
    ko = parse_engine(eng)
    missing = [idx for idx, _ in source if idx not in ko]
    with open(out, "w", encoding="utf-8") as f:
        for idx, ts in source:
            f.write(f"{idx}\n{ts}\n{ko.get(idx, '')}\n\n")
    print(f"{eng}: {len(source)} blocks, missing={len(missing)} {missing[:10]}")

if __name__ == "__main__":
    main()
