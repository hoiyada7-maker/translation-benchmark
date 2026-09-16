#!/usr/bin/env python3
"""Translate an SRT to Korean with argos-translate (offline en->ko).

Usage: python translate_argos.py <src.srt> <out.srt> [logfile]
"""
import sys, time
from argostranslate import package, translate

def parse_srt(path):
    raw = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    out = []
    for ch in raw.strip().split("\n\n"):
        l = ch.split("\n")
        if len(l) < 2 or not l[0].strip().isdigit():
            continue
        out.append((l[0].strip(), l[1].strip(), " ".join(x.strip() for x in l[2:]).strip()))
    return out

def ensure_en_ko():
    installed = translate.get_installed_languages()
    def find():
        try:
            en = next(l for l in installed if l.code == "en")
            ko = next(l for l in installed if l.code == "ko")
            return en.get_translation(ko)
        except StopIteration:
            return None
    tr = find()
    if tr:
        return tr
    package.update_package_index()
    avail = package.get_available_packages()
    pkg = next(p for p in avail if p.from_code == "en" and p.to_code == "ko")
    package.install_from_path(pkg.download())
    installed[:] = translate.get_installed_languages()
    en = next(l for l in translate.get_installed_languages() if l.code == "en")
    ko = next(l for l in translate.get_installed_languages() if l.code == "ko")
    return en.get_translation(ko)

def main():
    src, out = sys.argv[1], sys.argv[2]
    logfile = sys.argv[3] if len(sys.argv) > 3 else None
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        if logfile:
            open(logfile, "a", encoding="utf-8").write(line + "\n")
    tr = ensure_en_ko()
    blocks = parse_srt(src)
    n = len(blocks)
    log(f"argos: {n} blocks -> {out}")
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        for i, (idx, ts, en) in enumerate(blocks):
            ko = tr.translate(en) if en.strip() else ""
            f.write(f"{idx}\n{ts}\n{ko}\n\n")
            if (i + 1) % 100 == 0 or i == n - 1:
                log(f"  {i+1}/{n}")
    log(f"argos: DONE in {(time.time()-t0)/60:.1f}min")

if __name__ == "__main__":
    main()
