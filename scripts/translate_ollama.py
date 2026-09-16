#!/usr/bin/env python3
"""Translate an SRT file to Korean via a local Ollama model, in batches.

Keeps SRT structure identical (same index + timestamps), replaces only the text.
Batches N blocks per request (numbered) to amortize prompt overhead; falls back
to per-block translation for any block the batch response drops. Checkpoints
after every batch.

Usage: python translate_ollama.py <model> <src.srt> <out.srt> [logfile] [batch]
"""
import sys, re, json, urllib.request, time

OLLAMA = "http://localhost:11434/api/generate"
SYS = ("You are a professional English->Korean subtitle translator for a Siemens "
       "corporate town hall speech (speakers Scott, Tino, Frank, Stephanie; topics "
       "sales, SK Hynix, Samsung, Merck, org changes). Translate into natural, fluent "
       "Korean. Keep company/person/product proper nouns. If a segment is already "
       "Korean, keep it.")

def parse_srt(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read().replace("\r\n", "\n")
    blocks = []
    for chunk in raw.strip().split("\n\n"):
        lines = chunk.split("\n")
        if len(lines) < 2:
            continue
        blocks.append({"idx": lines[0].strip(), "ts": lines[1].strip(),
                       "text": " ".join(l.strip() for l in lines[2:]).strip()})
    return blocks

def call(model, prompt):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "think": False, "options": {"temperature": 0.2}}).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read())
    s = re.sub(r"<think>.*?</think>", "", resp.get("response", ""), flags=re.DOTALL)
    return s.strip()

def translate_one(model, cur, prev, nxt):
    if not cur.strip():
        return ""
    p = (SYS + "\nOutput ONLY the Korean translation of the CURRENT segment - "
         "no notes, no English, no quotes.\n\n"
         f"[previous]: {prev}\n[next]: {nxt}\n[CURRENT]: {cur}\n\n"
         "Korean:/no_think")
    out = call(model, p).strip().strip('"').strip()
    return re.sub(r"\s+", " ", out.replace("\n", " ")).strip()

def translate_batch(model, items):
    """items: list of (localnum, text) with non-empty text. Returns dict num->ko."""
    numbered = "\n".join(f"{n}. {t}" for n, t in items)
    p = (SYS + "\nTranslate each numbered English segment. Output EXACTLY one line "
         "per segment, same numbering, in the form `N. <Korean>`. No commentary, "
         "no blank lines, do not merge or renumber.\n\n" + numbered + "\n/no_think")
    out = call(model, p)
    res = {}
    for line in out.split("\n"):
        m = re.match(r"\s*(\d+)[.)]\s?(.*)$", line)
        if m:
            res[int(m.group(1))] = m.group(2).strip().strip('"').strip()
    return res

def write_srt(blocks, path):
    with open(path, "w", encoding="utf-8") as f:
        for b in blocks:
            f.write(f"{b['idx']}\n{b['ts']}\n{b.get('ko', b['text'])}\n\n")

def log(logfile, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def main():
    # per-block PARALLEL translation: 1 block = 1 request -> guaranteed alignment.
    # batching by numbered lists was dropped: models silently drop/merge lines,
    # shifting whole regions. Parallel requests keep speed without that risk.
    from concurrent.futures import ThreadPoolExecutor
    model, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
    logfile = sys.argv[4] if len(sys.argv) > 4 else None
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    blocks = parse_srt(src)
    n = len(blocks)
    log(logfile, f"{model}: {n} blocks, per-block x{workers} workers -> {out}")
    t0 = time.time()
    done = [0]
    def work(i):
        b = blocks[i]
        prev = blocks[i-1]["text"] if i > 0 else ""
        nxt = blocks[i+1]["text"] if i < n-1 else ""
        try:
            b["ko"] = translate_one(model, b["text"], prev, nxt)
        except Exception as e:
            b["ko"] = b["text"]
            log(logfile, f"  block {b['idx']} FAILED: {e}")
        done[0] += 1
        if done[0] % 30 == 0 or done[0] == n:
            write_srt(blocks, out)
            el = time.time() - t0
            log(logfile, f"  {done[0]}/{n} ({el/done[0]:.2f}s/blk, ~{el/done[0]*(n-done[0])/60:.1f}min left)")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, range(n)))
    write_srt(blocks, out)
    log(logfile, f"{model}: DONE in {(time.time()-t0)/60:.1f}min")

if __name__ == "__main__":
    main()
