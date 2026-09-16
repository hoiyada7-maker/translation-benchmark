#!/usr/bin/env python3
"""Per-block parallel Ollama translation with a MINIMAL prompt.

For translation-specialized models (e.g. translategemma) that echo complex
instruction prompts instead of following them. 1 block = 1 request -> aligned.

Usage: python translate_simple.py <model> <src.srt> <out.srt> [logfile] [workers]
"""
import sys, re, json, urllib.request, time
from concurrent.futures import ThreadPoolExecutor

OLLAMA = "http://localhost:11434/api/generate"

def parse_srt(path):
    raw = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    out = []
    for ch in raw.strip().split("\n\n"):
        l = ch.split("\n")
        if len(l) >= 2 and l[0].strip().isdigit():
            out.append({"idx": l[0].strip(), "ts": l[1].strip(),
                        "text": " ".join(x.strip() for x in l[2:]).strip()})
    return out

def translate(model, text):
    if not text.strip():
        return ""
    p = ("Translate the following English text to Korean. "
         "Output only the Korean translation, nothing else.\n\n" + text)
    body = json.dumps({"model": model, "prompt": p, "stream": False,
                       "options": {"temperature": 0.2}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        s = json.loads(r.read()).get("response", "")
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    return re.sub(r"\s+", " ", s.replace("\n", " ")).strip().strip('"').strip()

def write_srt(blocks, path):
    with open(path, "w", encoding="utf-8") as f:
        for b in blocks:
            f.write(f"{b['idx']}\n{b['ts']}\n{b.get('ko', b['text'])}\n\n")

def main():
    model, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
    logfile = sys.argv[4] if len(sys.argv) > 4 else None
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    blocks = parse_srt(src); n = len(blocks)
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
        if logfile: open(logfile, "a", encoding="utf-8").write(line + "\n")
    log(f"{model}: {n} blocks, simple per-block x{workers} -> {out}")
    t0 = time.time(); done = [0]
    def work(i):
        b = blocks[i]
        try: b["ko"] = translate(model, b["text"])
        except Exception as e: b["ko"] = b["text"]; log(f"  block {b['idx']} FAILED: {e}")
        done[0] += 1
        if done[0] % 30 == 0 or done[0] == n:
            write_srt(blocks, out)
            el = time.time() - t0
            log(f"  {done[0]}/{n} ({el/done[0]:.2f}s/blk, ~{el/done[0]*(n-done[0])/60:.1f}min left)")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, range(n)))
    write_srt(blocks, out)
    log(f"{model}: DONE in {(time.time()-t0)/60:.1f}min")

if __name__ == "__main__":
    main()
