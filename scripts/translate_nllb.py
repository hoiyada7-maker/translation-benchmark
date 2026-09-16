#!/usr/bin/env python3
"""Translate an SRT to Korean with NLLB-200 (distilled-600M) on GPU, batched.

Usage: python translate_nllb.py <src.srt> <out.srt> [logfile] [batch]
"""
import sys, time, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = "facebook/nllb-200-distilled-600M"

def parse_srt(path):
    raw = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    out = []
    for ch in raw.strip().split("\n\n"):
        l = ch.split("\n")
        if len(l) < 2 or not l[0].strip().isdigit():
            continue
        out.append([l[0].strip(), l[1].strip(), " ".join(x.strip() for x in l[2:]).strip()])
    return out

def main():
    src, out = sys.argv[1], sys.argv[2]
    logfile = sys.argv[3] if len(sys.argv) > 3 else None
    bs = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
        if logfile: open(logfile, "a", encoding="utf-8").write(line + "\n")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.src_lang = "eng_Latn"
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(dev).eval()
    bos = tok.convert_tokens_to_ids("kor_Hang")
    blocks = parse_srt(src)
    n = len(blocks)
    log(f"NLLB: {n} blocks on {dev} -> {out}")
    t0 = time.time()
    for s in range(0, n, bs):
        grp = blocks[s:s+bs]
        idx_nonempty = [i for i, b in enumerate(grp) if b[2].strip()]
        texts = [grp[i][2] for i in idx_nonempty]
        kos = [""] * len(grp)
        if texts:
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
            with torch.no_grad():
                g = model.generate(**enc, forced_bos_token_id=bos, max_length=512, num_beams=1)
            dec = tok.batch_decode(g, skip_special_tokens=True)
            for j, i in enumerate(idx_nonempty):
                kos[i] = dec[j].strip()
        for i, b in enumerate(grp):
            b.append(kos[i])
        with open(out, "w", encoding="utf-8") as f:
            for b in blocks:
                f.write(f"{b[0]}\n{b[1]}\n{b[3] if len(b)>3 else b[2]}\n\n")
        done = s + len(grp)
        el = time.time() - t0
        log(f"  {done}/{n} ({el/done:.2f}s/blk)")
    log(f"NLLB: DONE in {(time.time()-t0)/60:.1f}min")

if __name__ == "__main__":
    main()
