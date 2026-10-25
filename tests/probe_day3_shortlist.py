"""Verifier probe 3: shortlist top-2 co bo sot board dung khong?

BUG-DAY3-02 fix dung fingerprint gradient de shortlist top-2 template.
Neu board dung xep hang >=3 thi board-ID chac chan sai. Probe nay do
top-1/top-2/top-3 tren 10 template + 3 muc perturbation camera.

Chay: <venv>/python.exe tests/probe_day3_shortlist.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(__file__).resolve().parent.parent  # PORTABLE: tuong doi theo project root
sys.path.insert(0, str(PROJ))

from configs.resource_paths import LOCAL_TEMPLATES, NEW_PROJECT_DIR, assert_writable_ok  # noqa: E402

import src.reference.board_id_and_align as ba  # noqa: E402


def read_gray(p):
    d = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(d, cv2.IMREAD_GRAYSCALE)


def perturb(gray, rng):
    """Mo phong camera: dich +-25px, xoay +-2deg, gain 0.85-1.15, bias +-12."""
    h, w = gray.shape
    ang = rng.uniform(-2.0, 2.0)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    m[0, 2] += rng.uniform(-25, 25)
    m[1, 2] += rng.uniform(-25, 25)
    out = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    gain = rng.uniform(0.85, 1.15)
    bias = rng.uniform(-12, 12)
    out = np.clip(out.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    return out


def main():
    cache = ba.load_cache()
    ids = cache["board_ids"]
    fps = cache["fingerprints"]
    rows = []
    rng = np.random.default_rng(20261023)
    for bid in ids:
        g = read_gray(LOCAL_TEMPLATES / f"{bid}.JPG")
        variants = [("original", g)]
        for k in range(5):
            variants.append((f"camera{k}", perturb(g, rng)))
        for tag, img in variants:
            cand = ba.shortlist_boards(img, fps, ids, top_k=len(ids))
            ranks = [b for b, _ in cand]
            rank = ranks.index(bid) + 1 if bid in ranks else len(ids) + 1
            rows.append({"board": bid, "variant": tag, "rank": rank,
                         "top3": [c[0] for c in cand[:3]],
                         "top5": [c[0] for c in cand[:5]],
                         "cos_true": round(next(c[1] for c in cand if c[0] == bid), 4)})
            print(f"{bid} {tag:10s} rank={rank:2d} top3={rows[-1]['top3']} "
                  f"cos={rows[-1]['cos_true']}")
    n = len(rows)
    s = {"n_cases": n,
         "top1": sum(r["rank"] == 1 for r in rows),
         "top2": sum(r["rank"] <= 2 for r in rows),
         "top3": sum(r["rank"] <= 3 for r in rows),
         "top2_failures": [r for r in rows if r["rank"] > 2],
         "note": "10 template x (goc + 5 camera variants); SHORTLIST_TOP_K=2",
         "rows": rows}
    out = assert_writable_ok(NEW_PROJECT_DIR / "results" / "probe_day3_shortlist.json")
    out.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[probe] top1 {s['top1']}/{n}  top2 {s['top2']}/{n}  top3 {s['top3']}/{n}")
    if s["top2_failures"]:
        print("[probe] TOP-2 FAILURES:", json.dumps(s["top2_failures"], indent=1))
    print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()
