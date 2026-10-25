"""Verifier probe 4: shortlist top-2 co lam SAI board-ID end-to-end khong?

Tai hien truong hop probe_day3_shortlist phat hien (board 06 rank 4) va
do truc tiep identify_and_align(fast=True) tren tap perturbation rong hon
(+-25px, +-2deg, gain 0.85-1.15, bias +-12, vignette, noise).

Chay: <venv>/python.exe tests/probe_day3_boardid_stress.py
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


def perturb(gray, rng, vignette=True, noise=True):
    h, w = gray.shape
    ang = rng.uniform(-2.0, 2.0)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    m[0, 2] += rng.uniform(-25, 25)
    m[1, 2] += rng.uniform(-25, 25)
    out = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    out *= rng.uniform(0.85, 1.15)
    out += rng.uniform(-12, 12)
    if vignette:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        out *= (1.0 - 0.25 * np.clip(r, 0, 1) ** 2)
    if noise:
        out += rng.normal(0, 3.0, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    cache = ba.load_cache()
    ids = cache["board_ids"]
    rows = []
    for bid in ids:
        g = read_gray(LOCAL_TEMPLATES / f"{bid}.JPG")
        for k in range(8):
            rng = np.random.default_rng(hash((bid, k)) % (2 ** 31))
            img = perturb(g, rng)
            cand = ba.shortlist_boards(img, cache["fingerprints"], ids,
                                       top_k=len(ids))
            ranks = [b for b, _ in cand]
            rank = ranks.index(bid) + 1
            img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            try:
                r = ba.identify_and_align(img3, cache, fast=True)
                pred, inl, err = r["board_id"], r["inliers"], None
            except Exception as e:
                pred, inl, err = None, 0, f"{type(e).__name__}: {e}"
            ok = pred == bid
            rows.append({"board": bid, "k": k, "rank": rank, "pred": pred,
                         "inliers": inl, "ok": bool(ok), "error": err,
                         "top5": ranks[:5]})
            if not ok or rank > 2:
                print(f"!! {bid} k={k} rank={rank} pred={pred} inl={inl} "
                      f"top5={ranks[:5]} err={err}")
    n = len(rows)
    fails = [r for r in rows if not r["ok"]]
    rank_miss = [r for r in rows if r["rank"] > ba.SHORTLIST_TOP_K]
    s = {"n_cases": n, "shortlist_top_k": ba.SHORTLIST_TOP_K,
         "board_id_ok": n - len(fails), "acc": round((n - len(fails)) / n, 4),
         "shortlist_rank_gt_k": len(rank_miss),
         "failures": fails,
         "note": "10 template x 8 camera variants (shift+-25px, rot+-2deg, "
                 "gain 0.85-1.15, bias+-12, vignette, noise sigma=3)",
         "rows": rows}
    out = assert_writable_ok(NEW_PROJECT_DIR / "results" /
                             "probe_day3_boardid_stress.json")
    out.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[probe] board-ID {s['board_id_ok']}/{n} = {s['acc']} | "
          f"shortlist rank>{ba.SHORTLIST_TOP_K}: {len(rank_miss)}")
    for f in fails:
        print(f"  FAIL {f['board']} k={f['k']} rank={f['rank']} pred={f['pred']} "
              f"top5={f['top5']}")
    print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()
