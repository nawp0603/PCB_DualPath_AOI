"""Verifier probe 5: shortlist top-2 -> board-ID, seed SO CO DINH (tat dinh).

Probe truoc dung hash((bid,k)) — Python randomize hash chuoi theo process nen
KHONG tai lap duoc. Probe nay dung seed so duoc tinh tu (chi so board, k).

Chay: <venv>/python.exe tests/probe_day3_boardid_deterministic.py
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


def perturb(gray, rng, shift=25.0, vignette=True, noise=3.0):
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-2.0, 2.0), 1.0)
    m[0, 2] += rng.uniform(-shift, shift)
    m[1, 2] += rng.uniform(-shift, shift)
    out = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    out *= rng.uniform(0.85, 1.15)
    out += rng.uniform(-12, 12)
    if vignette:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        out *= (1.0 - 0.25 * np.clip(r, 0, 1) ** 2)
    if noise:
        out += rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def run(shift: float, n_per_board: int):
    cache = ba.load_cache()
    ids = cache["board_ids"]
    rows = []
    for bi, bid in enumerate(ids):
        g = read_gray(LOCAL_TEMPLATES / f"{bid}.JPG")
        for k in range(n_per_board):
            rng = np.random.default_rng(20261023 + bi * 1000 + k)
            img = perturb(g, rng, shift=shift)
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
            rows.append({"board": bid, "k": k, "rank": rank, "pred": pred,
                         "inliers": inl, "ok": pred == bid, "error": err,
                         "top4": ranks[:4]})
    return rows


def main():
    out = {"shift_px": {}, "note": "seed co dinh 20261023+bi*1000+k; "
                                   "shift +-25px, rot +-2deg, gain 0.85-1.15, "
                                   "bias +-12, vignette, noise sigma=3"}
    for shift in (25.0, 50.0):
        rows = run(shift, 8)
        n = len(rows)
        miss = [r for r in rows if r["rank"] > ba.SHORTLIST_TOP_K]
        fail = [r for r in rows if not r["ok"]]
        out["shift_px"][str(int(shift))] = {
            "n": n, "board_id_ok": n - len(fail),
            "acc": round((n - len(fail)) / n, 4),
            "shortlist_miss": len(miss), "failures": fail,
            "shortlist_misses": miss}
        print(f"shift +-{int(shift)}px: board-ID {n - len(fail)}/{n}, "
              f"shortlist rank>{ba.SHORTLIST_TOP_K}: {len(miss)}")
        for f in fail:
            print(f"   FAIL {f['board']} k={f['k']} rank={f['rank']} "
                  f"pred={f['pred']} top4={f['top4']}")
        for m in miss:
            print(f"   MISS {m['board']} k={m['k']} rank={m['rank']} "
                  f"pred={m['pred']} top4={m['top4']}")
    p = assert_writable_ok(NEW_PROJECT_DIR / "results" /
                           "probe_day3_boardid_deterministic.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[probe] wrote {p}")


if __name__ == "__main__":
    main()
