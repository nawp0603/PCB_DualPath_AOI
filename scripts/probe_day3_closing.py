"""Ngay 3 FIX — probe "dong hinh thai" (morphological closing) thay ha min_area.

Van de: o camera perturbation, nhieu hat lam ban do seed bi VO VUN; mot loi
that 50x50px bi cat thanh nhieu manh <60px2 -> bi MIN_AREA_PX2=60 (plan
Phan 2.3 Buoc 4) loai. Ha nguong xuong 30px2 se bat lai duoc (probe truoc:
cam 93->98) nhung pha vo hop dong plan.

Thu phuong an giu nguyen MIN_AREA_PX2=60: dong hinh thai (cv2.morphologyEx
MORPH_CLOSE) tren ban do seed de noi manh vun TRUOC connectedComponents.
Kiem tra ca FAR tren GOOD camera (nguong open-set 3b Z>=3.5 & area>=100).

Khong ghi file nao; chi in stdout.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

_spec = importlib.util.spec_from_file_location(
    "probe_b", str(PROJ / "scripts" / "probe_day3_branchB.py"))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)
_spec2 = importlib.util.spec_from_file_location(
    "probe_v", str(PROJ / "scripts" / "probe_day3_validmask.py"))
pv = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(pv)

import src.reference.board_id_and_align as ba  # noqa: E402
import src.reference.tolerance_diff as td  # noqa: E402
from configs.resource_paths import LOCAL_TEMPLATES  # noqa: E402


def extract_closed(diff, min_area, sigma_floor, seed_k, top_k, close_px):
    d = np.maximum(np.asarray(diff, np.float32), 0.0)
    flat = d.ravel().astype(np.float64)
    bg_mean, bg_std = float(flat.mean()), float(flat.std())
    denom_seed = max(bg_std, float(td.FLOOR_SEED))
    seed_thr = bg_mean + float(seed_k) * denom_seed
    denom_z = max(bg_std, float(sigma_floor), 1e-6)
    bin_map = (d >= seed_thr).astype(np.uint8)
    if close_px and close_px > 1:
        bin_map = cv2.morphologyEx(
            bin_map, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(bin_map, 8)
    out = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        m = lab == i
        vals = d[m]
        mean_res = float(vals.mean())
        z = (mean_res - bg_mean) / denom_z
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        out.append({"centroid_tpl": [float(cents[i][0]), float(cents[i][1])],
                    "area": area, "bbox_tpl": [x, y, x + w, y + h], "z_diff": z})
    out.sort(key=lambda b: b["z_diff"], reverse=True)
    return out[:max(1, int(top_k))]


def prep(im, tpl_g, cache):
    r = ba.identify_and_align(im, cache, fast=True)
    if r["warped"].shape[:2] != tpl_g.shape:
        tpl_g = cv2.resize(tpl_g, (r["warped"].shape[1], r["warped"].shape[0]),
                           interpolation=cv2.INTER_AREA)
    valid = pv.valid_mask_from_h(im.shape, r["H"], r["warped"].shape[:2])
    wg = cv2.cvtColor(r["warped"], cv2.COLOR_BGR2GRAY)
    wg_f = np.where(valid, wg, tpl_g).astype(np.uint8)
    tf = tpl_g.astype(np.float32)
    corr = td.correct_lowfreq_gain(wg_f.astype(np.float32), tf)
    mask = np.maximum(td.structure_mask_from_template(tpl_g),
                      td.structure_mask_from_input(
                          np.clip(np.round(corr), 0, 255).astype(np.uint8)))
    diff_raw = td.tolerance_diff(corr, tf, td.TAU_PX, mask, None)
    vals = diff_raw[mask] if mask.any() else diff_raw.ravel()
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    tol = np.maximum(med + 3.0 * mad, 0.5).astype(np.float32)
    tol = np.where(mask, tol, np.ones_like(tf, np.float32))
    diff = td.tolerance_diff(corr, tf, td.TAU_PX, mask, tol)
    return r, diff


def main() -> None:
    cache = ba.load_cache()
    tpls = {b: probe.read_gray(LOCAL_TEMPLATES / f"{b}.JPG")
            for b in cache["board_ids"]}
    items = probe.collect()

    variants = [(k, sk) for k in (0, 3, 5, 7) for sk in (1.0, 1.25, 1.5)]
    score = {i: {"std": 0, "cam": 0, "npos": 0, "far": 0, "blob": 0}
             for i in range(len(variants))}

    for idx, it in enumerate(items):
        img = probe.read_bgr(Path(it["path"]))
        gt0 = [list(b) for _, b in it["gt"]]
        img_c, gt_c = probe.perturb(img, gt0, 20261023 + 101 + idx)
        for tag, im, gt in (("std", img, gt0), ("cam", img_c, gt_c)):
            r, diff = prep(im, tpls[it["board"]], cache)
            for vi, (k, sk) in enumerate(variants):
                blobs = extract_closed(diff, td.MIN_AREA_PX2, td.SIGMA_FLOOR,
                                       sk, td.TOP_K, k)
                score[vi][tag] += pv.recall_of(blobs, r["H_inv"], gt)
                score[vi]["npos"] += len(gt)
                score[vi]["blob"] += len(blobs)

    for bid in ("01", "04"):
        img = probe.read_bgr(LOCAL_TEMPLATES / f"{bid}.JPG")
        img_c, _ = probe.perturb(img, [], 20261023 + 999)
        _, diff = prep(img_c, tpls[bid], cache)
        for vi, (k, sk) in enumerate(variants):
            blobs = extract_closed(diff, td.MIN_AREA_PX2, td.SIGMA_FLOOR,
                                   sk, td.TOP_K, k)
            score[vi]["far"] += sum(1 for b in blobs
                                    if b["z_diff"] >= 3.5 and b["area"] >= 100)

    print(f"{'close':>6}{'seedK':>7}{'std':>6}{'cam':>6}{'npos':>6}"
          f"{'FARgood':>9}{'blob':>6}")
    for vi, (k, sk) in enumerate(variants):
        s = score[vi]
        print(f"{k:>6}{sk:>7}{s['std']:>6}{s['cam']:>6}{s['npos']:>6}"
              f"{s['far']:>9}{s['blob']:>6}")


if __name__ == "__main__":
    main()
