"""Ngay 3 FIX — probe bien the recall (CPU-only, khong GPU).

Da xac nhan (probe_day3_validmask.py): vung canvas Template nam NGOAI anh dau
vao bi BORDER_REPLICATE lap day -> 59% blob camera la gia. Lap vung do bang
noi dung Golden Template dua camera recall 77 -> 93/105.

Probe nay tim phan con thieu: do la BAN CHINH XAC (centroid blob roi ngoai GT
box du residual trong GT rat cao, resmax~27-29). Thu cac bien the:
  * centroid nhi phan (hien tai) vs centroid trong so residual
  * seed_k, min_area, tau
va do ca FAR tren 2 anh GOOD (camera) de khong "chinh so ao".

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


def blobs_variant(diff, min_area, sigma_floor, seed_k, top_k,
                  weighted: bool):
    """extract_blobs + tuy chon centroid trong so residual."""
    d = np.maximum(np.asarray(diff, np.float32), 0.0)
    flat = d.ravel().astype(np.float64)
    bg_mean, bg_std = float(flat.mean()), float(flat.std())
    denom_seed = max(bg_std, float(td.FLOOR_SEED))
    seed_thr = bg_mean + float(seed_k) * denom_seed
    denom_z = max(bg_std, float(sigma_floor), 1e-6)
    bin_map = (d >= seed_thr).astype(np.uint8)
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
        if weighted and float(vals.sum()) > 0:
            ys, xs = np.nonzero(m)
            ws = d[m].astype(np.float64)
            cx = float((xs * ws).sum() / ws.sum())
            cy = float((ys * ws).sum() / ws.sum())
        else:
            cx, cy = float(cents[i][0]), float(cents[i][1])
        out.append({"centroid_tpl": [cx, cy], "area": area,
                    "bbox_tpl": [x, y, x + w, y + h], "z_diff": z})
    out.sort(key=lambda b: b["z_diff"], reverse=True)
    return out[:max(1, int(top_k))]


def main() -> None:
    cache = ba.load_cache()
    tpls = {b: probe.read_gray(LOCAL_TEMPLATES / f"{b}.JPG")
            for b in cache["board_ids"]}
    items = probe.collect()
    goods = [("01", LOCAL_TEMPLATES / "01.JPG"),
             ("04", LOCAL_TEMPLATES / "04.JPG")]

    variants = []
    for weighted in (False, True):
        for seed_k in (1.0, 1.25, 1.5, 2.0):
            for min_area in (30, 60):
                variants.append(dict(weighted=weighted, seed_k=seed_k,
                                     min_area=min_area, tau=3))
    # them vai bien the tau
    for tau in (2, 4):
        variants.append(dict(weighted=True, seed_k=1.25, min_area=60, tau=tau))

    score = {i: {"std": 0, "cam": 0, "npos": 0, "far": 0, "nblob": 0}
             for i in range(len(variants))}

    for idx, it in enumerate(items):
        img = probe.read_bgr(Path(it["path"]))
        gt0 = [list(b) for _, b in it["gt"]]
        img_c, gt_c = probe.perturb(img, gt0, 20261023 + 101 + idx)
        for tag, im, gt in (("std", img, gt0), ("cam", img_c, gt_c)):
            r = ba.identify_and_align(im, cache, fast=True)
            tpl_g = tpls[r["board_id"]]
            if r["warped"].shape[:2] != tpl_g.shape:
                tpl_g = cv2.resize(tpl_g, (r["warped"].shape[1],
                                           r["warped"].shape[0]),
                                   interpolation=cv2.INTER_AREA)
            valid = pv.valid_mask_from_h(im.shape, r["H"], r["warped"].shape[:2])
            wg = cv2.cvtColor(r["warped"], cv2.COLOR_BGR2GRAY)
            wg_f = np.where(valid, wg, tpl_g).astype(np.uint8)
            tf = tpl_g.astype(np.float32)
            corr = td.correct_lowfreq_gain(wg_f.astype(np.float32), tf)
            mask_t = td.structure_mask_from_template(tpl_g)
            mask_i = td.structure_mask_from_input(
                np.clip(np.round(corr), 0, 255).astype(np.uint8))
            mask = np.maximum(mask_t, mask_i)
            for vi, v in enumerate(variants):
                diff_raw = td.tolerance_diff(corr, tf, v["tau"], mask, None)
                vals = diff_raw[mask] if mask.any() else diff_raw.ravel()
                med = float(np.median(vals))
                mad = float(np.median(np.abs(vals - med)))
                tol = np.maximum(med + 3.0 * mad, 0.5).astype(np.float32)
                tol = np.where(mask, tol, np.ones_like(tf, np.float32))
                diff = td.tolerance_diff(corr, tf, v["tau"], mask, tol)
                blobs = blobs_variant(diff, v["min_area"], td.SIGMA_FLOOR,
                                      v["seed_k"], td.TOP_K, v["weighted"])
                score[vi][tag] += pv.recall_of(blobs, r["H_inv"], gt)
                score[vi]["npos"] += len(gt)
                score[vi]["nblob"] += len(blobs)

    # FAR tren GOOD camera (nguong open-set 3b: Z>=3.5 AND area>=100)
    for bid, tp in goods:
        tpl_g = tpls[bid]
        img = probe.read_bgr(tp)
        img_c, _ = probe.perturb(img, [], 20261023 + 999)
        r = ba.identify_and_align(img_c, cache, fast=True)
        valid = pv.valid_mask_from_h(img_c.shape, r["H"], r["warped"].shape[:2])
        wg = cv2.cvtColor(r["warped"], cv2.COLOR_BGR2GRAY)
        wg_f = np.where(valid, wg, tpl_g).astype(np.uint8)
        tf = tpl_g.astype(np.float32)
        corr = td.correct_lowfreq_gain(wg_f.astype(np.float32), tf)
        mask = np.maximum(td.structure_mask_from_template(tpl_g),
                          td.structure_mask_from_input(
                              np.clip(np.round(corr), 0, 255).astype(np.uint8)))
        for vi, v in enumerate(variants):
            diff_raw = td.tolerance_diff(corr, tf, v["tau"], mask, None)
            vals = diff_raw[mask] if mask.any() else diff_raw.ravel()
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            tol = np.maximum(med + 3.0 * mad, 0.5).astype(np.float32)
            tol = np.where(mask, tol, np.ones_like(tf, np.float32))
            diff = td.tolerance_diff(corr, tf, v["tau"], mask, tol)
            blobs = blobs_variant(diff, v["min_area"], td.SIGMA_FLOOR,
                                  v["seed_k"], td.TOP_K, v["weighted"])
            score[vi]["far"] += sum(1 for b in blobs
                                    if b["z_diff"] >= 3.5 and b["area"] >= 100)

    print(f"{'wgt':>4}{'seedK':>7}{'area':>6}{'tau':>5}{'std':>6}{'cam':>6}"
          f"{'npos':>6}{'FARgood':>9}{'blob':>6}")
    for vi, v in enumerate(variants):
        s = score[vi]
        print(f"{int(v['weighted']):>4}{v['seed_k']:>7}{v['min_area']:>6}"
              f"{v['tau']:>5}{s['std']:>6}{s['cam']:>6}{s['npos']:>6}"
              f"{s['far']:>9}{s['nblob']:>6}")


if __name__ == "__main__":
    main()
