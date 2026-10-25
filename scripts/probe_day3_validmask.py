"""Ngay 3 FIX — probe gia thuyet "ria ngoai FOV" (CPU-only, khong GPU).

Gia thuyet: khi warp anh ve khong gian Template voi BORDER_REPLICATE, vung
canvas Template nam NGOAI anh dau vao bi lap day bang pixel bien (khong co
du lieu that). Vung nay tao residual lon, lam:
  (a) chiem TOP_K=8 blob -> blob loi that bi day ra ngoai;
  (b) bg_std toan anh bi phong len -> seed_thr qua cao -> mat loi nho.

Probe do 3 thu tren 36 anh loi 01/04 (standard + camera):
  1. Ty le blob top-8 nam ngoai vung FOV hop le (valid mask tu H).
  2. Recall khi giu nguyen vs khi lap vung ngoai FOV bang noi dung Template
     (coi nhu "khong co bang chung" o vung khong co du lieu).
  3. bg_mean/bg_std toan anh vs chi tren vung FOV hop le.

Khong ghi file nao; chi in stdout. So lieu chan doan — Verifier do lai bang
tests/bench_day3_branchB.py.
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

import src.reference.board_id_and_align as ba  # noqa: E402
import src.reference.tolerance_diff as td  # noqa: E402
from configs.resource_paths import LOCAL_TEMPLATES  # noqa: E402


def valid_mask_from_h(img_shape, h, tpl_shape, erode_px: int = 11) -> np.ndarray:
    """Vung canvas Template thuc su duoc phu boi anh dau vao (sau warp H)."""
    ones = np.ones(img_shape[:2], np.uint8)
    th, tw = int(tpl_shape[0]), int(tpl_shape[1])
    v = cv2.warpPerspective(ones, h, (tw, th), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if erode_px > 1:
        v = cv2.erode(v, np.ones((erode_px, erode_px), np.uint8))
    return v > 0


def recall_of(blobs, h_inv, gt_boxes, tau_tol: float = 3.0) -> int:
    if not blobs or not gt_boxes:
        return 0
    cents = np.float32([b["centroid_tpl"] for b in blobs])
    proj = ba.project_points(cents, h_inv)
    used = set()
    for (px, py) in proj:
        for gi, gb in enumerate(gt_boxes):
            if gi in used:
                continue
            if gb[0] - tau_tol <= px <= gb[2] + tau_tol and \
               gb[1] - tau_tol <= py <= gb[3] + tau_tol:
                used.add(gi)
                break
    return len(used)


def main() -> None:
    cache = ba.load_cache()
    tpls = {b: probe.read_gray(LOCAL_TEMPLATES / f"{b}.JPG")
            for b in cache["board_ids"]}
    items = probe.collect()
    acc: dict[str, dict] = {}

    for idx, it in enumerate(items):
        img = probe.read_bgr(Path(it["path"]))
        gt0 = [list(b) for _, b in it["gt"]]
        img_c, gt_c = probe.perturb(img, gt0, 20261023 + 101 + idx)
        for tag, im, gt in (("standard", img, gt0), ("camera", img_c, gt_c)):
            r = ba.identify_and_align(im, cache, fast=True)
            tpl_g = tpls[r["board_id"]]
            if r["warped"].shape[:2] != tpl_g.shape:
                tpl_g = cv2.resize(tpl_g, (r["warped"].shape[1],
                                           r["warped"].shape[0]),
                                   interpolation=cv2.INTER_AREA)
            valid = valid_mask_from_h(im.shape, r["H"], r["warped"].shape[:2])
            wg = cv2.cvtColor(r["warped"], cv2.COLOR_BGR2GRAY)
            wg_f = np.where(valid, wg, tpl_g).astype(np.uint8)

            out = td.run_branch_b(r["warped"], tpl_g)
            out_f = td.run_branch_b(wg_f, tpl_g)
            n_out = sum(1 for b in out["blobs"]
                        if not valid[int(b["centroid_tpl"][1]),
                                     int(b["centroid_tpl"][0])])
            rec = recall_of(out["blobs"], r["H_inv"], gt)
            rec_f = recall_of(out_f["blobs"], r["H_inv"], gt)

            diff = out["diff_map"]
            d_all = float(diff.std())
            d_val = float(diff[valid].std()) if valid.any() else 0.0
            frac_invalid = 1.0 - float(valid.mean())

            s = acc.setdefault(tag, {"n": 0, "rec": 0, "rec_f": 0, "npos": 0,
                                     "out_fov": 0, "nblob": 0, "nblob_f": 0,
                                     "std_all": [], "std_val": [],
                                     "frac_inv": [], "zero_after": 0})
            s["n"] += 1
            s["rec"] += rec
            s["rec_f"] += rec_f
            s["npos"] += len(gt)
            s["out_fov"] += n_out
            s["nblob"] += len(out["blobs"])
            s["nblob_f"] += len(out_f["blobs"])
            s["std_all"].append(d_all)
            s["std_val"].append(d_val)
            s["frac_inv"].append(frac_invalid)
            if len(out["blobs"]) == 8 and len(out_f["blobs"]) < 8:
                s["zero_after"] += 1

    print(f"{'scen':<9}{'n':>4}{'recall':>9}{'recallFill':>11}{'npos':>6}"
          f"{'blob':>6}{'blobFill':>9}{'outFOV':>8}{'stdAll':>8}{'stdValid':>9}"
          f"{'invArea':>9}{'satFix':>8}")
    for tag in ("standard", "camera"):
        s = acc[tag]
        print(f"{tag:<9}{s['n']:>4}{s['rec']:>9}{s['rec_f']:>11}{s['npos']:>6}"
              f"{s['nblob']:>6}{s['nblob_f']:>9}{s['out_fov']:>8}"
              f"{np.mean(s['std_all']):>8.2f}{np.mean(s['std_val']):>9.2f}"
              f"{np.mean(s['frac_inv']):>9.3f}{s['zero_after']:>8}")


if __name__ == "__main__":
    main()
