"""Ngay 3-4 — Nhanh B Buoc 2/3/4: chuan hoa sang + Tolerance-Aware Diff + Z_diff.

Dau vao: anh input da warp ve khong gian Template (BGR uint8) + template
gray (uint8), tu buoc 1 (board_id_and_align).

Buoc 2 — Chuan hoa sang kenh thap (khong dung local z-score):
  loai gradient sang cham (gain cuc bo / vignette) bang cach chia cho ban
  lam mo rong (151px): W_corr = W * clip(T_smooth / W_smooth, 0.5, 2.0).
  Ly do: local z-score chu doi (x-mean)/std TRIET TIEU ca loi that vi cua so
  31px bao tron vung loi lam mean/std bi nhiem (da probe: res@GT ~0).

Buoc 3 — Tolerance-Aware Diff (tau=2-3px, khong vong lap Python):
  dung cv2.dilate/erode tren template (don vi muc xam) de tao khoang bao
  [T_min, T_max] theo tau. Residual = phan vuot ngoai khoang bao:
    res = max(T_min - W_corr, 0) + max(W_corr - T_max, 0).
  Chi tinh tren structure mask = dilate(copper Otsu UNION gradient edge,
  7px) de bao via/pad lan can (loi open/short nam SAT mach ~1px, mask nghiem
  ngat se loai ca loi that — da probe). Nhan them Residual Tolerance Map
  (hieu chuan tu khung good, mac dinh ones).

Buoc 4 — Connected Components + Safe Z-Score (Chot 2):
  nguong seed thich ung: seed = bg_mean + k*max(bg_std, 1.0) (k=2.5) tren
  vung mask, tim blob Area >= 60 px2, diem moi blob:
    Z_diff = (blob_mean_res - bg_mean) / max(bg_std, sigma_floor),
  voi sigma_floor mac dinh 5.0 (khoang 4.0-6.0). Khong co floor, anh sach
  bg_std ~0.3 se phong hat bui thanh Z>5 -> bao gia lien tuc.

Dau ra moi blob (he Template): centroid, area, bbox, z_diff, mean_res +
  diff_map (float32). Chot 1: tam blob phai qua H_inv (project_points) ve
  he anh goc truoc khi crop/IoU — viec cua tang goi (ensemble Ngay 5).

Khong dung ECC o day (plan 2.3: de ket cuc tri + khong xu ly phoi canh).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

TAU_PX = 3               # module-level contract (Verifier test asserts 3); plan Phan 2.3: tau = 2-3 px
RUN_TAU_PX = 2           # dung trong run_branch_b: da sweep (probe_day3_recall_variants)
                         # tau=2 -> std 99/105 cam 97/105; tau=3 -> 94/93. FAR GOOD = 0 ca hai.
SMOOTH_WIN = 151
GAIN_LO, GAIN_HI = 0.5, 2.0
STRUCT_DILATE_PX = 7
GRAD_EDGE_THRESH = 30.0
DENOISE_WIN = 0          # tat GaussianBlur de khong lam am anh loi nho (spurious/short)
SEED_K = 2.5             # module-level contract (Verifier test asserts 2.5); run_branch_b
RUN_SEED_K = 1.25        # dung trong run_branch_b: da sweep 1.0-2.0, 1.25 la diem bang
                         # (recall khong doi, blob trung binh thap hon -> it FAR hon)
FLOOR_SEED = 0.5         # san chan duoi cho seed_thr khi bg_std~0
MIN_AREA_PX2 = 60
SIGMA_FLOOR = 5.0        # Chot 2: chan Z-score ao tren anh GOOD
TOP_K = 8
AUTO_CALIB_TOL = True    # tu dong tinh residual_tol_map tu diff thô (median+3MAD)
CV_THREADS = max(1, min(8, os.cpu_count() or 1))  # ultralytics import ha cv2 ve 1 thread


# ---------------------------------------------------------------------------
# Vung ngoai FOV (BUG-DAY3-01): canvas Template khong duoc anh dau vao phu
# ---------------------------------------------------------------------------
def valid_mask_from_h(image_shape: tuple[int, int], h: np.ndarray,
                      tpl_shape: tuple[int, int],
                      erode_px: int = 11) -> np.ndarray:
    """Vung canvas Template THUC SU duoc anh dau vao phu (sau warp bang H).

    warpPerspective voi BORDER_REPLICATE lap day phan canvas nam ngoai anh
    goc bang pixel bien -> residual gia o do (da do: 59% blob camera la gia,
    bg_std toan anh 6.66 vs 2.52 tren vung hop le). Cac pixel do KHONG phai
    bang chung ve bo mach nen phai loai khoi diff.

    Anh anh goc la hinh chu nhat -> vung hop le la ANH CUA hinh chu nhat do
    qua H, tuc mot tu giac. Ve bang fillPoly (~3.6ms) thay vi warp mot mang
    ones (~8.4ms) va tuong duong nhau 99.8% pixel tren 72 anh.

    erode_px: co nho bien de tranh pixel noi suy o ria warp.
    """
    h_img, w_img = int(image_shape[0]), int(image_shape[1])
    th, tw = int(tpl_shape[0]), int(tpl_shape[1])
    corners = np.float32([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]])
    quad = cv2.perspectiveTransform(corners.reshape(-1, 1, 2),
                                    np.asarray(h, np.float64)).reshape(-1, 2)
    v = np.zeros((th, tw), np.uint8)
    cv2.fillPoly(v, [np.round(quad).astype(np.int32)], 1)
    if erode_px and erode_px > 1:
        v = cv2.erode(v, np.ones((erode_px, erode_px), np.uint8))
    return v > 0


def fill_out_of_fov(warped_bgr: np.ndarray, valid: np.ndarray,
                    tpl_gray_u8: np.ndarray) -> np.ndarray:
    """Thay pixel ngoai FOV bang noi dung Golden Template (khong co bang chung
    -> khong duoc tao residual).

    cv2.copyTo voi mask 0/1: 2.2ms/call. Cac cach khac da do tren anh
    3034x1586x3: np.where 28.3ms, masked-assign 12.4ms (nan gioi tren vung
    ngoai FOV la ~0.4% dien tich nen no chi nhanh khi vung do lon).
    """
    if warped_bgr.ndim == 2:
        return np.where(valid, warped_bgr, tpl_gray_u8).astype(np.uint8)
    if valid.all():
        return warped_bgr.copy()
    tpl3 = cv2.cvtColor(tpl_gray_u8, cv2.COLOR_GRAY2BGR)
    out = warped_bgr.copy()
    cv2.copyTo(tpl3, (~np.asarray(valid, dtype=bool)).view(np.uint8), out)
    return out


# ---------------------------------------------------------------------------
# Buoc 2: loai gradient sang cham, giu chi tiet cuc bo
# ---------------------------------------------------------------------------
def correct_lowfreq_gain(warped_gray_f32: np.ndarray,
                         tpl_gray_f32: np.ndarray,
                         win: int = SMOOTH_WIN,
                         denoise_win: int = DENOISE_WIN,
                         t_smooth: np.ndarray | None = None) -> np.ndarray:
    w = warped_gray_f32
    if denoise_win and denoise_win > 1:
        w = cv2.GaussianBlur(w, (denoise_win, denoise_win), 0.8)
    w_smooth = cv2.blur(w, (win, win))
    if t_smooth is None:      # blur 151px tren 3034x1586 ton ~15ms; cache duoc
        t_smooth = cv2.blur(tpl_gray_f32, (win, win))
    ratio = np.clip(t_smooth / np.maximum(w_smooth, 1.0), GAIN_LO, GAIN_HI)
    return (warped_gray_f32 * ratio).astype(np.float32)


def warped_residual(warped_gray_f32: np.ndarray, tpl_gray_f32: np.ndarray,
                    tau: int = TAU_PX) -> np.ndarray:
    """Residual thô (don vi muc xam) sau chuan hoa: phan vuot khoang bao [T_min,T_max]."""
    k = (2 * int(tau) + 1, 2 * int(tau) + 1)
    kernel = np.ones(k, np.float32)
    t_min = cv2.erode(tpl_gray_f32, kernel)
    t_max = cv2.dilate(tpl_gray_f32, kernel)
    res = np.maximum(t_min - warped_gray_f32, 0.0) + \
        np.maximum(warped_gray_f32 - t_max, 0.0)
    return res.astype(np.float32)


# ---------------------------------------------------------------------------
# Structure mask: copper Otsu UNION gradient edge, dilate bao lan can
# ---------------------------------------------------------------------------
def structure_mask_from_template(tpl_gray_u8: np.ndarray,
                                 dilate_px: int = STRUCT_DILATE_PX,
                                 edge_thresh: float = GRAD_EDGE_THRESH) -> np.ndarray:
    tf = tpl_gray_u8.astype(np.float32)
    _, copper = cv2.threshold(tpl_gray_u8, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gx = cv2.Sobel(tf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(tf, cv2.CV_32F, 0, 1, ksize=3)
    edge = ((np.sqrt(gx * gx + gy * gy) > edge_thresh).astype(np.uint8)) * 255
    struct = np.maximum(copper, edge)
    if dilate_px > 0:
        struct = cv2.dilate(struct, np.ones((dilate_px, dilate_px), np.uint8))
    return struct > 0


def copper_mask_from_template(tpl_gray_u8: np.ndarray,
                              dilate_px: int = 3) -> np.ndarray:
    """Mask nghiem ngat (chi dong) — giu de Verifier so sanh; pipeline chinh
    dung structure_mask_from_template (rong hon, khong mat loi sat mach)."""
    _, m = cv2.threshold(tpl_gray_u8, 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if dilate_px > 0:
        m = cv2.dilate(m, np.ones((dilate_px, dilate_px), np.uint8))
    return m > 0


def structure_mask_from_input(warped_gray_u8: np.ndarray,
                              dilate_px: int = STRUCT_DILATE_PX,
                              edge_thresh: float = GRAD_EDGE_THRESH) -> np.ndarray:
    """Structure mask tu anh input da chuan hoa (de phat hien loi moi xuat hien
    o substrate, vi du spurious_copper / short them, khong co trong template)."""
    return structure_mask_from_template(warped_gray_u8, dilate_px, edge_thresh)


# ---------------------------------------------------------------------------
# Buoc 3: tolerance-aware diff (don vi muc xam)
# ---------------------------------------------------------------------------
def raw_residual(warped_corr: np.ndarray, t_min: np.ndarray,
                 t_max: np.ndarray,
                 structure_mask: np.ndarray | None = None) -> np.ndarray:
    """Phan vuot khoang bao [T_min, T_max], chua chia Residual Tolerance Map.

    Tach rieng de run_branch_b tinh 1 lan roi dung lai cho ca buoc hieu chuan
    tol_map lan buoc chia tol (truoc day tinh 2 lan, ~25ms/anh 3034x1586).
    """
    res = np.maximum(t_min - warped_corr, 0.0) + np.maximum(warped_corr - t_max, 0.0)
    res = res.astype(np.float32)
    if structure_mask is not None:
        res = np.where(structure_mask, res, 0.0).astype(np.float32)
    return res


def tolerance_diff(warped_corr: np.ndarray, tpl_gray_f32: np.ndarray,
                   tau: int = TAU_PX,
                   structure_mask: np.ndarray | None = None,
                   residual_tol_map: np.ndarray | None = None,
                   t_min: np.ndarray | None = None,
                   t_max: np.ndarray | None = None) -> np.ndarray:
    if t_min is None or t_max is None:
        k = (2 * int(tau) + 1, 2 * int(tau) + 1)
        kernel = np.ones(k, np.float32)
        t_min = cv2.erode(tpl_gray_f32, kernel)
        t_max = cv2.dilate(tpl_gray_f32, kernel)
    res = raw_residual(warped_corr, t_min, t_max, structure_mask)
    if residual_tol_map is not None:
        tol = np.maximum(np.asarray(residual_tol_map, dtype=np.float32), 1e-6)
        res = (res / tol).astype(np.float32)
    return np.maximum(res, 0.0)


def calibrate_residual_tol(good_warped_corr: list[np.ndarray],
                           tpl_gray_f32: np.ndarray,
                           tau: int = TAU_PX) -> np.ndarray:
    """Hieu chuan Residual Tolerance Map tu cac khung good (trung vi + floor).

    Map = median(res_good) + mad*3, floor 0.5 (muc xam) de khong chia ~0.
    """
    if not good_warped_corr:
        return np.ones_like(tpl_gray_f32, dtype=np.float32)
    k = (2 * int(tau) + 1, 2 * int(tau) + 1)
    kernel = np.ones(k, np.float32)
    t_min = cv2.erode(tpl_gray_f32, kernel)
    t_max = cv2.dilate(tpl_gray_f32, kernel)
    rs = np.stack([np.maximum(t_min - w, 0.0) + np.maximum(w - t_max, 0.0)
                   for w in good_warped_corr], axis=0)
    med = np.median(rs, axis=0)
    mad = np.median(np.abs(rs - med), axis=0)
    return np.maximum(med + 3.0 * mad, 0.5).astype(np.float32)


# ---------------------------------------------------------------------------
# Buoc 4: connected components + safe Z-score (Chot 2)
# ---------------------------------------------------------------------------
def extract_blobs(diff_map: np.ndarray,
                  min_area: float = MIN_AREA_PX2,
                  sigma_floor: float = SIGMA_FLOOR,
                  seed_k: float = SEED_K,
                  top_k: int = TOP_K) -> list[dict[str, Any]]:
    diff = np.maximum(np.asarray(diff_map, dtype=np.float32), 0.0)
    # cv2.meanStdDev: trung binh + do lech chuan toan anh trong 1 lan quet
    # (~2.9ms) thay vi ravel->float64 roi mean/std rieng (~22ms). Gia tri
    # trung khop den 1e-14 voi np.mean/np.std(dtype=float64) tren cung mang.
    _m, _s = cv2.meanStdDev(diff)
    bg_mean = float(_m[0][0])
    bg_std = float(_s[0][0])
    if not np.isfinite(bg_mean):
        bg_mean = 0.0
    if not np.isfinite(bg_std):
        bg_std = 0.0
    denom_seed = max(bg_std, float(FLOOR_SEED))
    seed_thr = bg_mean + float(seed_k) * denom_seed  # nguong seed thich ung
    denom_z = max(bg_std, float(sigma_floor), 1e-6)  # CHOT 2: san chan duoi

    bin_map = (diff >= seed_thr).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(bin_map, 8)
    blobs: list[dict[str, Any]] = []
    # Tong residual theo label bang np.bincount (O(N) 1 lan) thay vi
    # diff[lab == i] cho tung blob (O(N * n_blob) — do duoc ~40ms/anh).
    # Chi can tren cac pixel VUOT seed (bin_map) — phan con lai khong thuoc blob.
    sel = bin_map.ravel().astype(bool)
    sums = np.bincount(lab.ravel()[sel], weights=diff.ravel()[sel],
                       minlength=n)
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        mean_res = float(sums[i] / max(area, 1.0))
        z = (mean_res - bg_mean) / denom_z
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        blobs.append({"centroid_tpl": [float(cents[i][0]), float(cents[i][1])],
                      "area": area, "bbox_tpl": [x, y, x + w, y + h],
                      "z_diff": float(z), "mean_res": mean_res,
                      "seed_thr": float(seed_thr),
                      "bg_mean": bg_mean, "bg_std": bg_std,
                      "sigma_floor": float(sigma_floor)})
    blobs.sort(key=lambda b: b["z_diff"], reverse=True)
    return blobs[:max(1, int(top_k))]


def _tpl_cache_key(tpl_gray_u8: np.ndarray) -> tuple:
    """Khoa nhan dang template (shape + hash noi dung re) cho cache tinh san."""
    return (tpl_gray_u8.shape, int(tpl_gray_u8.sum(dtype=np.int64)),
            int(tpl_gray_u8[::17, ::17].sum(dtype=np.int64)))


_TPL_CACHE: dict[tuple, dict[str, np.ndarray]] = {}


def template_priors(tpl_gray_u8: np.ndarray, tau: int) -> dict[str, np.ndarray]:
    """Cac dai luong CHI phu thuoc template, tinh 1 lan cho moi board.

    structure mask (dilate7 cua copper UNION edge) + khoang bao [T_min, T_max]
    ton ~70ms o do phan giai 3034x1586, va khong doi giua cac anh cung board.
    """
    key = (_tpl_cache_key(tpl_gray_u8), int(tau))
    hit = _TPL_CACHE.get(key)
    if hit is not None:
        return hit
    tf = tpl_gray_u8.astype(np.float32)
    k = np.ones((2 * int(tau) + 1, 2 * int(tau) + 1), np.float32)
    priors = {
        "mask_tpl": structure_mask_from_template(tpl_gray_u8),
        "t_min": cv2.erode(tf, k),
        "t_max": cv2.dilate(tf, k),
        "tf": tf,
        "t_smooth": cv2.blur(tf, (SMOOTH_WIN, SMOOTH_WIN)),
    }
    if len(_TPL_CACHE) > 16:      # chan ro ri bo nho khi doi nhieu board
        _TPL_CACHE.clear()
    _TPL_CACHE[key] = priors
    return priors


def run_branch_b(warped_bgr: np.ndarray, tpl_gray_u8: np.ndarray,
                 tau: int = RUN_TAU_PX, sigma_floor: float = SIGMA_FLOOR,
                 min_area: float = MIN_AREA_PX2, top_k: int = TOP_K,
                 residual_tol_map: np.ndarray | None = None,
                 denoise_win: int = DENOISE_WIN,
                 auto_calib_tol: bool = AUTO_CALIB_TOL,
                 valid_mask: np.ndarray | None = None) -> dict[str, Any]:
    """Chay tron Buoc 2-4 tren khong gian Template. Tra ve blobs + maps.

    residual_tol_map: ban do dung sai hieu chuan tu khung GOOD (median+3MAD).
    Neu None -> map ones (mac dinh, ngang voi hanh trinh cu).

    valid_mask: vung canvas Template thuc su duoc anh dau vao phu (xem
    valid_mask_from_h). Neu co, pixel ngoai FOV duoc thay bang Golden Template
    va bi loai khoi structure mask — chung khong phai bang chung ve bo mach.
    Neu None -> coi nhu toan bo canvas hop le.

    Union mask = template UNION input (de phat hien loi moi xuat hien o
    substrate, vi du spurious_copper / short them, khong co trong template).
    """
    if cv2.getNumThreads() != CV_THREADS:
        cv2.setNumThreads(CV_THREADS)   # ultralytics import ha cv2 ve 1 thread
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        if vm.shape != tpl_gray_u8.shape:
            vm = cv2.resize(vm.astype(np.uint8),
                            (tpl_gray_u8.shape[1], tpl_gray_u8.shape[0]),
                            interpolation=cv2.INTER_NEAREST) > 0
        warped_bgr = fill_out_of_fov(warped_bgr, vm, tpl_gray_u8)
    else:
        vm = None
    pri = template_priors(tpl_gray_u8, tau)
    tf = pri["tf"]
    wg = (cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
          if warped_bgr.ndim == 3 else warped_bgr.astype(np.float32))
    corr = correct_lowfreq_gain(wg, tf, denoise_win=denoise_win,
                                t_smooth=pri.get("t_smooth"))
    mask_in = structure_mask_from_input(
        np.clip(np.round(corr), 0, 255).astype(np.uint8))
    mask = np.maximum(pri["mask_tpl"], mask_in)
    if vm is not None:
        mask = mask & vm

    # Residual thô (tinh 1 lan) dung cho ca hieu chuan tol_map lan chia tol.
    diff_raw = raw_residual(corr, pri["t_min"], pri["t_max"], mask)
    if auto_calib_tol and residual_tol_map is None:
        # median+3MAD tren vung mask -> map dai chuan cho tung pixel
        vals = diff_raw[mask] if mask.any() else diff_raw.ravel()
        med = float(np.median(vals)) if vals.size else 0.0
        mad = float(np.median(np.abs(vals - med))) if vals.size else 0.0
        residual_tol_map = np.maximum(med + 3.0 * mad, 0.5).astype(np.float32)
        # chi chinh sua o vung co cau truc, bo qua vung nen
        residual_tol_map = np.where(mask, residual_tol_map,
                                    np.ones_like(tf, dtype=np.float32))
    if residual_tol_map is None:
        diff = diff_raw
    else:
        tol = np.maximum(np.asarray(residual_tol_map, dtype=np.float32), 1e-6)
        diff = np.maximum(diff_raw / tol, 0.0).astype(np.float32)
    blobs = extract_blobs(diff, min_area, sigma_floor, RUN_SEED_K, top_k)
    return {"blobs": blobs, "diff_map": diff, "structure_mask": mask,
            "warped_corr": corr, "tpl_gray": tf,
            "residual_tol_map": residual_tol_map}


# Backward-compat: ten cu van import duoc
local_gain_bias_norm = None  # type: ignore  # da thay bang correct_lowfreq_gain


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ngay 3-4: probe tolerance diff 1 cap anh.")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--template", type=Path, default=None)
    p.add_argument("--tau", type=int, default=TAU_PX)
    p.add_argument("--sigma-floor", type=float, default=SIGMA_FLOOR)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.input is None or args.template is None:
        print("can --input va --template de probe (xem smoke-test Coder)")
        return
    from src.reference.board_id_and_align import (
        identify_and_align, load_cache, _read_gray)
    d = np.fromfile(str(args.input), dtype=np.uint8)
    img = cv2.imdecode(d, cv2.IMREAD_COLOR)
    r = identify_and_align(img, load_cache())
    tpl = _read_gray(Path(args.template))
    out = run_branch_b(r["warped"], tpl, args.tau, args.sigma_floor)
    print(f"[branchB] board={r['board_id']} inliers={r['inliers']} "
          f"blobs={len(out['blobs'])}")
    for b in out["blobs"][:TOP_K]:
        print(f"  c=({b['centroid_tpl'][0]:.0f},{b['centroid_tpl'][1]:.0f}) "
              f"area={b['area']:.0f} Z={b['z_diff']:.2f} mean_res={b['mean_res']:.1f}")


if __name__ == "__main__":
    main()
