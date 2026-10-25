"""Ngay 3 — Nhanh B Buoc 1: Board ID & Coarse Alignment (ORB + RANSAC Homography).

Chuc nang:
  * Tien trich ORB dac trung cua 10 Golden Templates (assets/templates/,
    01/04/05-12.JPG) -> cache `.npz` trong project moi (keypoints, descriptors,
    kich thuoc anh).
  * Voi anh dau vao: khop BFMatcher + Lowe ratio test voi tung template, chon
    ma board co so inliers cao nhat, uoc luong Homography H (Input -> Template)
    bang cv2.findHomography(RANSAC) va H_inv = H^-1 (Template -> Input).

3 chot chan ap dung tai day:
  * Chot 1 (Coordinate Sync): moi blob tim trong khong gian Template BAT BUOC
    chieu nguoc qua H_inv (cv2.perspectiveTransform) ve khong gian anh goc
    truoc khi crop 640 va tinh IoU. Ham warp_to_template() + project_points()
    thuc thi dung 2 chieu nay.
  * Khong dung cv2.findTransformECC lam nhanh chinh (de ket cuc tri dia
    phuong tren luoi pad + khong xu ly phoi canh 8-DoF). Neu refine ECC sau
    nay: inputMask=None hoac CV_8UC1 (tranh crash OpenCV 4.11 _mask.type()).
  * Zero-touch: template doc tu assets/templates (trong project); .npz cache
    ghi trong project moi (qua assert_writable_ok).

Thuat ngu: 01/04 la "Zero-defect-training Boards" (co Golden Sample nhung
chua co du lieu loi) — khong goi la "unseen".
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.resource_paths import (  # noqa: E402
    LOCAL_TEMPLATES,
    NEW_PROJECT_DIR,
    TEMPLATE_FILES,
    assert_writable_ok,
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.reference.tolerance_diff import (  # noqa: E402
    fill_out_of_fov,
    valid_mask_from_h,
)

DEFAULT_CACHE = NEW_PROJECT_DIR / "assets" / "templates" / "orb_cache.npz"
N_FEATURES = 8000
RATIO_TEST = 0.75
RANSAC_THRESH = 5.0
MIN_INLIERS_BOARD_ID = 30
TOP_PER_TEMPLATE_MATCHES = 4000
EARLY_EXIT_INLIERS = 60        # inliers >= nguong nay -> chac board dung, dung som
                               # (do tren 80 anh stress cua Verifier — shift 25px,
                               #  rot 2deg, gain 0.85-1.15, bias 12, vignette, noise:
                               #  board dung MIN 107 inliers, board canh tranh MAX 13
                               #  -> bien 4.6x tren doi thu, 1.8x duoi truong hop xau nhat)
FAST_TOP_MATCHES = 400         # cap matches dùng RANSAC ở chế độ fast

# Shortlist bằng "van tay" gradient thumbnail (BUG-DAY3-02): knnMatch x10
# template la nghen chinh (~120ms). Van tay gradient @48x96 chi ton ~10ms va
# dat top-1 = 100% tren 92 anh bench (36 std + 36 camera + 16 train/val +
# 4 GOOD) -> chi can ORB-match 1-2 template la du.
SHORTLIST_TOP_K = 2            # so template di tiep vao ORB + RANSAC
SHORTLIST_H, SHORTLIST_W = 48, 96

# BUG-DAY3-02 (nghen that su): `import ultralytics` goi cv2.setNumThreads(1)
# va dat OMP_NUM_THREADS=1 (side-effect toan cuc). Align chay tren 1 thread
# cham gap ~4x: 55.9ms (16 thread) -> 215ms (1 thread). Dat lai so thread
# truoc moi lan align; cv2.getNumThreads() rat re nen khong anh huong latency.
CV_THREADS = max(1, min(8, os.cpu_count() or 1))

# Template 8000 keypoint lam knnMatch ton ~18ms/template (10 templates = 180ms).
# Giu mau cach deu 3000 descriptor/template: sweep tren 72 anh std+cam cho
# recall Nhanh B tot nhat trong cac muc thu (std 99/105, cam 98/105; cap 0 =
# 99/97, cap 4000 = 99/96, cap 2000 = 95/99) va align 54.4 -> 41.4ms.
# Lay mau cach deu (linspace) de phu deu toan bo bo mach thay vi lay 3000 dau
# tien (ORB tra ve theo thu tu response giam dan -> don ve mot phia).
FAST_TPL_MAX_DESC = 3000

# FLANN LSH params cho ORB descriptor. LUU Y: LSH chi nhan descriptor CV_8U;
# truyen float32 se nem cv2.error -> roi ve BFMatcher (da do: BF nhanh hon
# FLANN cho 10 template 32-byte, nen BF la duong mac dinh, FLANN la fallback).
FLANN_INDEX_LSH = 6
FLANN_PARAMS = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6, key_size=12, multi_probe_level=1)
FLANN_SEARCH_PARAMS = dict(checks=32)


# ---------------------------------------------------------------------------
# Cache build
# ---------------------------------------------------------------------------
def _orb() -> cv2.ORB:
    return cv2.ORB_create(nfeatures=N_FEATURES)


def _read_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")
    return img


def build_cache(cache_path: Path = DEFAULT_CACHE,
                n_features: int = N_FEATURES) -> dict[str, Any]:
    """Trich ORB tung template, luu .npz (ghi trong project moi)."""
    cache_path = assert_writable_ok(Path(cache_path))
    orb = cv2.ORB_create(nfeatures=n_features)
    out: dict[str, Any] = {"board_ids": [], "shapes": []}
    descs: list[np.ndarray] = []
    kps_xy: list[np.ndarray] = []
    for fname in TEMPLATE_FILES:
        bid = Path(fname).stem
        img = _read_gray(LOCAL_TEMPLATES / fname)
        kp, des = orb.detectAndCompute(img, None)
        if des is None or len(kp) == 0:
            raise RuntimeError(f"ORB khong tim duoc keypoint: {fname}")
        out["board_ids"].append(bid)
        out["shapes"].append(list(img.shape))
        kps_xy.append(np.float32([p.pt for p in kp]))
        descs.append(des.astype(np.uint8))
    n = [len(k) for k in kps_xy]
    max_kp = max(n)
    kp_pad = np.zeros((len(kps_xy), max_kp, 2), np.float32)
    des_pad = np.zeros((len(descs), max_kp, 32), np.uint8)
    for i, (k, d) in enumerate(zip(kps_xy, descs)):
        kp_pad[i, :len(k)] = k
        des_pad[i, :len(d)] = d
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path, board_ids=np.array(out["board_ids"]),
        shapes=np.array(out["shapes"], np.int32), n_kp=np.array(n, np.int32),
        keypoints=kp_pad, descriptors=des_pad)
    out.update({"cache": str(cache_path), "n_kp": n})
    for b, c in zip(out["board_ids"], n):
        print(f"  [cache] board {b}: {c} keypoints")
    print(f"[cache] wrote {cache_path}")
    return out


def load_cache(cache_path: Path = DEFAULT_CACHE) -> dict[str, Any]:
    z = np.load(str(cache_path), allow_pickle=False)
    board_ids = [str(b) for b in z["board_ids"]]
    n_kp = [int(v) for v in z["n_kp"]]
    kps, des = z["keypoints"], z["descriptors"]
    kps_list = [np.ascontiguousarray(kps[i, :n_kp[i]]) for i in range(len(board_ids))]
    des_list = [np.ascontiguousarray(des[i, :n_kp[i]]) for i in range(len(board_ids))]
    # FLANN LSH indexer cho tung template descriptor (fallback khi BF cham).
    # Descriptor phai giu CV_8U — LSH khong nhan float32.
    flann_indexers: list[cv2.FlannBasedMatcher | None] = []
    for d in des_list:
        try:
            fm = cv2.FlannBasedMatcher(FLANN_PARAMS, FLANN_SEARCH_PARAMS)
            fm.add([d.astype(np.uint8)])
            fm.train()
            flann_indexers.append(fm)
        except cv2.error:
            flann_indexers.append(None)
    # Ban "thua" cua descriptor de match nhanh (xem FAST_TPL_MAX_DESC).
    kps_fast, des_fast = [], []
    for k, d in zip(kps_list, des_list):
        if len(d) > FAST_TPL_MAX_DESC:
            sel = np.linspace(0, len(d) - 1, FAST_TPL_MAX_DESC).astype(np.int32)
            kps_fast.append(np.ascontiguousarray(k[sel]))
            des_fast.append(np.ascontiguousarray(d[sel]))
        else:
            kps_fast.append(k)
            des_fast.append(d)
    # Van tay gradient thumbnail + anh gray cua tung template. Doc 1 lan o day:
    # decode JPEG 3034x1586 ton ~60-100ms, khong duoc lap lai moi lan align.
    tpl_grays = [_read_gray(LOCAL_TEMPLATES / f"{b}.JPG") for b in board_ids]
    fps = np.stack([fingerprint(g) for g in tpl_grays], axis=0)
    return {"board_ids": board_ids, "shapes": [tuple(s) for s in z["shapes"]],
            "n_kp": n_kp, "keypoints": kps_list, "descriptors": des_list,
            "keypoints_fast": kps_fast, "descriptors_fast": des_fast,
            "flann_indexers": flann_indexers, "fingerprints": fps,
            "templates_gray": tpl_grays}


# ---------------------------------------------------------------------------
# Shortlist: van tay gradient thumbnail (bat bien gain/bias/vignette)
# ---------------------------------------------------------------------------
def fingerprint(gray: np.ndarray,
                th: tuple[int, int] = (SHORTLIST_H, SHORTLIST_W)) -> np.ndarray:
    """Vector dac trung gradient @48x96, chuan hoa mean/std -> cos ~ NCC.

    Dung gradient (khong dung cuong do) de triet gain/bias/vignette cua kich
    ban camera. Da do: top-1 = 100% tren 92 anh bench.
    INTER_LINEAR chu khong INTER_AREA: ha mau 3034->96 bang AREA ton ~12ms,
    LINEAR ~0.1ms, va van dat top-1 100% o 48x96 (AREA khong tot hon).
    """
    t = cv2.resize(gray, (int(th[1]), int(th[0])),
                   interpolation=cv2.INTER_LINEAR)
    f = t.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    f = np.sqrt(gx * gx + gy * gy)
    f = (f - float(f.mean())) / max(float(f.std()), 1e-6)
    return f.ravel().astype(np.float32)


def shortlist_boards(gray: np.ndarray, fingerprints: np.ndarray,
                     board_ids: list[str],
                     top_k: int = SHORTLIST_TOP_K) -> list[tuple[str, float]]:
    """Xep hang template theo cosine voi van tay gradient cua anh dau vao."""
    fp = fingerprint(gray)
    n = np.linalg.norm(fp)
    t = np.linalg.norm(fingerprints, axis=1)
    cos = (fingerprints @ fp) / np.maximum(t * max(float(n), 1e-9), 1e-9)
    order = np.argsort(-cos)[:max(1, int(top_k))]
    return [(board_ids[i], float(cos[i])) for i in order]


# ---------------------------------------------------------------------------
# Board ID + H / H_inv
# ---------------------------------------------------------------------------
def match_template(input_des: np.ndarray, tpl_des: np.ndarray,
                   ratio: float = RATIO_TEST,
                   flann_indexer: cv2.FlannBasedMatcher | None = None,
                   flann_k: int = 2) -> list[cv2.DMatch]:
    if input_des is None or tpl_des is None or len(input_des) < 2 or len(tpl_des) < 2:
        return []
    if flann_indexer is not None:
        try:
            # kNN qua FLANN LSH; LSH CHI nhan descriptor CV_8U (float32 -> cv2.error)
            matches = flann_indexer.knnMatch(np.ascontiguousarray(input_des, np.uint8),
                                             k=flann_k)
            good: list[cv2.DMatch] = []
            for pair in matches:
                if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                    good.append(pair[0])
            good.sort(key=lambda m: m.distance)
            return good[:TOP_PER_TEMPLATE_MATCHES]
        except cv2.error:
            pass  # fallback ve BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = bf.knnMatch(input_des, tpl_des, k=2)
    good: list[cv2.DMatch] = []
    for p in pairs:
        if len(p) == 2 and p[0].distance < ratio * p[1].distance:
            good.append(p[0])
    good.sort(key=lambda m: m.distance)
    return good[:TOP_PER_TEMPLATE_MATCHES]


def estimate_h(kp_in: np.ndarray, kp_tpl: np.ndarray,
               matches: list[cv2.DMatch],
               ransac_thresh: float = RANSAC_THRESH) -> tuple[np.ndarray | None, int]:
    """H: Input -> Template (RANSAC). Tra ve (H | None, n_inliers)."""
    if len(matches) < 4:
        return None, 0
    src = np.float32([kp_in[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_tpl[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if h is None or mask is None:
        return None, 0
    return h.astype(np.float64), int(mask.sum())


def _match_one_board(bid: str, des_in: np.ndarray, kp_in: np.ndarray,
                     cache: dict[str, Any], ransac_thresh: float,
                     best: dict[str, Any]) -> int:
    """ORB-match 1 template o che do fast (descriptor da cap) + cap nhat `best`.

    Tach rieng de dung chung cho ca shortlist lan buoc escalation (BUG-DAY3-04).
    Tra ve so inliers cua template nay.
    """
    idx = cache["board_ids"].index(bid)
    kt = np.ascontiguousarray(cache["keypoints_fast"][idx])
    dt = np.ascontiguousarray(cache["descriptors_fast"][idx])
    shape = cache["shapes"][idx]
    m = match_template(des_in, dt)
    h, inl = estimate_h(kp_in, kt, m[:FAST_TOP_MATCHES], ransac_thresh)
    best["scores"][bid] = inl
    if inl > best["inliers"]:
        best.update(board_id=bid, H=h, inliers=inl, tpl_shape=tuple(shape))
    return inl


def identify_and_align(image_bgr: np.ndarray, cache: dict[str, Any] | None = None,
                       cache_path: Path = DEFAULT_CACHE,
                       ransac_thresh: float = RANSAC_THRESH,
                       fast: bool = False,
                       fast_nfeatures: int = 1200,
                       fast_scale: float = 0.5,
                       fast_topk_h: int = SHORTLIST_TOP_K) -> dict[str, Any]:
    """Nhan dien board + H / H_inv + warp ve khong gian Template.

    Tra ve dict: board_id, H (Input->Template), H_inv (Template->Input),
    inliers, warped (anh input da warp ve khong gian Template, BGR uint8),
    tpl_shape, scores {bid: inliers}, mode ("full"/"fast").

    fast=True: detect ORB tren anh half-res + vote so match 1 lan cho ca 10
    template, chi chay RANSAC H cho top-3 (muc tieu ~45ms/anh, dung cho
    benchmark; che do mac dinh full chinh xac hon cho Verifier doi chieu).
    """
    if cache is None:
        cache = load_cache(cache_path)
    if cv2.getNumThreads() != CV_THREADS:
        # ultralytics import da ha ve 1 thread (xem CV_THREADS) — khoi phuc
        cv2.setNumThreads(CV_THREADS)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    if fast:
        small = cv2.resize(gray, None, fx=fast_scale, fy=fast_scale,
                           interpolation=cv2.INTER_AREA)
        orb = cv2.ORB_create(nfeatures=fast_nfeatures)
        kp_in_list, des_in = orb.detectAndCompute(small, None)
        scale_back = 1.0 / fast_scale
    else:
        orb = _orb()
        kp_in_list, des_in = orb.detectAndCompute(gray, None)
        scale_back = 1.0
    if des_in is None or not kp_in_list:
        raise RuntimeError("ORB khong tim duoc keypoint tren anh dau vao")
    kp_in = np.float32([p.pt for p in kp_in_list]) * scale_back

    best: dict[str, Any] = {"board_id": None, "H": None, "inliers": -1,
                            "scores": {}, "shortlist_escalated": False}
    if fast:
        # Shortlist van tay gradient -> chi ORB-match top-K template (BUG-DAY3-02:
        # knnMatch x10 la nghen ~120-250ms; shortlist ~1ms + match 2 template
        # ~25ms). Da sweep tren 130 anh (72 std+cam + 54 train/val + 4 GOOD):
        # scale 0.5 / nf 1200 / top-2 / BF -> 130/130 dung, median ~46ms.
        # GIU nf=1200: ha xuong 700 nhanh hon ~5ms nhung H kem chinh xac hon
        # tren anh camera (recall cam 97 -> 93/105) — khong dang doi.
        # FLANN LSH cham gap ~2x voi descriptor 32-byte nen duong fast dung BFMatcher.
        #
        # BUG-DAY3-04: shortlist la GOI Y, khong phai LOC CUNG. Van tay gradient
        # co the xep board dung ra ngoai top-2 khi anh bi dich/xoay manh (probe
        # cua Verifier: 06 k=0 rank 3, 08 k=4 rank 3 -> pred=None, mat hoan toan
        # ket qua). Neu top-K chua cho inliers du tu tin thi mo rong ORB-match
        # sang cac template con lai theo thu tu cosine. Chi tra gia nay khi
        # that su can: do tren tap stress 160 anh, escalation kich hoat 1/80
        # (shift 25px) va 3/80 (shift 50px), ~170ms cho ca that bai; duong
        # binh thuong khong doi (~43-50ms).
        cand = shortlist_boards(gray, cache["fingerprints"], cache["board_ids"])
        for bid, cos in cand:
            best["scores"][bid] = round(cos, 4)
        order = [b for b, _c in cand]
        for bid in order:
            _match_one_board(bid, des_in, kp_in, cache, ransac_thresh, best)
            if best["inliers"] >= EARLY_EXIT_INLIERS:
                break
        if best["inliers"] < EARLY_EXIT_INLIERS:
            # Escalation: mo rong sang moi template con lai theo thu tu cosine.
            # Dieu kien la "chua du tu tin" (<EARLY_EXIT) chu khong phai "da hong"
            # (<MIN_INLIERS): shortlist sai co the van cho inliers 30-60 tu
            # template gan giong, va khi do ta se tra nham board.
            for bid in [b for b in cache["board_ids"] if b not in best["scores"]]:
                _match_one_board(bid, des_in, kp_in, cache, ransac_thresh, best)
                if best["inliers"] >= EARLY_EXIT_INLIERS:
                    break
            best["shortlist_escalated"] = True
    else:
        for i, (bid, kt, dt, shape) in enumerate(zip(cache["board_ids"], cache["keypoints"],
                                      cache["descriptors"], cache["shapes"])):
            fi = cache["flann_indexers"][i]
            m = match_template(des_in, np.ascontiguousarray(dt), flann_indexer=fi)
            h, inl = estimate_h(kp_in, np.ascontiguousarray(kt), m, ransac_thresh)
            best["scores"][bid] = inl
            if inl > best["inliers"]:
                best.update(board_id=bid, H=h, inliers=inl, tpl_shape=tuple(shape))
    best["mode"] = "fast" if fast else "full"
    if best["H"] is None or best["inliers"] < MIN_INLIERS_BOARD_ID:
        raise RuntimeError(f"Khong nhan dien duoc board (best={best['board_id']}, "
                           f"inliers={best['inliers']}). scores={best['scores']}")
    h = best["H"]
    try:
        h_inv = np.linalg.inv(h)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"H suy bien, khong nghich dao duoc: {e}")
    th, tw = best["tpl_shape"]
    warped = cv2.warpPerspective(image_bgr, h, (tw, th),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    # BUG-DAY3-01: BORDER_REPLICATE lap day phan canvas nam NGOAI anh dau vao
    # bang pixel bien -> residual gia (da do: 59% blob camera la gia, bg_std
    # 6.66 vs 2.52 tren vung hop le). Vung do khong co bang chung ve bo mach
    # nen thay bang Golden Template: residual = 0 dung nghia "khong thay gi".
    valid = valid_mask_from_h(image_bgr.shape[:2], h, (th, tw))
    tpl_gray = cache["templates_gray"][cache["board_ids"].index(best["board_id"])]
    if tpl_gray.shape != (th, tw):
        tpl_gray = cv2.resize(tpl_gray, (tw, th), interpolation=cv2.INTER_AREA)
    warped_filled = fill_out_of_fov(warped, valid, tpl_gray)
    best["H_inv"] = h_inv
    best["warped"] = warped_filled       # da dien vung ngoai FOV (dung cho diff)
    best["warped_raw"] = warped          # warp thuan (dung cho hien thi UI)
    best["valid_mask"] = valid
    return best


def project_points(pts_xy: np.ndarray, h_inv: np.ndarray) -> np.ndarray:
    """Chot 1: Template (x, y) -> Input goc qua H_inv (perspectiveTransform)."""
    arr = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, h_inv.astype(np.float64)).reshape(-1, 2)


def warp_to_template(image_bgr: np.ndarray, h: np.ndarray,
                     tpl_shape: tuple[int, int]) -> np.ndarray:
    th, tw = int(tpl_shape[0]), int(tpl_shape[1])
    return cv2.warpPerspective(image_bgr, h, (tw, th),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ngay 3: build ORB cache + align thu 1 anh.")
    p.add_argument("--build-cache", action="store_true")
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--image", type=Path, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.build_cache or args.image is None:
        build_cache(args.cache)
    if args.image is not None:
        cache = load_cache(args.cache)
        data = np.fromfile(str(args.image), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot decode image: {args.image}")
        r = identify_and_align(img, cache)
        print(f"[align] board={r['board_id']} inliers={r['inliers']} scores={r['scores']}")
        print(f"[align] H|H_inv ok, warped={r['warped'].shape} tpl={r['tpl_shape']}")


if __name__ == "__main__":
    main()
