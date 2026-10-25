"""Ngay 2 — Nhanh A: Semantic Defect Detector (YOLO11n Full-Res).

Hop dong du lieu (giong project cu, giu tuong thich voi evaluate.py):
  * predict(image): nhan image RGB uint8 (H, W, 3) HOAC BGR ndarray HOAC
    list ndarray (batch crops Nhanh B) HOAC duong dan anh.
  * tra ve list[dict]: {"bbox": [x1, y1, x2, y2], "class_id": int,
    "class_name": str, "conf": float}, sap xep theo conf giam dan.
    Voi batch input: tra ve list (moi phan tu la list dict cua 1 anh).

3 chot chan ap dung tai day:
  * Full-res: imgsz=3072 (hoac 3584), conf=0.20.
  * Chong Square Padding: bat buoc rect=True, half=True khi cuda trong moi
    lan goi model.predict() (giam latency 361ms -> ~160-190ms).
  * predict_batch(): gom TOP_K crops 640x640 goi 1 lan duy nhat (Chot 3);
    kem spatial_center_gating(R=80px) de triet FP ria crop (Chot Nhanh B).

Trong so mac dinh: assets/weights/yolo11n_base.pt (ban copy cua best.pt goc,
khong ghi de). Fine-tune Ngay 2 xuat file moi yolo11n_day2.pt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

CLASS_NAMES = {
    0: "open_circuit",
    1: "short",
    2: "mouse_bite",
    3: "spur",
    4: "spurious_copper",
    5: "missing_hole",
}
EXPECTED_NAMES = CLASS_NAMES

DEFAULT_IMGSZ = 3072
DEFAULT_CONF = 0.20
CENTER_GATE_RADIUS = 80.0  # px (plan.md 2.3 buoc 5)

# --- PORTABLE hardware adaptation (Step 2, release engineer) ---
# May dich co the yeu (VRAM 4GB) hoac khong co GPU. Tu dong ha bac thay vi crash.
CPU_FALLBACK_IMGSZ = 1920   # CPU: ep xuong de tranh crash + cham
LOW_VRAM_GB = 6.0           # duoi nguong nay: ha bac graceful
LOW_VRAM_IMGSZ = 2560       # VRAM 4-6GB: cap 2560 (giu rect+half)
TINY_VRAM_GB = 4.0          # duoi 4GB: xuong 1920
TINY_VRAM_IMGSZ = 1920
_OOM_FALLBACKS = (2560, 1920)  # thu lai khi OutOfMemoryError


def _cuda_total_gb() -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.get_device_properties(0).total_memory) / (1024 ** 3)
    except Exception:
        return 0.0


def resolve_device_and_imgsz(requested_device: str | None = None,
                             requested_imgsz: int | None = None
                             ) -> tuple[str, bool, int, str]:
    """Chon (device, half, imgsz, ly_do) phu hop phan cung hien tai.

    - Khong CUDA -> ("cpu", False, min(requested, 1920)).
      (half=True tren CPU gay crash Half precision not supported.)
    - CUDA VRAM < 6GB -> cap 2560; < 4GB -> cap 1920 (giu half=True).
    - Con lai giu nguyen yeu cau.
    """
    import torch
    want_imgsz = int(requested_imgsz) if requested_imgsz else DEFAULT_IMGSZ
    if requested_device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(requested_device).lower()
        if device not in ("cuda", "cpu"):
            raise ValueError("device must be 'cuda', 'cpu' or None (auto)")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
    if device == "cpu":
        if want_imgsz > CPU_FALLBACK_IMGSZ:
            return (device, False, CPU_FALLBACK_IMGSZ,
                    f"CPU fallback: imgsz {want_imgsz}->{CPU_FALLBACK_IMGSZ}, half=False")
        return (device, False, want_imgsz, "CPU, half=False")
    # cuda kha dung
    try:
        total_gb = _cuda_total_gb()
    except Exception:
        total_gb = 0.0
    if total_gb > 0.0 and total_gb < TINY_VRAM_GB and want_imgsz > TINY_VRAM_IMGSZ:
        return (device, True, TINY_VRAM_IMGSZ,
                f"VRAM {total_gb:.1f}GB<4GB: imgsz {want_imgsz}->{TINY_VRAM_IMGSZ}")
    if total_gb > 0.0 and total_gb < LOW_VRAM_GB and want_imgsz > LOW_VRAM_IMGSZ:
        return (device, True, LOW_VRAM_IMGSZ,
                f"VRAM {total_gb:.1f}GB<6GB: imgsz {want_imgsz}->{LOW_VRAM_IMGSZ}")
    return (device, True, want_imgsz, f"GPU {total_gb:.1f}GB, giu imgsz={want_imgsz}")


class YOLODetector:
    def __init__(self, model_path: str | Path, conf_threshold: float = DEFAULT_CONF,
                 imgsz: int | None = DEFAULT_IMGSZ, device: str | None = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.conf_threshold = float(conf_threshold)
        # PORTABLE: tu dong thich ung phan cung (CPU -> half=False + cap 1920;
        # VRAM <6GB -> 2560; <4GB -> 1920). Giu rect=True + half khi cuda.
        self.device, self.half, self.imgsz, self.adapt_note = \
            resolve_device_and_imgsz(device, imgsz)
        self.requested_imgsz = int(imgsz) if imgsz else DEFAULT_IMGSZ
        if self.adapt_note and ("fallback" in self.adapt_note
                                or "VRAM" in self.adapt_note):
            print(f"[YOLODetector] {self.adapt_note}")
        self.model = YOLO(str(self.model_path), task="detect")

    # -- tien xu ly input -------------------------------------------------
    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        """Nhan RGB uint8 (hop dong) -> BGR cho Ultralytics."""
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a np.ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (H, W, 3)")
        if image.dtype != np.uint8:
            raise ValueError("image dtype must be uint8")
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def _oom_retry(self, source, conf: float, imgsz: int):
        """Thu lai 2560/1920 khi OutOfMemoryError (giu rect=True, half=True)."""
        import torch
        torch.cuda.empty_cache()
        last: Exception | None = None
        for fb in _OOM_FALLBACKS:
            if fb >= int(imgsz):
                continue
            try:
                print(f"[YOLODetector] OOM o imgsz={imgsz}, thu lai {fb} "
                      f"(rect=True, half=True).")
                return self.model.predict(
                    source=source, conf=float(conf), imgsz=int(fb),
                    device=self.device, verbose=False,
                    rect=True, half=self.half)
            except torch.cuda.OutOfMemoryError as e:
                last = e
                torch.cuda.empty_cache()
                continue
        raise last if last is not None else RuntimeError("OOM retry failed")

    # -- single image ------------------------------------------------------
    def predict(self, image: np.ndarray | str | Path,
                conf: float | None = None,
                imgsz: int | None = None) -> list[dict[str, Any]]:
        """Suy luan 1 anh full-board. Bat buoc rect=True, half=self.half (Chot 3)."""
        if isinstance(image, (str, Path)):
            data = np.fromfile(str(image), dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"Cannot decode image: {image}")
        else:
            bgr = self._to_bgr(image)
        import torch
        try:
            # CHOT: chong Square Padding + half khi cuda, goi 1 lan duy nhat.
            results = self.model.predict(
                source=bgr,
                conf=self.conf_threshold if conf is None else float(conf),
                imgsz=self.imgsz if imgsz is None else int(imgsz),
                device=self.device, verbose=False,
                rect=True, half=self.half)
        except torch.cuda.OutOfMemoryError:
            if self.device != "cuda":
                raise
            results = self._oom_retry(
                bgr, self.conf_threshold if conf is None else float(conf),
                self.imgsz if imgsz is None else int(imgsz))
        return self._parse_results(results)

    # -- batch crops (Nhanh B ROI verification, Chot 3) ---------------------
    def predict_batch(self, images: list[np.ndarray],
                      conf: float | None = None,
                      imgsz: int = 640) -> list[list[dict[str, Any]]]:
        """Goi GPU 1 lan duy nhat cho N crops 640x640. Khong dung for-predict."""
        if not images:
            return []
        bgrs = [self._to_bgr(im) for im in images]
        import torch
        try:
            # CHOT: batch 1 lan duy nhat, rect=True, half=self.half.
            results = self.model.predict(
                source=bgrs,
                conf=self.conf_threshold if conf is None else float(conf),
                imgsz=int(imgsz), device=self.device, verbose=False,
                rect=True, half=self.half)
        except torch.cuda.OutOfMemoryError:
            if self.device != "cuda":
                raise
            results = self._oom_retry(
                bgrs, self.conf_threshold if conf is None else float(conf),
                int(imgsz))
        return [self._parse_one(r) for r in results]

    # -- spatial center gating (Nhanh B buoc 5) ------------------------------
    @staticmethod
    def _box_iou(a: list[float], b: list[float]) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0.0 else 0.0

    @staticmethod
    def spatial_center_gating(dets: list[dict[str, Any]],
                              center_xy: tuple[float, float],
                              radius: float = CENTER_GATE_RADIUS,
                              crop_origin: tuple[float, float] = (0.0, 0.0),
                              require_overlap: bool = True,
                              crop_size: int = 640,
                              diff_box: list[float] | None = None) -> list[dict[str, Any]]:
        """Giu box neu tam box trong R=80px quanh tam diff HOAC IoU>0.

        dets: bbox theo toa do crop; center_xy/crop_origin theo toa do anh goc.
        Chuyen tam box ve toa do anh goc truoc khi so sanh (dong bo he quy
        chieu — tinh than Chot 1 ap cho Nhanh B).

        require_overlap=True (mac dinh): chap nhan them box co tam ngoai R
        nhung giao cat vung diff (IoU>0 voi o vuong 2R quanh tam diff, hoac
        voi diff_box neu duoc truyen). require_overlap=False: chi xet R.
        """
        cx, cy = float(center_xy[0]), float(center_xy[1])
        ox, oy = float(crop_origin[0]), float(crop_origin[1])
        r = float(radius)
        if diff_box is None:
            diff_win = [cx - r, cy - r, cx + r, cy + r]
        else:
            diff_win = [float(v) for v in diff_box]
        kept: list[dict[str, Any]] = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            gx1, gy1, gx2, gy2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy
            bx = (gx1 + gx2) / 2.0
            by = (gy1 + gy2) / 2.0
            dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
            if dist <= r:
                kept.append(d)
                continue
            if require_overlap and YOLODetector._box_iou(
                    [gx1, gy1, gx2, gy2], diff_win) > 0.0:
                kept.append(d)
        return kept

    # -- parse ---------------------------------------------------------------
    def _parse_one(self, r: Any) -> list[dict[str, Any]]:
        dets: list[dict[str, Any]] = []
        if r.boxes is None or len(r.boxes) == 0:
            return dets
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            dets.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": cls_id,
                "class_name": str(CLASS_NAMES.get(
                    cls_id, self.model.names.get(cls_id, str(cls_id)))),
                "conf": conf,
            })
        dets.sort(key=lambda d: d["conf"], reverse=True)
        return dets

    def _parse_results(self, results: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in results:
            out.extend(self._parse_one(r))
        out.sort(key=lambda d: d["conf"], reverse=True)
        return out
