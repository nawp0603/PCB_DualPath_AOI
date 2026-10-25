"""Quy tac quyet dinh 4 muc: FAIL / REWORK / REVIEW / PASS (plan.md Phan 2.4).

Moi evidence item (output cua Box Fusion / ensemble) la dict voi cac truong:
  class_id (int), class_name (str), conf (float | None),   # S_final (0..1)
  source ("yolo"|"branchb"|"fused"), bbox (list[float]),
  z_diff (float | None), area (float | None).

Uu tien: FAIL > REWORK > REVIEW > PASS.
KHONG dung dieu kien OR nhan "anom_z >= 2.0 AND yolo_conf_max >= 0.30"
(neu YOLO bo sot, he thong van chon PASS). Hai truong hop REVIEW duoc tach
doc lap nhu bang quy tac 2.4 (3a cuu recall 6 lop; 3b open-set thuat an).
"""

from __future__ import annotations

from typing import Any

# --- Nhom lop theo plan.md Phan 2.4 ----------------------------------------
FAIL_CLASSES = {0, 1, 5}        # open_circuit, short, missing_hole
REWORK_CLASSES = {2, 3, 4}      # mouse_bite, spur, spurious_copper
ALL_CLASSES = FAIL_CLASSES | REWORK_CLASSES

# --- Nguong theo plan.md Phan 2.4 ------------------------------------------
CONF_HIGH = 0.55        # muc 1: S_final >= 0.55 -> FAIL / HIGH
CONF_MEDIUM = 0.35      # muc 2: S_final >= 0.35 -> REWORK / MEDIUM
CONF_LOW_LO = 0.20      # muc 3a: 0.20 <= S_final < 0.35 (du doan thap)
Z_REVIEW = 2.0          # muc 3a: Z_diff >= 2.0 trung vung Nhanh B -> cuu recall
Z_OPENSET = 3.5         # muc 3b: Z_diff >= 3.5 va Area >= 100 -> open-set
AREA_OPENSET = 100.0    # px2, dieu kien kem theo Z_OPENSET


def decide(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Chuyen danh sach evidence da fuse ve ket luan 4 muc.

    Tra ve:
      {"level": int, "conclusion": str, "verdict": str,
       "reasons": list[str], "details": list[dict]}
    """
    reasons: list[str] = []
    details: list[dict[str, Any]] = []

    def note(level: int, conclusion: str, verdict: str, why: str,
             item: dict[str, Any] | None = None) -> None:
        reasons.append(why)
        if item is not None:
            details.append({
                "level": level, "conclusion": conclusion, "verdict": verdict,
                "why": why, "class_id": item.get("class_id"),
                "class_name": item.get("class_name"), "conf": item.get("conf"),
                "source": item.get("source"), "z_diff": item.get("z_diff"),
                "area": item.get("area"), "bbox": item.get("bbox")})

    def _has(pred) -> list[dict[str, Any]]:
        return [it for it in items if pred(it)]

    # --- Muc 1: FAIL / HIGH ------------------------------------------------
    fail_items = _has(lambda it: (
        it.get("class_id") in FAIL_CLASSES
        and it.get("conf") is not None
        and it["conf"] >= CONF_HIGH))
    if fail_items:
        for it in fail_items:
            note(1, "FAIL", "HIGH",
                 f"'{it['class_name']}' S_final={it['conf']:.3f} >= {CONF_HIGH}", it)
        return _pack(1, "FAIL", "HIGH", reasons, details)

    # --- Muc 2: REWORK / MEDIUM --------------------------------------------
    rework_items = _has(lambda it: (
        it.get("class_id") in REWORK_CLASSES
        and it.get("conf") is not None
        and it["conf"] >= CONF_MEDIUM))
    if rework_items:
        for it in rework_items:
            note(2, "REWORK", "MEDIUM",
                 f"'{it['class_name']}' S_final={it['conf']:.3f} >= {CONF_MEDIUM}", it)
        return _pack(2, "REWORK", "MEDIUM", reasons, details)

    # --- Muc 3a: REVIEW cuu recall (6 lop, conf thap, trung Nhanh B Z>=2.0) -
    review3a = _has(lambda it: (
        it.get("class_id") in ALL_CLASSES
        and it.get("conf") is not None
        and CONF_LOW_LO <= it["conf"] < CONF_MEDIUM
        and it.get("z_diff") is not None
        and it["z_diff"] >= Z_REVIEW))
    if review3a:
        # plan: "Nang len REWORK / MEDIUM hoac REVIEW" -> xet REVIEW o day.
        for it in review3a:
            note(3, "REVIEW", "RECALL_RESCUE",
                 f"'{it['class_name']}' S_final={it['conf']:.3f} in [0.20,0.35) "
                 f"va Z_diff={it['z_diff']:.2f} >= {Z_REVIEW}", it)
        return _pack(3, "REVIEW", "RECALL_RESCUE", reasons, details)

    # --- Muc 3b: REVIEW open-set (YOLO bo sot, Nhanh B phat hien vung lon) --
    review3b = _has(lambda it: (
        it.get("source") == "branchb"
        and it.get("z_diff") is not None
        and it["z_diff"] >= Z_OPENSET
        and it.get("area") is not None
        and it["area"] >= AREA_OPENSET))
    if review3b:
        for it in review3b:
            note(3, "REVIEW", "UNKNOWN_DEFECT",
                 f"Nhanh B Z_diff={it['z_diff']:.2f} >= {Z_OPENSET} "
                 f"va Area={it['area']:.0f} >= {AREA_OPENSET} (YOLO chua phat hien)", it)
        return _pack(3, "REVIEW", "UNKNOWN_DEFECT", reasons, details)

    # --- Muc 4: PASS / NONE -------------------------------------------------
    for it in items:
        note(4, "PASS", "NONE",
             "khong thoa man dieu kien FAIL/REWORK/REVIEW nao", it)
    return _pack(4, "PASS", "NONE", reasons, details)


def _pack(level: int, conclusion: str, verdict: str,
          reasons: list[str], details: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "level": level,
        "conclusion": conclusion,
        "verdict": verdict,
        "reasons": reasons,
        "details": details,
    }


def summary_text(res: dict[str, Any]) -> str:
    return f"{res['conclusion']} / {res['verdict']} (level {res['level']})"
