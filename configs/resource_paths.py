#!/usr/bin/env python3
"""Central resource registry — PORTABLE (zero hardcoded drive paths).

RULE:
  Every path is derived from PROJECT_ROOT
  (Path(__file__).resolve().parent.parent) so the project runs from ANY
  folder/drive on a new Windows machine. NEVER hardcode a drive letter
  in runtime code.

  - PROJECT_ROOT / NEW_PROJECT_DIR : this project (writable).
  - LEGACY_* : sibling reference dirs on the OLD dev machine, computed as
    PROJECT_ROOT.parent / "<sibling>" (no drive literal). They may NOT exist
    on a new machine — runtime code MUST fallback to in-project
    assets/ + data/demo_and_test_samples/ and never crash.
  - Writable local copies live under assets/ (weights, templates).

Portable lookup priority for PKU data:
  LOCAL_PKU_ROOT (full, optional) -> DEMO_ROOT (bundled 01/04) -> LEGACY
  (old machine only) -> Kaggle download.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys

# --------------------------------------------------------------------------
# Project roots (PORTABLE — no drive letters)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEW_PROJECT_DIR = PROJECT_ROOT  # alias for backward-compat imports

# --- Legacy siblings (DEPRECATED, old-machine only, computed relatively) ---
# On the old dev machine these resolve to the original folders; on a new
# machine they simply do not exist and code must use the bundled fallback.
LEGACY_PROJECT_DIR = PROJECT_ROOT.parent / "PCB_Anomaly_Inspector_Portable"
GITHUB_BENCH_DIR = PROJECT_ROOT.parent / "Test_github"

# --------------------------------------------------------------------------
# Weights / templates (inside project — always available offline)
# --------------------------------------------------------------------------

# 1. YOLO11n weights — local copies used for training/inference.
LOCAL_BASE_PT = PROJECT_ROOT / "assets" / "weights" / "yolo11n_base.pt"
LOCAL_DAY2_PT = PROJECT_ROOT / "assets" / "weights" / "yolo11n_day2.pt"


def default_weights() -> Path:
    """Best fine-tuned weights if present, else base (portable default)."""
    if LOCAL_DAY2_PT.exists():
        return LOCAL_DAY2_PT
    return LOCAL_BASE_PT


# Backward-compat alias (old code imports LOCAL_BASE_PT directly).
DEFAULT_WEIGHTS = LOCAL_DAY2_PT  # resolved at runtime via default_weights()

# 2. PKU-Market-PCB raw data + Golden Templates (10 files: 01, 04-12).
LEGACY_PKU_ROOT = LEGACY_PROJECT_DIR / "data" / "raw" / "pku"
LEGACY_PCB_USED = LEGACY_PKU_ROOT / "PCB_USED"
LEGACY_PKU_IMAGES = LEGACY_PKU_ROOT / "images"      # 6 class subfolders
LEGACY_PKU_ANNOT = LEGACY_PKU_ROOT / "Annotations"
LOCAL_TEMPLATES = PROJECT_ROOT / "assets" / "templates"
TEMPLATE_FILES = ["01.JPG", "04.JPG", "05.JPG", "06.JPG", "07.JPG", "08.JPG",
                  "09.JPG", "10.JPG", "11.JPG", "12.JPG"]

# Unified 10k pool (old-machine sibling, optional).
PCBU10K_ROOT = PROJECT_ROOT.parent / "pcb_unified_10k"

# Target location for the full PKU-Market-PCB download (Kaggle fallback).
LOCAL_PKU_ROOT = PROJECT_ROOT / "data" / "PKU_Market_PCB"

# Bundled demo/test set (INSIDE project — ships in portable zip).
# Mirrors PKU layout: images/<class>/*.jpg + Annotations/<class>/*.xml
# for Zero-defect-training Boards 01/04 (36 defect + GOOD via templates).
DEMO_ROOT = PROJECT_ROOT / "data" / "demo_and_test_samples"
DEMO_IMAGES_ROOT = DEMO_ROOT / "images"
DEMO_ANNOT_ROOT = DEMO_ROOT / "Annotations"

# 3. Legacy docs & experiment numbers (old-machine reference only).
LEGACY_AUDIT_MD = LEGACY_PROJECT_DIR / "docs" / "AUDIT_vs_TDDNet.md"
LEGACY_PATCHCORE_CKPT = LEGACY_PROJECT_DIR / "models" / "patchcore" / "model.ckpt"
LEGACY_PATCHCORE_ONNX = LEGACY_PROJECT_DIR / "models" / "onnx" / "patchcore.onnx"
# comparison.json lives OUTSIDE the legacy project (sibling folder).
HYBRID_COMPARISON_JSON = (
    PROJECT_ROOT.parent
    / "hybird_test_3branch-20260920T081632Z-1-001"
    / "hybird_test_3branch" / "results" / "comparison.json"
)
# temp/check20*/check21* registration scripts: NOT FOUND during scan.
LEGACY_CHECK_SCRIPTS: list = []
LEGACY_PKU_IMAGES_ALIAS = LEGACY_PKU_IMAGES
LEGACY_PKU_ANNOT_ALIAS = LEGACY_PKU_ANNOT

# 4. Benchmark repos (old-machine reference only).
TDD_PORTABLE_DIR = GITHUB_BENCH_DIR / "Tiny-Defect-Detection-for-PCB"
FPN_DIR = GITHUB_BENCH_DIR / "FPN_Res101_0117_OHEM"
TDD_ZIP = GITHUB_BENCH_DIR / "TDD-Net-PCB-portable.zip"

# 5. Python environment: portable venv lives at PROJECT_ROOT/.venv
# (created by 1_setup_env.bat on the new machine). No absolute venv path.
PORTABLE_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
LEGACY_VENV_PYTHON = LEGACY_PROJECT_DIR / ".venv" / "Scripts" / "python.exe"


# --------------------------------------------------------------------------
# Read-only guard
# --------------------------------------------------------------------------
def _is_inside(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except Exception:
        return False


def assert_writable_ok(path: Path) -> Path:
    """Block writes outside the new project directory."""
    p = Path(path)
    if _is_inside(p, PROJECT_ROOT):
        return p
    raise PermissionError(
        "[STRICT READ-ONLY] Refusing to write outside the new project: %s\n"
        "Allowed root: %s" % (p, PROJECT_ROOT)
    )


def find_pku_root() -> Path | None:
    """Portable PKU lookup: local full -> bundled demo -> legacy (old PC).

    Returns None when nothing is available (caller must handle gracefully).
    """
    for cand in (LOCAL_PKU_ROOT, DEMO_ROOT, LEGACY_PKU_ROOT):
        try:
            if Path(cand).is_dir() and (Path(cand) / "images").is_dir():
                return Path(cand)
        except Exception:
            continue
    return None


def ensure_pku_dataset() -> Path:
    """Return a usable PKU root (portable priority).

    Priority: local data/PKU_Market_PCB -> bundled
    data/demo_and_test_samples -> legacy read-only pku -> Kaggle
    download into the new project data/ folder.
    NEVER writes into the legacy project.
    """
    if LOCAL_PKU_ROOT.is_dir():
        return LOCAL_PKU_ROOT
    if DEMO_ROOT.is_dir() and (DEMO_ROOT / "images").is_dir():
        return DEMO_ROOT
    if LEGACY_PKU_ROOT.is_dir():
        return LEGACY_PKU_ROOT
    try:
        import kagglehub  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Thieu dataset PKU (tim LOCAL_PKU_ROOT, DEMO_ROOT, legacy deu "
            "khong thay) va thieu kagglehub (%s). Cai dat: pip install "
            "kagglehub" % e
        )
    dl = Path(kagglehub.dataset_download("akhatova/pcb-defects"))
    target = assert_writable_ok(LOCAL_PKU_ROOT)
    target.mkdir(parents=True, exist_ok=True)
    # Copy (one-way) into the new project; never touch legacy dirs.
    import shutil
    for item in dl.iterdir():
        dest = target / item.name
        if dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    return target


def resource_table() -> str:
    """Printable summary for the user to verify."""
    rows = [
        ("Project root (PORTABLE)", str(PROJECT_ROOT)),
        ("YOLO base (viet duoc)", str(LOCAL_BASE_PT)),
        ("YOLO day2 fine-tune", str(LOCAL_DAY2_PT)),
        ("Templates copy (viet duoc)", str(LOCAL_TEMPLATES)),
        ("Demo/test bundled", str(DEMO_ROOT)),
        ("PKU local (optional)", str(LOCAL_PKU_ROOT)),
        ("PKU legacy (old PC only)", str(LEGACY_PKU_ROOT)),
        ("Legacy project (old PC)", str(LEGACY_PROJECT_DIR)),
        ("Github bench (old PC)", str(GITHUB_BENCH_DIR)),
        ("Portable venv", str(PORTABLE_VENV_PYTHON)),
    ]
    return "\n".join("%-32s %s" % r for r in rows)


if __name__ == "__main__":
    print(resource_table())
    if "--check" in sys.argv:
        portable = [
            ("base", LOCAL_BASE_PT), ("day2", LOCAL_DAY2_PT),
            ("templates", LOCAL_TEMPLATES), ("demo", DEMO_ROOT),
        ]
        missing = [n for n, p in portable if not Path(p).exists()]
        print("\nPortable missing: %s" % (missing if missing else "none"))
        print("PKU root: %s" % (find_pku_root() or "NONE"))
