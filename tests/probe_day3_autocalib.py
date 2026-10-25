"""Verifier probe: AUTO_CALIB_TOL co 'nuot' loi that khong?

Kiem dinh doc lap (ghi results/). Khong sua src/.
Chay: <venv>/python.exe tests/probe_day3_autocalib.py
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
import src.reference.tolerance_diff as td  # noqa: E402


def read_gray(p):
    d = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(d, cv2.IMREAD_GRAYSCALE)


def main():
    cache = ba.load_cache()
    tpl = read_gray(LOCAL_TEMPLATES / "01.JPG")
    # anh "good" = template -> diff ~ 0; ta tiem loi nhan tao kich thuoc tang dan
    base = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
    rows = []
    for size in (20, 40, 60, 80, 120, 160, 200, 260):
        for delta in (40, 80, 120):
            img = base.copy()
            # dat loi o vung co cau truc (giua anh)
            cy, cx = tpl.shape[0] // 2, tpl.shape[1] // 2
            y0, y1 = cy - size // 2, cy + size // 2
            x0, x1 = cx - size // 2, cx + size // 2
            patch = img[y0:y1, x0:x1].astype(np.int16) + delta
            img[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)

            r = ba.identify_and_align(img, cache, fast=True)
            t = cache["templates_gray"][cache["board_ids"].index(r["board_id"])]
            if t.shape != r["warped"].shape[:2]:
                t = cv2.resize(t, (r["warped"].shape[1], r["warped"].shape[0]),
                               interpolation=cv2.INTER_AREA)

            out_on = td.run_branch_b(r["warped"], t, valid_mask=r["valid_mask"])
            out_off = td.run_branch_b(r["warped"], t, valid_mask=r["valid_mask"],
                                      auto_calib_tol=False)
            c = ba.project_points(np.float32([[cx, cy]]), r["H_inv"])[0]

            def hit(blobs):
                for b in blobs:
                    bx = ba.project_points(
                        np.float32([b["centroid_tpl"]]), r["H_inv"])[0]
                    if abs(bx[0] - c[0]) <= size / 2 + 5 and \
                       abs(bx[1] - c[1]) <= size / 2 + 5:
                        return True, b["z_diff"]
                return False, 0.0

            h_on, z_on = hit(out_on["blobs"])
            h_off, z_off = hit(out_off["blobs"])
            rows.append({"size": size, "delta": delta,
                         "auto_calib": {"hit": bool(h_on), "z": round(float(z_on), 2),
                                        "n_blob": len(out_on["blobs"])},
                         "no_calib": {"hit": bool(h_off), "z": round(float(z_off), 2),
                                      "n_blob": len(out_off["blobs"])}})
            print(f"size={size:3d} delta={delta:3d} | auto {h_on!s:5s} Z={z_on:6.2f} "
                  f"n={len(out_on['blobs']):2d} | noCalib {h_off!s:5s} Z={z_off:6.2f} "
                  f"n={len(out_off['blobs']):2d}")

    n_on = sum(r["auto_calib"]["hit"] for r in rows)
    n_off = sum(r["no_calib"]["hit"] for r in rows)
    summary = {"n_cases": len(rows), "auto_calib_hits": n_on,
               "no_calib_hits": n_off,
               "note": "anh 01.JPG + loi nhan tao vuong size x size, delta muc xam",
               "rows": rows}
    out = assert_writable_ok(NEW_PROJECT_DIR / "results" / "probe_day3_autocalib.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[probe] auto_calib {n_on}/{len(rows)}  no_calib {n_off}/{len(rows)}")
    print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()
