"""Verifier probe 2: open-set lon co bi AUTO_CALIB_TOL nuot khong?

Rule 3b (plan 2.4): YOLO khong bat duoc (S<0.20) NHUNG Z_diff>=3.5 VA
Area>=100 -> REVIEW/UNKNOWN_DEFECT. Neu AUTO_CALIB_TOL (tu hieu chuan tu
chinh anh dang xet) nang nguong theo dien tich loi thi open-set se bi vo hieu.

Chay: <venv>/python.exe tests/probe_day3_openset_large.py
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
    rows = []
    for bid in ("01", "04"):
        tpl = read_gray(LOCAL_TEMPLATES / f"{bid}.JPG")
        base = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
        H, W = tpl.shape
        total = H * W
        # open-set: vet o hoa chat / chay mach — mang lon, tuong phan vua
        for frac in (0.02, 0.05, 0.10, 0.20, 0.35):
            side = int((total * frac) ** 0.5)
            side = min(side, min(H, W) - 10)
            for delta in (60, 100):
                img = base.copy()
                cy, cx = H // 2, W // 2
                y0, y1 = cy - side // 2, cy + side // 2
                x0, x1 = cx - side // 2, cx + side // 2
                patch = img[y0:y1, x0:x1].astype(np.int16) + delta
                img[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)

                r = ba.identify_and_align(img, cache, fast=True)
                t = cache["templates_gray"][cache["board_ids"].index(r["board_id"])]
                if t.shape != r["warped"].shape[:2]:
                    t = cv2.resize(t, (r["warped"].shape[1], r["warped"].shape[0]),
                                   interpolation=cv2.INTER_AREA)
                out = td.run_branch_b(r["warped"], t, valid_mask=r["valid_mask"])
                out_raw = td.run_branch_b(r["warped"], t, valid_mask=r["valid_mask"],
                                          auto_calib_tol=False)
                c = ba.project_points(np.float32([[cx, cy]]), r["H_inv"])[0]

                def best_hit(blobs):
                    best = (False, 0.0, 0.0)
                    for b in blobs:
                        bx = ba.project_points(np.float32([b["centroid_tpl"]]),
                                               r["H_inv"])[0]
                        if abs(bx[0] - c[0]) <= side / 2 + 10 and \
                           abs(bx[1] - c[1]) <= side / 2 + 10:
                            if b["z_diff"] > best[1]:
                                best = (True, b["z_diff"], b["area"])
                    return best

                h_on, z_on, a_on = best_hit(out["blobs"])
                h_off, z_off, a_off = best_hit(out_raw["blobs"])
                # rule 3b: Z>=3.5 AND Area>=100
                trig_on = bool(h_on and z_on >= 3.5 and a_on >= 100)
                trig_off = bool(h_off and z_off >= 3.5 and a_off >= 100)
                rows.append({"board": bid, "area_frac": frac, "side": side,
                             "delta": delta,
                             "auto": {"hit": bool(h_on), "z": round(float(z_on), 2),
                                      "area": float(a_on), "rule3b": trig_on},
                             "no_calib": {"hit": bool(h_off), "z": round(float(z_off), 2),
                                          "area": float(a_off), "rule3b": trig_off}})
                print(f"{bid} frac={frac:4.2f} side={side:3d} d={delta:3d} | "
                      f"auto hit={h_on!s:5s} Z={z_on:6.2f} A={a_on:7.0f} 3b={trig_on!s:5s} | "
                      f"raw hit={h_off!s:5s} Z={z_off:6.2f} A={a_off:7.0f} 3b={trig_off!s:5s}")

    n = len(rows)
    s = {"n_cases": n,
         "auto_hits": sum(r["auto"]["hit"] for r in rows),
         "auto_rule3b": sum(r["auto"]["rule3b"] for r in rows),
         "no_calib_hits": sum(r["no_calib"]["hit"] for r in rows),
         "no_calib_rule3b": sum(r["no_calib"]["rule3b"] for r in rows),
         "note": "open-set mang lon (2-35% dien tich) tren template 01/04",
         "rows": rows}
    out = assert_writable_ok(NEW_PROJECT_DIR / "results" / "probe_day3_openset_large.json")
    out.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[probe] auto: hit {s['auto_hits']}/{n} rule3b {s['auto_rule3b']}/{n} | "
          f"raw: hit {s['no_calib_hits']}/{n} rule3b {s['no_calib_rule3b']}/{n}")
    print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()
