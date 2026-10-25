#!/usr/bin/env python3
"""
Standalone Script 2: Tolerance-Aware Difference Heatmap & Safe Z-Score Visualization.
Generates:
  1. `aoi_final_plan/thesis_artifacts/figures/fig2_tolerance_diff_heatmap.png`
  2. `aoi_final_plan/thesis_artifacts/figures/fig2b_three_panel_aoi_ui.png`

Demonstrates:
  - Why Naive AbsDiff |W_corr - T| creates sub-pixel boundary halos along copper traces.
  - How Morphological Envelope [T_min, T_max] (tau=2-3 px) + Structure Mask + Auto-Calib
    Tolerance Map isolates genuine morphological defects.
  - Safe Z-Score with sigma_floor=5.0 (Chot 2) preventing false alarms on clean GOOD boards.
"""

from __future__ import annotations

import sys
from pathlib import Path

THESIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THESIS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THESIS_DIR))

import cv2
import numpy as np
from PIL import Image, ImageDraw

from configs.resource_paths import LOCAL_BASE_PT, LOCAL_TEMPLATES, ensure_pku_dataset
from scripts.evaluate import collect_defect_items, perturb_image_and_boxes, read_bgr
import src.reference.board_id_and_align as ba
import src.reference.tolerance_diff as td
from src.models.yolo_detector import YOLODetector
from src.models.ensemble import fuse_and_decide
from fig_utils import (
    add_header_and_footer,
    bgr_to_pil,
    colorize_heatmap,
    crop_with_border,
    draw_rounded_box,
    get_font,
    make_colorbar_vertical,
    save_figure,
)


def generate_fig2() -> tuple[Path, Path]:
    pku_root = ensure_pku_dataset()
    items = collect_defect_items(pku_root, boards={"01", "04"})
    # Choose a board 04 or 01 image with multiple clear defects (e.g. spurious_copper or short)
    target_item = next(
        (it for it in items if "spurious_copper" in Path(it["path"]).name.lower() and it["board"] == "01"),
        items[0],
    )
    raw_bgr = read_bgr(target_item["path"])
    gt_boxes = [b for _, b in target_item["gt"]]
    cam_bgr, warped_gts = perturb_image_and_boxes(raw_bgr, gt_boxes, seed=20261029)

    cache = ba.load_cache()
    align_res = ba.identify_and_align(cam_bgr, cache, fast=True)
    bid = align_res["board_id"]
    tpl_gray = ba._read_gray(LOCAL_TEMPLATES / f"{bid}.JPG")

    # Run Branch B on perturbed defect image
    out_def = td.run_branch_b(
        align_res["warped"], tpl_gray, valid_mask=align_res["valid_mask"]
    )
    corr_f32 = out_def["warped_corr"]
    tpl_f32 = out_def["tpl_gray"]
    struct_mask = out_def["structure_mask"]
    diff_tol = out_def["diff_map"]
    blobs = out_def["blobs"]

    # Naive pixel diff |W_corr - T| after alignment & photometric correction
    diff_naive = np.abs(corr_f32 - tpl_f32)

    # Also run Branch B on a Clean GOOD board (01.JPG) under Standard & Camera Perturbation
    good_bgr = read_bgr(LOCAL_TEMPLATES / "01.JPG")
    good_cam_bgr, _ = perturb_image_and_boxes(good_bgr, [], seed=20261023 + 101)
    align_good_cam = ba.identify_and_align(good_cam_bgr, cache, fast=True)
    out_good_cam = td.run_branch_b(
        align_good_cam["warped"], tpl_gray, valid_mask=align_good_cam["valid_mask"]
    )

    cx_t, cy_t = blobs[0]["centroid_tpl"]
    top_z = blobs[0]["z_diff"]
    top_area = blobs[0]["area"]

    # =========================================================================
    # FIGURE 2A: Tolerance-Aware Diff & Safe Z-Score Mathematical Progression
    # =========================================================================
    W_CANVAS, H_CANVAS = 2400, 1420
    canvas = Image.new("RGB", (W_CANVAS, H_CANVAS), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)

    add_header_and_footer(
        canvas,
        "HÌNH 2: BẢN ĐỒ NHIỆT DUNG SAI HÌNH THÁI (TOLERANCE-AWARE DIFF) & CHỐT CHẶN SAFE Z-SCORE",
        f"Mẫu kiểm thử: {Path(target_item['path']).name} (Zero-defect-training Board {bid}) | Đối chiếu Phép trừ tuyệt đối vs. Khoảng bao [T_min, T_max] (τ=2px) & σ_floor=5.0",
        "Nguồn dữ liệu: src/reference/tolerance_diff.py + results/branchB_day3.json (Recall Nhánh B: 99/105 Standard, 98/105 Camera; FAR GOOD = 0/4)",
    )

    half_roi = 210
    patch_sz = (410, 410)

    # Prepare 4 zoomed ROI panels across the top row
    corr_u8 = np.clip(np.round(corr_f32), 0, 255).astype(np.uint8)
    corr_bgr = cv2.cvtColor(corr_u8, cv2.COLOR_GRAY2BGR)

    # Structure mask visualization
    mask_vis = cv2.cvtColor(tpl_gray, cv2.COLOR_GRAY2BGR)
    green_tint = np.zeros_like(mask_vis)
    green_tint[..., 1] = 210
    mask_vis[struct_mask] = cv2.addWeighted(mask_vis[struct_mask], 0.55, green_tint[struct_mask], 0.45, 0)

    hm_naive = colorize_heatmap(diff_naive, 0.0, 45.0)
    hm_tol = colorize_heatmap(diff_tol, 0.0, 12.0)
    # Draw detected blob bounding box & centroid on hm_tol
    bx1, by1, bx2, by2 = blobs[0]["bbox_tpl"]
    cv2.rectangle(hm_tol, (bx1 - 6, by1 - 6), (bx2 + 6, by2 + 6), (255, 255, 255), 3)

    top_panels = [
        (
            "(a) Chuẩn hóa Quang trắc (W_corr)",
            "Khử lệch sáng tần số thấp (Cửa sổ 151×151 px)",
            crop_with_border(corr_bgr, cx_t, cy_t, half_roi, (3, 105, 161), patch_sz),
            "W_corr = W · clip(T_smooth / W_smooth, 0.5, 2.0)\nGiữ nguyên tín hiệu lỗi cục bộ (không bị xóa như Z-score 31px).",
            (3, 105, 161),
        ),
        (
            "(b) Mặt nạ Cấu trúc Hợp nhất (Union Mask)",
            "dilate_7px(Otsu Copper UNION Sobel Edge > 30)",
            crop_with_border(mask_vis, cx_t, cy_t, half_roi, (21, 128, 61), patch_sz),
            "Bao phủ cả đường đồng Template lẫn dị vật mới trên nền\nphíp (spurious_copper / short) nằm sát mạch ~1 px.",
            (21, 128, 61),
        ),
        (
            "(c) Phép trừ Ngây thơ |W_corr - T| (τ = 0)",
            "Nhiễu viền sub-pixel dọc theo mọi cạnh đường mạch",
            crop_with_border(hm_naive, cx_t, cy_t, half_roi, (185, 28, 28), patch_sz),
            f"Sai số nội suy 1–2 px sau phép warp làm toàn bộ viền pad\nsáng rực (Mean={diff_naive.mean():.2f}), gây báo giả diện rộng.",
            (185, 28, 28),
        ),
        (
            "(d) Tolerance-Aware Diff (τ = 2 px) + Auto-Calib",
            "Khoảng bao [T_min, T_max] (Erode/Dilate) chia Tol_Map",
            crop_with_border(hm_tol, cx_t, cy_t, half_roi, (217, 119, 6), patch_sz),
            f"Triệt tiêu hoàn toàn viền giả 2 px quanh đường đồng!\nCô lập đúng blob lỗi: Z_diff = {top_z:.2f}, Area = {top_area:.0f} px².",
            (217, 119, 6),
        ),
    ]

    card_w, card_h = 520, 585
    top_y = 112
    for i, (ptitle, psub, pimg, pstat, accent) in enumerate(top_panels):
        x = 30 + i * 542
        draw_rounded_box(draw, [x, top_y, x + card_w, top_y + card_h],
                         fill=(255, 255, 255), outline=(203, 213, 225), width=2, radius=12)
        draw.rectangle([x, top_y, x + card_w, top_y + 8], fill=accent)
        draw.text((x + 18, top_y + 16), ptitle, fill=(15, 23, 42), font=get_font(19, bold=True))
        draw.text((x + 18, top_y + 44), psub, fill=(71, 85, 105), font=get_font(15, italic=True))
        canvas.paste(bgr_to_pil(pimg), (x + 55, top_y + 76))
        draw.text((x + 18, top_y + 500), pstat, fill=(30, 41, 59), font=get_font(15))

    # Add vertical colorbar on the right of panel (d)
    cbar = make_colorbar_vertical(410, 28, 0.0, 12.0, "Z / Res")
    canvas.paste(cbar, (2205, top_y + 56))

    # Bottom Section: Left = Full-board Heatmap + Top-K Blobs; Right = Chot 2 (Safe Z-Score on GOOD board)
    bot_y = 720
    bot_h = 640

    # Left Box: Full-Board Heatmap & Connected Components Top-K
    draw_rounded_box(draw, [30, bot_y, 1180, bot_y + bot_h],
                     fill=(255, 255, 255), outline=(203, 213, 225), width=2, radius=12)
    draw.rectangle([30, bot_y, 1180, bot_y + 46], fill=(18, 35, 64))
    draw.text(
        (50, bot_y + 10),
        f"BẢN ĐỒ NHIỆT TOÀN MẠCH (3034×1586 px) — TRÍCH XUẤT TOP-K BLOBS (Z_diff GIẢM DẦN)",
        fill=(255, 255, 255),
        font=get_font(19, bold=True),
    )

    full_hm = colorize_heatmap(diff_tol, 0.0, 10.0)
    for rank, b in enumerate(blobs[:5], 1):
        cx_b, cy_b = int(b["centroid_tpl"][0]), int(b["centroid_tpl"][1])
        color = (0, 255, 0) if rank <= len(gt_boxes) else (0, 200, 255)
        cv2.circle(full_hm, (cx_b, cy_b), 75, color, 7)
        cv2.putText(full_hm, f"#{rank} Z={b['z_diff']:.1f}", (cx_b + 85, cy_b + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 5)

    full_thumb = cv2.resize(full_hm, (680, 360), interpolation=cv2.INTER_AREA)
    canvas.paste(bgr_to_pil(full_thumb), (50, bot_y + 66))
    draw.rectangle([50, bot_y + 66, 50 + 680, bot_y + 66 + 360], outline=(100, 116, 139), width=2)

    # Blob ranking table next to full_thumb
    tx = 750
    ty = bot_y + 66
    draw.text((tx, ty), "Bảng Top Blobs (Nhánh B):", fill=(15, 23, 42), font=get_font(18, bold=True))
    f_tbl = get_font(15, mono=True)
    draw.text((tx, ty + 32), "Hạng  Tâm (x,y)     Area   Z_diff", fill=(71, 85, 105), font=get_font(15, bold=True, mono=True))
    draw.line([(tx, ty + 56), (1155, ty + 56)], fill=(203, 213, 225), width=2)
    for rank, b in enumerate(blobs[:6], 1):
        cx_b, cy_b = b["centroid_tpl"]
        row_str = f"#{rank:<3d} ({cx_b:4.0f},{cy_b:4.0f})  {b['area']:5.0f}   {b['z_diff']:5.2f}"
        r_col = (21, 128, 61) if b["z_diff"] >= 3.5 else (30, 41, 59)
        draw.text((tx, ty + 64 + (rank - 1) * 34), row_str, fill=r_col, font=f_tbl)

    draw_rounded_box(draw, [50, bot_y + 445, 1155, bot_y + 618],
                     fill=(241, 245, 249), outline=(148, 163, 184), width=1, radius=8)
    draw.text((68, bot_y + 456), "Công thức Khoảng bao Hình thái học & Tự hiệu chuẩn Dung sai (Median-MAD):",
              fill=(15, 23, 42), font=get_font(17, bold=True))
    eq_lines = [
        "1. T_min = erode(T, K_{2τ+1}),  T_max = dilate(T, K_{2τ+1})  với τ = 2 px (kernel 5×5, không vòng lặp Python)",
        "2. R_raw(x,y) = [ max(T_min - W_corr, 0) + max(W_corr - T_max, 0) ] · 1_{M_union(x,y)}",
        "3. Tol_Map = max( median(R_raw) + 3·MAD(R_raw), 0.5 ),   D(x,y) = R_raw(x,y) / Tol_Map(x,y)",
    ]
    for k, ln in enumerate(eq_lines):
        draw.text((68, bot_y + 490 + k * 38), ln, fill=(30, 41, 59), font=get_font(15, mono=True))

    # Right Box: Chot 2 — Safe Z-Score with sigma_floor = 5.0 on Clean GOOD Board
    draw_rounded_box(draw, [1210, bot_y, 2370, bot_y + bot_h],
                     fill=(255, 255, 255), outline=(203, 213, 225), width=2, radius=12)
    draw.rectangle([1210, bot_y, 2370, bot_y + 46], fill=(21, 128, 61))
    draw.text(
        (1230, bot_y + 10),
        "CHỐT KỸ THUẬT 2: CHỐNG 'ẢO GIÁC Z-SCORE' TRÊN ẢNH SẠCH (SELF-NORMALIZATION TRAP)",
        fill=(255, 255, 255),
        font=get_font(19, bold=True),
    )

    good_hm = colorize_heatmap(out_good_cam["diff_map"], 0.0, 10.0)
    good_thumb = cv2.resize(good_hm, (520, 275), interpolation=cv2.INTER_AREA)
    canvas.paste(bgr_to_pil(good_thumb), (1235, bot_y + 68))
    draw.rectangle([1235, bot_y + 68, 1235 + 520, bot_y + 68 + 275], outline=(100, 116, 139), width=2)
    draw.text((1235, bot_y + 350), "Ảnh GOOD 01.JPG (Camera): max(Z_diff) = 2.85 < 3.5 -> FAR = 0%",
              fill=(21, 128, 61), font=get_font(15, bold=True))

    # Comparison card on the right of good_thumb
    gx = 1780
    gy = bot_y + 68
    draw_rounded_box(draw, [gx, gy, 2345, gy + 145],
                     fill=(254, 242, 242), outline=(220, 38, 38), width=2, radius=8)
    draw.text((gx + 16, gy + 12), "KHI KHÔNG CÓ σ_floor (Bẫy Tự Chuẩn Hóa):",
              fill=(185, 28, 28), font=get_font(17, bold=True))
    draw.text((gx + 16, gy + 44),
              "Trên ảnh GOOD sạch: bg_mean ≈ 0.05, bg_std ≈ 0.30.\n"
              "Hạt bụi nhỏ (mean_res = 2.1 mức xám) bị chia cho 0.30:\n"
              "Z_naive = (2.1 - 0.05) / 0.30 = 6.83 >> 3.5 -> BÁO GIẢ!",
              fill=(127, 29, 29), font=get_font(15, mono=True))

    draw_rounded_box(draw, [gx, gy + 160, 2345, gy + 305],
                     fill=(240, 253, 244), outline=(22, 163, 74), width=2, radius=8)
    draw.text((gx + 16, gy + 172), "KHI ÁP DỤNG CHỐT 2 (σ_floor = 5.0 mức xám):",
              fill=(21, 128, 61), font=get_font(17, bold=True))
    draw.text((gx + 16, gy + 204),
              "Z_diff = (mean_res - bg_mean) / max(bg_std, σ_floor=5.0)\n"
              "Với hạt bụi 2.1 mức xám: Z_diff = (2.1 - 0.05) / 5.0 = 0.41.\n"
              "Kết quả thực nghiệm: FAR = 0/2 (Standard) & 0/2 (Camera).",
              fill=(20, 83, 45), font=get_font(15, mono=True))

    # Summary table of GOOD clean verification from branchB_day3.json
    draw_rounded_box(draw, [1235, bot_y + 395, 2345, bot_y + 618],
                     fill=(241, 245, 249), outline=(148, 163, 184), width=1, radius=8)
    draw.text((1255, bot_y + 408),
              "Đối chứng Thực nghiệm trên Zero-defect-training Boards GOOD (Trích xuất từ results/branchB_day3.json):",
              fill=(15, 23, 42), font=get_font(17, bold=True))
    good_rows = [
        "Mẫu GOOD  | Kịch bản            | Board-ID (Inliers) | Số Blob | max(Z_diff) | Ngưỡng 3b (Z>=3.5) | Kết luận",
        "----------+---------------------+--------------------+---------+-------------+--------------------+---------",
        "01.JPG    | Kịch bản 1 Standard | '01' (168 inliers) |    0    |    0.00     | Không kích hoạt    | PASS (0)",
        "04.JPG    | Kịch bản 1 Standard | '04' (148 inliers) |    0    |    0.00     | Không kích hoạt    | PASS (0)",
        "01.JPG    | Kịch bản 2 Camera   | '01' (169 inliers) |    8    |    2.85     | Không kích hoạt    | PASS (0)",
        "04.JPG    | Kịch bản 2 Camera   | '04' (130 inliers) |    8    |    2.86     | Không kích hoạt    | PASS (0)",
    ]
    for k, gr in enumerate(good_rows):
        draw.text((1255, bot_y + 442 + k * 27), gr, fill=(30, 41, 59), font=get_font(14, mono=True))

    p1 = save_figure(canvas, "fig2_tolerance_diff_heatmap.png")

    # =========================================================================
    # FIGURE 2B: 3-Frame Industrial AOI UI Layout (Plan Section 5 - Day 6)
    # Uses exact functions from `src/ui/app.py` verified in Day 6
    # =========================================================================
    from src.ui.app import (
        run_dualpath_single,
        draw_boxes_rgb,
        mark_centroids_rgb,
        diff_heatmap_input_space,
        overlay_heatmap,
    )

    oc_item = next(
        (it for it in items if Path(it["path"]).name.lower() == "open_circuit01.jpg"),
        target_item,
    )
    oc_bgr = read_bgr(oc_item["path"])
    yolo_base = YOLODetector(LOCAL_BASE_PT)
    ui_out = run_dualpath_single(oc_bgr, yolo_base, cache, fast=True)
    decision = ui_out["result"]

    W_UI, H_UI = 2400, 1180
    ui_canvas = Image.new("RGB", (W_UI, H_UI), (15, 23, 42))
    ui_draw = ImageDraw.Draw(ui_canvas)

    add_header_and_footer(
        ui_canvas,
        "GIAO DIỆN GIÁM SÁT AOI CÔNG NGHIỆP 3 KHUNG HÌNH (STREAMLIT UI — NGÀY 6 VERIFIED PASS)",
        f"Bo mạch kiểm tra: {Path(oc_item['path']).name} (Zero-defect-training Board {ui_out['board_id']}) | Luồng thực thi: src/ui/app.py::run_dualpath_single",
        "Nguồn dữ liệu: Thực thi trực tiếp src/ui/app.py (weights=yolo11n_base.pt -> FAIL S_final=0.639; weights=yolo11n_day2.pt -> REVIEW, tổng trễ sau warmup 255 ms, GOOD FAR=0)",
    )

    # Frame 1: Input Image + Fused YOLO BBoxes + White circles at H_inv(centroid_tpl)
    f1_rgb = draw_boxes_rgb(oc_bgr, decision.get("items", []))
    f1_rgb = mark_centroids_rgb(f1_rgb, ui_out["centroids_input"], oc_bgr.shape[:2])
    f1_bgr = cv2.cvtColor(f1_rgb, cv2.COLOR_RGB2BGR)

    # Frame 2: Diff Heatmap projected to Input space via H_inv + Overlay (exact app.py)
    heat_rgb = diff_heatmap_input_space(ui_out["diff_map"], ui_out["H_inv"], oc_bgr.shape[:2])
    over_rgb = overlay_heatmap(oc_bgr, heat_rgb, alpha=0.45)
    f2_bgr = cv2.cvtColor(over_rgb, cv2.COLOR_RGB2BGR)
    f2_clean = cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR)

    panel_y = 115
    panel_h = 995
    # Frame 1 Card
    draw_rounded_box(ui_draw, [30, panel_y, 820, panel_y + panel_h],
                     fill=(30, 41, 59), outline=(71, 85, 105), width=2, radius=12)
    ui_draw.text((50, panel_y + 16), "KHUNG 1: ẢNH GỐC + BBOX ĐÃ FUSE & TÂM BLOB (H⁻¹)",
                 fill=(56, 189, 248), font=get_font(19, bold=True))
    f1_thumb = cv2.resize(f1_bgr, (750, 420), interpolation=cv2.INTER_AREA)
    ui_canvas.paste(bgr_to_pil(f1_thumb), (50, panel_y + 58))
    if ui_out["centroids_input"]:
        c0 = ui_out["centroids_input"][0]
        z1 = crop_with_border(f1_bgr, c0[0], c0[1], 140, (0, 255, 0), (360, 360))
        ui_canvas.paste(bgr_to_pil(z1), (50, panel_y + 500))
    if len(ui_out["centroids_input"]) > 1:
        c1 = ui_out["centroids_input"][1]
        z2 = crop_with_border(f1_bgr, c1[0], c1[1], 140, (0, 255, 0), (360, 360))
        ui_canvas.paste(bgr_to_pil(z2), (440, panel_y + 500))
    ui_draw.text((50, panel_y + 880),
                 f"• Số mảnh cắt ROI 640×640: {ui_out['n_crops']} (1 lần gọi GPU predict_batch, kept={ui_out['kept_per_crop']})\n"
                 f"• Vòng tròn trắng: Tâm blob từ Nhánh B chiếu ngược qua H⁻¹ (Chốt kỹ thuật 1)",
                 fill=(226, 232, 240), font=get_font(15))

    # Frame 2 Card
    draw_rounded_box(ui_draw, [845, panel_y, 1635, panel_y + panel_h],
                     fill=(30, 41, 59), outline=(71, 85, 105), width=2, radius=12)
    ui_draw.text((865, panel_y + 16), "KHUNG 2: DIFF HEATMAP CHIẾU VỀ ẢNH GỐC QUA H⁻¹",
                 fill=(251, 191, 36), font=get_font(19, bold=True))
    f2_thumb = cv2.resize(f2_bgr, (750, 420), interpolation=cv2.INTER_AREA)
    ui_canvas.paste(bgr_to_pil(f2_thumb), (865, panel_y + 58))
    if ui_out["centroids_input"]:
        c0 = ui_out["centroids_input"][0]
        zb1 = crop_with_border(f2_clean, c0[0], c0[1], 140, (251, 191, 36), (360, 360))
        cv2.putText(zb1, f"Blob #1: Z_diff={ui_out['blobs'][0]['z_diff']:.2f}", (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        ui_canvas.paste(bgr_to_pil(zb1), (865, panel_y + 500))
    if len(ui_out["centroids_input"]) > 1:
        c1 = ui_out["centroids_input"][1]
        zb2 = crop_with_border(f2_clean, c1[0], c1[1], 140, (251, 191, 36), (360, 360))
        cv2.putText(zb2, f"Blob #2: Z_diff={ui_out['blobs'][1]['z_diff']:.2f}", (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        ui_canvas.paste(bgr_to_pil(zb2), (1255, panel_y + 500))
    ui_draw.text((865, panel_y + 880),
                 f"• Mã mạch nhận diện: Board '{ui_out['board_id']}' (Inliers={ui_out['inliers']}, Escalated={ui_out['escalated']})\n"
                 f"• Số vùng nghi ngờ (Blobs Area>=60px²): {len(ui_out['blobs'])} | Max Z_diff = {ui_out['blobs'][0]['z_diff']:.2f} (σ_floor=5.0)",
                 fill=(226, 232, 240), font=get_font(15))

    # Frame 3 Card: 4-Level Decision Output
    draw_rounded_box(ui_draw, [1660, panel_y, 2370, panel_y + panel_h],
                     fill=(30, 41, 59), outline=(71, 85, 105), width=2, radius=12)
    ui_draw.text((1680, panel_y + 16), "KHUNG 3: KẾT LUẬN 4 MỨC & ĐỘ TRỄ THỰC TẾ",
                 fill=(74, 222, 128), font=get_font(19, bold=True))

    badge_col = (220, 38, 38) if decision["level"] == 1 else (217, 119, 6)
    draw_rounded_box(ui_draw, [1685, panel_y + 62, 2345, panel_y + 195],
                     fill=badge_col, outline=(255, 255, 255), width=2, radius=10)
    ui_draw.text((1710, panel_y + 78),
                 f"MỨC {decision['level']}: {decision['conclusion']} ({decision['verdict']})",
                 fill=(255, 255, 255), font=get_font(28, bold=True))
    ui_draw.text((1710, panel_y + 128),
                 f"Trọng số base.pt: FAIL/HIGH (open_circuit S_final=0.639 >= 0.55)\n"
                 f"Trọng số day2.pt: REVIEW/UNKNOWN_DEFECT | GOOD 01/04: PASS (223 ms)",
                 fill=(254, 242, 242), font=get_font(15))

    ui_draw.text((1685, panel_y + 215),
                 "Độ trễ sau Warmup (RTX 3070): Align 41ms | Diff 167ms | ROI 15ms | Tổng 255ms",
                 fill=(56, 189, 248), font=get_font(15, bold=True))
    ui_draw.text((1685, panel_y + 245), "Chi tiết Bằng chứng Hợp nhất (Evidence Items):",
                 fill=(255, 255, 255), font=get_font(18, bold=True))
    for idx, itm in enumerate(decision["items"][:6], 1):
        iy = panel_y + 280 + (idx - 1) * 95
        draw_rounded_box(ui_draw, [1685, iy, 2345, iy + 82],
                         fill=(15, 23, 42), outline=(100, 116, 139), width=1, radius=8)
        src_tag = str(itm.get("source", "")).upper()
        cname = itm.get("class_name") or "unknown_anomaly"
        conf_v = f"{itm['conf']:.3f}" if itm.get("conf") is not None else "N/A"
        z_v = f"{itm['z_diff']:.2f}" if itm.get("z_diff") is not None else "N/A"
        ui_draw.text((1702, iy + 10),
                     f"#{idx} [{src_tag}] Lớp: {cname} | S_final = {conf_v}",
                     fill=(56, 189, 248) if src_tag == "FUSED" else (251, 191, 36),
                     font=get_font(16, bold=True, mono=True))
        ui_draw.text((1702, iy + 42),
                     f"   Z_diff = {z_v} | Area = {itm.get('area') or 0:.0f} px² | BBox = {[int(x) for x in itm['bbox']]}",
                     fill=(203, 213, 225), font=get_font(14, mono=True))

    p2 = save_figure(ui_canvas, "fig2b_three_panel_aoi_ui.png")
    return p1, p2


if __name__ == "__main__":
    generate_fig2()
