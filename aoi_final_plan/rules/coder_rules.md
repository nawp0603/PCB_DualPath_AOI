# CODER AGENT — RULES (CHONG QUEN VAI TRO)

## VAI TRO
- Ban la `CODER AGENT` (nguoi xay dung he thong Dual-Path Industrial AOI).
- Nhiem vu duy nhat: implement code theo dung hang muc ngay duoc giao trong `aoi_final_plan/plan.md`.
- Moi hanh dong phai phuc vu pipeline Dual-Path (Nhanh A + Nhanh B + fusion trong `logic.py`), khong tu y mo rong pham vi.

## QUYEN GHI (WRITE SCOPE)
- Chi duoc sua/tao cac file code trong `src/` va `scripts/` dung theo hang muc ngay duoc giao trong `aoi_final_plan/plan.md`.
- Duoc cap nhat trang thai cong viec vao cuoi file `aoi_final_plan/status_and_bugs.md` (phan LOG GIAO TIEP & BAO LOI).
- Khong duoc ghi ra bat ky noi nao khac.

## VUNG CAM (FORBIDDEN)
1. CAM sua file trong `tests/`, `results/`, hoac sua noi dung `aoi_final_plan/plan.md`.
2. CAM chay danh gia toan tap du lieu hay huan luyen dai tren GPU (chi duoc chay smoke-test 1-2 anh de kiem tra loi cu phap/runtime).
3. CAM giu lai hoac goi bat ky logic nao cua PatchCore va SAHI trong luong quyet dinh chinh.
4. CAM tuyet doi viet/xoa/sua bat ky file nao trong 2 thu muc project cu:
   - `D:\FPTU\KLTN\PCB_Anomaly_Inspector_Portable` (STRICT READ-ONLY)
   - `D:\FPTU\KLTN\Test_github` (STRICT READ-ONLY)

## DUONG DAN TAI NGUYEN — STRICT READ-ONLY (dieu khoan bao ve)
- Moi duong dan toi Dataset va 2 Project cu duoc khai bao trong `configs/resource_paths.py` deu o che do **STRICT READ-ONLY**.
- Cam moi hanh vi mo file voi mode `'w'` hoac xoa/ghi de len cac duong dan cua project cu.
- Khi fine-tune YOLO hoac sinh dataset tile moi, bat buoc luu dau ra vao ben trong thu muc cua project moi (`D:\FPTU\KLTN\PCB_DualPath_AOI/...`), tuyet doi khong ghi de vao thu muc dataset dang dung cua project cu.
- Trong so `best.pt` goc chi doc tai `LEGACY_BEST_PT`; moi fine-tune bat dau tu ban copy `assets/weights/yolo11n_base.pt`.

## 3 CHOT CHAN CODE BAT BUOC PHAI TUAN THU
1. **Coordinate Sync:** Moi toa do vung loi tu Nhanh B (tren khong gian Template) phai duoc chieu nguoc ve khong gian Anh dau vao goc bang `H_inv` (`cv2.perspectiveTransform`) truoc khi crop `640x640` va truoc khi tinh IoU voi Nhanh A.
2. **Safe Z-Score:** Khi tinh `Z_diff`, bat buoc dung cong thuc co san chan duoi: `Z_diff = (blob_score - bg_mean) / max(bg_std, sigma_floor)` (voi `sigma_floor` ~ 4.0-6.0) de anh GOOD khong bi vot Z-score ao.
3. **High-Res & Batching:** Nhanh A phai bat `rect=True, half=True` o `imgsz=3072/3584`. Nhanh B phai gom toan bo manh crop `640x640` (toi da Top-8 blobs) vao 1 list de goi `model.predict(batch_crops)` trong 1 lan duy nhat kem **Spatial Center Gating** (`R = 80px`).
