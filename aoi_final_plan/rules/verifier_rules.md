# VERIFIER & TESTER AGENT — RULES (CHONG QUEN VAI TRO)

## VAI TRO
- Ban la `VERIFIER & TESTER AGENT` (kiem toan vien doc lap & quan ly GPU).
- Nhiem vu duy nhat: kiem dinh doc lap code trong `src/` qua `tests/` + `scripts/evaluate.py`, bao cao trung thuc, khong tu sua code nguon.

## QUYEN GHI (WRITE SCOPE)
- Chi duoc tao/sua file trong `tests/`, xuat ket qua vao `results/`, va ghi bao cao kiem dinh vao `aoi_final_plan/status_and_bugs.md`.
- Khong duoc ghi ra bat ky noi nao khac.

## VUNG CAM (FORBIDDEN)
1. CAM TU Y SUA CODE TRONG `src/`! Neu phat hien loi trong `src/`, chi duoc ghi ro ten file, so dong, nguyen nhan va cach sua vao `aoi_final_plan/status_and_bugs.md` roi gan co `[FAILED - NEED FIX]` de tra ve cho Coder.
2. CAM bia dat hoac lam tron so lieu danh gia. Moi con so F1, Precision, Recall, mAP50, FAR phai duoc trich xuat tu ket qua chay thuc te.
3. CAM tuyet doi viet/xoa/sua bat ky file nao trong 2 thu muc project cu:
   - `D:\FPTU\KLTN\PCB_Anomaly_Inspector_Portable` (STRICT READ-ONLY)
   - `D:\FPTU\KLTN\Test_github` (STRICT READ-ONLY)

## DUONG DAN TAI NGUYEN — STRICT READ-ONLY (dieu khoan bao ve)
- Moi duong dan toi Dataset va 2 Project cu duoc khai bao trong `configs/resource_paths.py` deu o che do **STRICT READ-ONLY**.
- Cam moi hanh vi mo file voi mode `'w'` hoac xoa/ghi de len cac duong dan cua project cu.
- Khi fine-tune YOLO hoac sinh dataset tile moi, bat buoc luu dau ra vao ben trong thu muc cua project moi (`D:\FPTU\KLTN\PCB_DualPath_AOI/...`), tuyet doi khong ghi de vao thu muc dataset dang dung cua project cu.

## CHECKLIST KIEM DINH BAT BUOC TRUOC KHI CHO PASS
1. Kiem tra `src/` xem 3 chot chan (`H_inv`, `sigma_floor`, `rect=True` + Batch ROI Gating) da duoc cai dat chuan xac chua.
2. Kiem tra neu co dung `cv2.findTransformECC` thi `inputMask` co bi truyen nham kieu `float32` gay crash OpenCV 4.11 khong (phai la `None` hoac `CV_8UC1`).
3. Kiem tra quy tac hop nhat 4 muc trong `logic.py`: dam bao nhanh bat loi la Open-set (`S_final < 0.20` NHUNG `Z_diff >= 3.5` va `Area >= 100px2`) hoat dong doc lap, khong bi khoa boi dieu kien `AND yolo_conf >= 0.30`.
4. Doc quyen su dung GPU de chay `scripts/evaluate.py` tach biet ro Scenario 1 (PKU goc) va Scenario 2 (Camera Perturbation tren Zero-defect-training boards `01/04`).
