# PCB_DualPath_AOI — Ban Portable chay doc lap 100% (Windows)

Goi nay giai nen ra **bat ky o dia/thu muc nao** tren may Windows khac
(khong can o `D:`, khong can project cu) van chay duoc Demo + tu kiem tra.

## 3 buoc chay demo

1. **Giai nen** file `PCB_DualPath_AOI_Portable_Release.zip` ra thu muc tuy y,
   vi du `E:\AOI\` hoac `C:\Users\Ban\PCB_DualPath_AOI\`.
2. **Cai moi truong** (1 lan duy nhat): nhay dup vao
   **`1_setup_env.bat`** — tu tao `.venv` tai cho, tu phat hien GPU
   (`nvidia-smi`) de cai torch CUDA `.cu124` hoac CPU, roi cai
   `requirements.txt`. May khong co GPU van chay duoc
   (YOLO tu ha `imgsz` 1920 + `half=False`, khong crash).
3. **Mo demo**: nhay dup vao **`2_run_ui_demo.bat`** — mo trinh duyet
   Streamlit voi 6 anh mau board 01 + 2 GOOD template, chay Dual-Path
   (Nhanh A + Nhanh B + Fusion 4 muc) offline 100%.

## Tu kiem tra tren may moi

- Nhay dup **`3_run_self_test.bat`**: chay `pytest tests/` + trace
  End-to-End 1 anh demo (`scripts/debug_e2e_trace.py`).
- Trace thu cong: `.venv\Scripts\python.exe scripts/debug_e2e_trace.py`
  (them `--good01` de thu anh GOOD, `--full` de align full,
  `--json results/e2e_trace.json` de luu trace).

## May da co Conda / Python san

Neu may moi da co Conda hoac Python 3.11 va muon dung moi truong rieng:

```bat
conda create -n aoi python=3.11 -y
conda activate aoi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
set PYTHONPATH=%CD%
python -m streamlit run src/ui/app.py
```

May chi CPU: thay `cu124` bang `https://download.pytorch.org/whl/cpu`.

## Cau hinh theo phan cung tren UI

- Sidebar → **imgsz Nhanh A**: `Auto` (mac dinh, giu 3072, tu ha khi VRAM
  yeu), `2560` (VRAM 4–6GB), `1920` (CPU / VRAM <4GB). Giu `rect=True +
  half=True` khi co CUDA; gap `OutOfMemoryError` tu thu lai 2560/1920.
- Sidebar → **Device**: `auto` / `cuda` / `cpu`.
- Trong so: mac dinh `yolo11n_day2.pt` (fine-tune), fallback `yolo11n_base.pt`.

## Du lieu di kem (offline 100%)

- `assets/weights/`: `yolo11n_day2.pt` (mac dinh) + `yolo11n_base.pt` +
  `best.pt` + `orb_cache.npz` + 10 Golden Template (01/04/05–12).
- `data/demo_and_test_samples/`: 36 anh loi + 36 XML board 01/04
  (6 lop × 6 anh) + GOOD lay tu `assets/templates/01.JPG|04.JPG` —
  du cho UI Demo, `debug_e2e_trace.py` va `evaluate.py --max-images`.
- `results/` + `aoi_final_plan/` (bao cao, bang so lieu, tai lieu Scribe)
  giu nguyen de tra cuu.

## Ghi chu mo phong nhieu camera (kich ban 2)

`evaluate.py --scenario camera` tu tong hop dich/ xoay/ phoi canh/
sang-vignette-nhieu tren anh demo (co warp GT boxes) — khong can them file.
