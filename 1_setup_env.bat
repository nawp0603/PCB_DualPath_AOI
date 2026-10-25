@echo off
REM =====================================================================
REM  1_setup_env.bat — Tao moi truong portable 1-click (Windows).
REM  Chay 1 lan duy nhat sau khi giai nen zip, tren bat ky o dia/thu muc nao.
REM =====================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo  [1/4] Kiem tra Python...
echo ============================================================
where python >nul 2>nul
if errorlevel 1 (
  echo [LOI] Khong tim thay 'python' trong PATH.
  echo Hay cai Python 3.11 64-bit tu https://www.python.org/downloads/
  echo (tick "Add python.exe to PATH" khi cai) roi chay lai file nay.
  pause
  exit /b 1
)
python --version

echo.
echo ============================================================
echo  [2/4] Tao virtualenv .venv tai cho (neu chua co)...
echo ============================================================
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 (
    echo [LOI] Tao .venv that bai.
    pause
    exit /b 1
  )
  echo [OK] Da tao .venv
) else (
  echo [OK] .venv da ton tai — dung lai.
)

echo.
echo ============================================================
echo  [3/4] Phat hien GPU (nvidia-smi) -^> chon ban torch...
echo ============================================================
set "TORCH_IDX=https://download.pytorch.org/whl/cpu"
nvidia-smi >nul 2>nul
if not errorlevel 1 (
  echo [GPU] Phat hien card NVIDIA -^> cai torch CUDA cu124.
  set "TORCH_IDX=https://download.pytorch.org/whl/cu124"
) else (
  echo [CPU] Khong thay nvidia-smi -^> cai torch ban CPU.
  echo      (pipeline van chay: YOLO tu ha imgsz 1920 + half=False)
)

echo.
echo ============================================================
echo  [4/4] Cai dat torch + requirements.txt...
echo ============================================================
".venv\Scripts\python.exe" -m pip install --upgrade pip
if not errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install torch torchvision --index-url %TORCH_IDX%
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
) else (
  echo [LOI] pip that bai.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo  [KIEM TRA NHANH] torch + ultralytics + streamlit...
echo ============================================================
".venv\Scripts\python.exe" -c "import torch,ultralytics,cv2,numpy,streamlit; print('torch',torch.__version__,'cuda=',torch.cuda.is_available()); print('ultralytics ok, cv2',cv2.__version__,'numpy',numpy.__version__,'streamlit ok')"

echo.
echo ============================================================
echo  XONG. Chay tiep 2_run_ui_demo.bat de mo demo.
echo ============================================================
pause
