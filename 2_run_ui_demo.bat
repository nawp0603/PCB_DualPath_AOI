@echo off
REM =====================================================================
REM  2_run_ui_demo.bat — Mo demo Streamlit Dual-Path AOI (1-click).
REM  Chay duoc o bat ky o dia/thu muc nao. Khong can o D:, khong can venv cu.
REM =====================================================================
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  echo [WARN] Chua co .venv — thu dung 'python' he thong.
  echo        (khuyen nghi chay 1_setup_env.bat truoc)
  set "PY=python"
)

set "PYTHONPATH=%CD%"
echo Mo UI: src/ui/app.py (PYTHONPATH=%CD%)
"%PY%" -m streamlit run src/ui/app.py
pause
