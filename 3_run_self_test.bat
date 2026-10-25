@echo off
REM =====================================================================
REM  3_run_self_test.bat — Tu kiem tra nhanh tren may moi (1-click).
REM  1. pytest tests/ (logic 4 muc, fusion, board-ID, gating, write-guard).
REM  2. Pre-flight end-to-end 1 anh demo (Nhanh A + Nhanh B + Fusion 4 muc).
REM =====================================================================
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

set "PYTHONPATH=%CD%"
echo ============================================================
echo  [1/2] pytest tests/
echo ============================================================
"%PY%" -m pytest tests/ -q
if errorlevel 1 (
  echo [LOI] pytest that bai — xem log tren.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo  [2/2] Pre-flight end-to-end 1 anh demo (GOOD 01 + 1 anh loi board 01)
echo ============================================================
"%PY%" scripts/debug_e2e_trace.py
if errorlevel 1 (
  echo [LOI] E2E trace that bai — xem log tren.
  pause
  exit /b 1
)

echo.
echo [OK] May moi san sang chay demo: 2_run_ui_demo.bat
pause
