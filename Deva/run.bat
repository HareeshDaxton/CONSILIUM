@echo off
REM CONSILIUM vision server (Windows). Double-click or run from this folder.
REM First run: creates .venv, installs GPU PyTorch + packages, downloads the models.
cd /d "%~dp0"

if not defined TORCH_INDEX set TORCH_INDEX=https://download.pytorch.org/whl/cu124

if not exist .venv\Scripts\python.exe (
  echo [setup] Creating Python environment in .venv ...
  python -m venv .venv || goto :nopython
  .venv\Scripts\python -m pip install --upgrade pip || goto :fail
  echo [setup] Installing PyTorch with CUDA from %TORCH_INDEX% ...
  .venv\Scripts\python -m pip install torch --index-url %TORCH_INDEX% || goto :fail
  echo [setup] Installing the other packages ...
  .venv\Scripts\python -m pip install -r requirements.txt || goto :fail
)

.venv\Scripts\python server.py
pause
exit /b 0

:nopython
echo Python was not found. Install Python 3.10-3.12 from python.org (tick "Add python.exe to PATH") and run again.
pause
exit /b 1

:fail
echo Setup failed - see the messages above. Delete the .venv folder and run again after fixing.
pause
exit /b 1
