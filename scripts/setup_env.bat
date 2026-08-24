@echo off
REM ===========================================================================
REM MamNexa AI - Phase I environment bootstrap (Windows)
REM Creates a local virtual environment and installs pinned dependencies.
REM
REM PREREQUISITE: a real Python interpreter must be installed and on PATH.
REM The Microsoft Store "python.exe" stub does NOT count - install CPython from
REM https://www.python.org/downloads/windows/ (check "Add python.exe to PATH")
REM or run:  winget install Python.Python.3.12
REM Verify with:  python --version   (must print e.g. "Python 3.12.x")
REM ===========================================================================

setlocal
cd /d "%~dp0\.."

echo [1/4] Verifying Python interpreter...
python --version 1>nul 2>nul
if errorlevel 1 (
    echo   ERROR: 'python' not found or not a real interpreter.
    echo   Install CPython 3.11/3.12 first, then re-run this script.
    exit /b 1
)
python --version

echo [2/4] Creating virtual environment at .venv ...
python -m venv .venv || (echo   ERROR: venv creation failed & exit /b 1)

echo [3/4] Upgrading pip ...
call .venv\Scripts\python.exe -m pip install --upgrade pip

echo [4/4] Installing pinned requirements ...
call .venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo Done. Activate the environment with:
echo     .venv\Scripts\activate
echo Then run the Phase I verification tests with:
echo     python -m pytest tests -v
endlocal
