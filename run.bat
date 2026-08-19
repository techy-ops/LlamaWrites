@echo off
title LlamaWrites v2.0
echo ===================================================
echo           Starting LlamaWrites v2.0
echo ===================================================
echo.

cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    streamlit run "LlamaWrite Code.py"
) else (
    echo [ERROR] Virtual environment not found in %~dp0venv.
    echo Please create the venv first or run: python -m venv venv
    pause
)
