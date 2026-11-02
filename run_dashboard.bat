@echo off
title Bank Fraud Prediction Dashboard
cd /d "%~dp0"
echo ========================================================
echo   Launching Intelligent Banking Fraud Detection Dashboard
echo ========================================================
echo.
call .venv\Scripts\activate.bat
streamlit run app.py
pause
