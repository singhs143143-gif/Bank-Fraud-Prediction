@echo off
"%~dp0.venv\Scripts\python.exe" -m streamlit %*
exit /b %errorlevel%
