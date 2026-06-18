@echo off
chcp 65001 >nul
title Hanui 6-model GraphRAG eval (do not close)
pushd "%~dp0"
set OLLAMA_MODELS=D:\ollama-models
set PYTHONUTF8=1
echo ============================================================
echo  6-model GraphRAG queue (resumes; completed models skipped)
echo  Keep this window OPEN until "ALL MODELS DONE".
echo ============================================================
echo.
python -u run_all_models.py
echo.
echo ================= ALL MODELS DONE =================
echo (You can close this window now.)
pause
