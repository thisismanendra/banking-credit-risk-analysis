@echo off
title Banking and Credit Risk Assistant
cd /d "C:\Users\Manendra\OneDrive\Documents\SLM project\lendingclub_credit_risk"
echo Starting Banking and Credit Risk Assistant...
"c:\Users\Manendra\OneDrive\Documents\SLM project\SLM_Work\Scripts\python.exe" "C:\Users\Manendra\OneDrive\Documents\SLM project\lendingclub_credit_risk\launch_banking_assistant.py"
if errorlevel 1 (
  echo.
  echo Application exited with an error.
  pause
)
