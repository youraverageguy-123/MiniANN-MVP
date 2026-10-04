@echo off
REM ========================================================
REM MiniANN Workspace Sanitizer / Clean Script
REM Safely removes build artifacts and temporary files
REM ========================================================

echo [clean] Cleaning build artifacts...

if exist obj rmdir /s /q obj
if exist build rmdir /s /q build

del /q *.o 2>nul
del /q *.exe 2>nul
del /q *.model 2>nul
del /q *.miniann 2>nul
del /q *_loss.csv 2>nul
del /q *_report.html 2>nul
del /q training_curve.png 2>nul

echo [clean] Workspace cleaned successfully.
