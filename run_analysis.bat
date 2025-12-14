@echo off
echo Starting Genetic Image Analysis System...
echo.

REM Start virtual environment
call venv\Scripts\activate

REM Run analysis system
python complete_system.py

pause