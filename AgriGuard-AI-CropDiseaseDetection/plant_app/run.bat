@echo off
REM Smart Plant Health Monitoring System - Windows Startup Script

echo ====================================
echo Smart Plant Health Monitoring System
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo [1/4] Checking dependencies...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo [2/4] Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo [2/4] Dependencies already installed
)

echo [3/4] Checking for .env file...
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env >nul
    echo Please edit .env file and add your Groq API key!
    pause
)

echo [4/4] Starting application...
echo.
echo ====================================
echo Server starting on http://localhost:5000
echo Press Ctrl+C to stop the server
echo ====================================
echo.

python app.py
pause
