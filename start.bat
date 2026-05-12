@echo off
REM Startup script for MTMC Tracker (Backend + Frontend) - Windows

echo ========================================
echo   MTMC Tracker System - Starting...
echo ========================================
echo.

REM Check Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

REM Check Node.js
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js not found. Please install Node.js 18+
    pause
    exit /b 1
)

echo [1/4] Installing backend dependencies...
pip install -r backend_requirements.txt -q

echo [2/4] Installing frontend dependencies...
cd frontend
if not exist "node_modules" (
    call npm install
)
cd ..

echo [3/4] Checking models...
if not exist "models" (
    echo.
    echo [WARNING] Models not found!
    echo Download from: https://www.kaggle.com/datasets/mrkdagods/mtmc-weights
    echo Extract to: ./models/
    echo.
    echo Press Enter to continue in DEMO MODE or Ctrl+C to exit...
    pause >nul
)

echo [4/4] Starting servers...
echo.

REM Start backend
echo Starting Backend API Server on port 8000...
start "MTMC Backend" python backend_api.py

REM Wait for backend
timeout /t 3 /nobreak >nul

REM Start frontend
echo Starting Frontend Dev Server on port 3000...
cd frontend
start "MTMC Frontend" npm run dev
cd ..

echo.
echo ========================================
echo   System is RUNNING!
echo ========================================
echo.
echo Backend API:  http://localhost:8000/docs
echo Frontend UI:  http://localhost:3000
echo Health Check: http://localhost:8000/api/health
echo.
echo Close this window to stop both servers
echo.
pause
