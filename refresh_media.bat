@echo off
REM Windows batch script to refresh WhatsApp media ID
REM This is designed to be run by Windows Task Scheduler

REM Change to script directory
cd /d "%~dp0"

REM Log the start
echo [%date% %time%] Starting media refresh... >> task_scheduler.log

REM Run the Python script
python refresh_media.py >> task_scheduler.log 2>&1

REM Log the completion
echo [%date% %time%] Media refresh completed with exit code: %errorlevel% >> task_scheduler.log
echo. >> task_scheduler.log

REM Exit with the same code as Python script
exit /b %errorlevel%