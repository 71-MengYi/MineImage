@echo off
chcp 65001 >nul
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "packaging\build_exe.ps1" %*
if errorlevel 1 pause
