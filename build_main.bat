@echo off
chcp 65001 >nul
echo =======================================================
echo Packing Classroom Behavior Analysis Tool
echo =======================================================
echo.
echo [1/4] Cleaning old build files...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist main.spec del /q main.spec
echo     Done.
echo.
echo [2/4] Running PyInstaller...
pyinstaller -F -w --clean --add-data "templates;templates" --add-data "tcl_tk_libs;tcl_tk_libs" --name "ClassroomTool" main.py
if %errorlevel% neq 0 (
    echo [ERROR] PyInstaller failed!
    pause
    exit /b 1
)
echo     Done.
echo.
echo [3/4] Creating release folder...
if exist release rmdir /s /q release
mkdir release
copy "dist\ClassroomTool.exe" "release\" >nul
copy ffmpeg.exe "release\" >nul
copy README.txt "release\" >nul
echo     Done.
echo.
echo =======================================================
echo [4/4] Build complete!
echo Release folder: release\
echo Contents:
dir /b release\
echo =======================================================
pause
