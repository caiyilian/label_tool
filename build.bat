@echo off
chcp 65001 >nul
echo =======================================================
echo 开始打包 Multi-Sync Web 为独立可执行文件 (.exe)
echo =======================================================
echo.

:: 1. 清理旧缓存（非常重要，防止错误的 dll 被缓存在 build 目录）
echo [1/3] 正在清理旧的编译缓存...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist web_app.spec del /q web_app.spec

:: 2. 更新 PyInstaller
echo.
echo [2/3] 正在确认环境...
:: 取消屏蔽环境变量，因为有的时候 conda 需要它来找正确的包

:: 3. 运行打包
echo.
echo [3/3] 开始执行 PyInstaller...
echo 命令: pyinstaller -F -w --clean --add-data "templates;templates" --add-data "tcl_tk_libs;tcl_tk_libs" web_app.py
pyinstaller -F -w --clean --add-data "templates;templates" --add-data "tcl_tk_libs;tcl_tk_libs" web_app.py

echo.
if %errorlevel% equ 0 (
    echo =======================================================
    echo [成功] 打包完成！
    echo 生成的 exe 文件位于: dist\web_app.exe
    echo.
    echo 提示: 
    echo 1. 运行 web_app.exe 即可打开控制台界面并自动弹出浏览器。
    echo 2. 请确保同级目录下有 ffmpeg.exe，否则可能无法正常进行抽帧操作。
    echo =======================================================
) else (
    echo [失败] 打包过程中出现错误，请检查日志。
)

pause