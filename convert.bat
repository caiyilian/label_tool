@echo off
for %%i in ("dav_files\*.dav") do (
    echo 正在转换 "%%i" ...
    ffmpeg -i "%%i" -c copy "%%~dpni.mp4"
)
echo 全部转换完成！
pause