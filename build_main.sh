#!/bin/bash
echo "======================================================="
echo "开始打包 课堂行为分析工具 为独立可执行文件"
echo "======================================================="
echo ""

# 1. 清理旧缓存
echo "[1/4] 正在清理旧的编译缓存..."
rm -rf build dist main.spec "课堂行为分析工具.spec"
echo "    完成。"

# 2. 运行 PyInstaller 打包
echo ""
echo "[2/4] 开始执行 PyInstaller..."
pyinstaller -F -w --clean \
    --add-data "templates:templates" \
    --add-data "tcl_tk_libs:tcl_tk_libs" \
    --name "课堂行为分析工具" \
    main.py

if [ $? -ne 0 ]; then
    echo "[错误] PyInstaller 打包失败！"
    exit 1
fi
echo "    PyInstaller 打包完成。"

# 3. 创建发布目录
echo ""
echo "[3/4] 创建发布目录..."
RELEASE_DIR="release"
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"

# 复制主程序
cp "dist/课堂行为分析工具" "$RELEASE_DIR/" 2>/dev/null || cp "dist/课堂行为分析工具.exe" "$RELEASE_DIR/" 2>/dev/null

# 复制 ffmpeg
if [ -f "ffmpeg.exe" ]; then
    cp ffmpeg.exe "$RELEASE_DIR/"
elif [ -f "ffmpeg" ]; then
    cp ffmpeg "$RELEASE_DIR/"
fi

# 复制使用文档
cp 使用文档.md "$RELEASE_DIR/" 2>/dev/null
cp README.txt "$RELEASE_DIR/" 2>/dev/null

echo "    完成。"

# 4. 显示结果
echo ""
echo "======================================================="
echo "[4/4] 打包完成！"
echo ""
echo "发布目录: $RELEASE_DIR/"
echo ""
echo "目录内容:"
ls -la "$RELEASE_DIR/"
echo ""
echo "使用方式:"
echo "  1. 将整个 $RELEASE_DIR 文件夹复制到目标电脑"
echo "  2. 双击运行主程序"
echo "  3. 浏览器会自动打开功能选择页面"
echo ""
echo "注意:"
echo "  - 目标电脑需要能打开网页浏览器"
echo "  - ffmpeg 必须与主程序在同一目录"
echo "  - 不支持中文路径下的视频文件"
echo "======================================================="
