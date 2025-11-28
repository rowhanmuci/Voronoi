@echo off
REM Voronoi Diagram - 快速打包腳本 (Windows)
REM 雙擊執行此檔案來建立可執行檔

echo ========================================
echo Voronoi Diagram - 快速打包工具
echo ========================================
echo.

REM 檢查 Python
echo 檢查 Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [錯誤] 找不到 Python
    echo 請先安裝 Python 3.7+
    pause
    exit /b 1
)
echo [OK] Python 已安裝

REM 檢查 PyInstaller
echo.
echo 檢查 PyInstaller...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [警告] PyInstaller 未安裝
    echo 正在安裝 PyInstaller...
    pip install pyinstaller
)
echo [OK] PyInstaller 已安裝

REM 清理舊檔案
echo.
echo 清理舊檔案...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist VoronoiDiagram.spec del VoronoiDiagram.spec
echo [OK] 清理完成

REM 執行打包
echo.
echo 開始打包（這可能需要幾分鐘）...
echo.
pyinstaller --name=VoronoiDiagram --onedir --windowed voronoi_gui.py

REM 檢查結果
echo.
if exist dist\VoronoiDiagram\VoronoiDiagram.exe (
    echo ========================================
    echo [成功] 可執行檔已生成！
    echo ========================================
    echo.
    echo 位置: dist\VoronoiDiagram\VoronoiDiagram.exe
    echo.
    
    REM 複製測試檔案
    echo 複製測試檔案...
    copy test_input.txt dist\VoronoiDiagram\ >nul 2>&1
    copy test_no_comment.txt dist\VoronoiDiagram\ >nul 2>&1
    copy test_with_comment.txt dist\VoronoiDiagram\ >nul 2>&1
    copy README.txt dist\VoronoiDiagram\ >nul 2>&1
    echo [OK] 測試檔案已複製
    
    echo.
    echo 下一步:
    echo 1. 測試: 進入 dist\VoronoiDiagram\ 雙擊 VoronoiDiagram.exe
    echo 2. 繳交: 壓縮整個 VoronoiDiagram 目錄為 .zip
    echo.
) else (
    echo [錯誤] 打包失敗
    echo 請檢查錯誤訊息
)

pause
