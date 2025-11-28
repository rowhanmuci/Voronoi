"""
製作可執行檔的腳本

使用方法:
1. 安裝 PyInstaller: pip install pyinstaller
2. 執行: python build_exe.py
3. 可執行檔會在 dist/VoronoiDiagram/ 目錄中
"""

import os
import subprocess
import shutil

print("=" * 70)
print("Voronoi Diagram - 可執行檔打包工具")
print("=" * 70)

# 檢查必要檔案
required_files = [
    'voronoi_gui.py',
    'voronoi_geometry.py',
    'voronoi_algorithm.py'
]

print("\n1. 檢查必要檔案...")
for file in required_files:
    if os.path.exists(file):
        print(f"   ✅ {file}")
    else:
        print(f"   ❌ {file} (缺失)")
        exit(1)

print("\n2. 檢查 PyInstaller...")
try:
    result = subprocess.run(['pyinstaller', '--version'], 
                          capture_output=True, text=True)
    print(f"   ✅ PyInstaller {result.stdout.strip()}")
except:
    print("   ❌ PyInstaller 未安裝")
    print("\n請執行: pip install pyinstaller")
    exit(1)

print("\n3. 清理舊檔案...")
if os.path.exists('build'):
    shutil.rmtree('build')
    print("   ✅ 清理 build/")
if os.path.exists('dist'):
    shutil.rmtree('dist')
    print("   ✅ 清理 dist/")
if os.path.exists('VoronoiDiagram.spec'):
    os.remove('VoronoiDiagram.spec')
    print("   ✅ 清理 VoronoiDiagram.spec")

print("\n4. 開始打包...")
print("   這可能需要幾分鐘，請稍候...")

# PyInstaller 參數
cmd = [
    'pyinstaller',
    '--name=VoronoiDiagram',           # 程式名稱
    '--onedir',                         # 打包成目錄（較小）
    '--windowed',                       # GUI 模式（不顯示 console）
    '--icon=NONE',                      # 不使用圖示
    '--add-data=voronoi_geometry.py;.',  # 包含額外檔案
    '--add-data=voronoi_algorithm.py;.',
    'voronoi_gui.py'                    # 主程式
]

# Windows 和 Linux/Mac 的路徑分隔符不同
if os.name != 'nt':  # 非 Windows
    cmd = [arg.replace(';', ':') for arg in cmd]

try:
    subprocess.run(cmd, check=True)
    print("\n   ✅ 打包成功！")
except subprocess.CalledProcessError as e:
    print(f"\n   ❌ 打包失敗: {e}")
    exit(1)

print("\n5. 驗證輸出...")
exe_path = os.path.join('dist', 'VoronoiDiagram', 'VoronoiDiagram.exe')
if os.name != 'nt':
    exe_path = os.path.join('dist', 'VoronoiDiagram', 'VoronoiDiagram')

if os.path.exists(exe_path):
    size_mb = os.path.getsize(exe_path) / (1024 * 1024)
    print(f"   ✅ 可執行檔已生成")
    print(f"   📂 位置: {exe_path}")
    print(f"   📊 大小: {size_mb:.1f} MB")
else:
    print(f"   ❌ 找不到可執行檔")
    exit(1)

print("\n6. 複製測試檔案...")
dist_dir = os.path.join('dist', 'VoronoiDiagram')

test_files = [
    'test_input.txt',
    'test_no_comment.txt',
    'test_with_comment.txt',
    'README.txt'
]

for file in test_files:
    if os.path.exists(file):
        shutil.copy(file, dist_dir)
        print(f"   ✅ {file}")

print("\n" + "=" * 70)
print("✅ 完成！")
print("=" * 70)
print(f"\n📂 可執行檔位置: dist/VoronoiDiagram/")
print(f"\n📝 測試方法:")
print(f"   1. 進入 dist/VoronoiDiagram/ 目錄")
print(f"   2. 雙擊 VoronoiDiagram.exe")
print(f"   3. 或在命令列執行: VoronoiDiagram.exe")
print(f"\n📦 繳交時請壓縮整個 VoronoiDiagram 目錄")
print("=" * 70)
