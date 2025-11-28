# Voronoi Diagram - Divide and Conquer Implementation

## 專案說明
這是一個使用 Divide-and-Conquer 方法實作的 Voronoi Diagram 程式，包含完整的 GUI 介面。

## 檔案結構
```
voronoi_geometry.py     # 幾何基礎類別（Point, Edge, VoronoiDiagram 等）
voronoi_algorithm.py    # Divide-and-Conquer 核心演算法
voronoi_gui.py          # GUI 主程式
test_input.txt          # 測試輸入檔案範例
README.md               # 說明文件
```

## 執行方式

### 1. 啟動程式
```bash
python voronoi_gui.py
```

### 2. 輸入點的方式

#### 方法一：滑鼠點擊
- 直接在畫布上點擊滑鼠左鍵，即可添加點

#### 方法二：載入輸入檔案
1. 點選選單：File -> Open Input File
2. 選擇輸入檔案（格式見下方說明）

### 3. 執行演算法

#### Run 模式
- 點擊 "Run" 按鈕，一次執行完整的演算法並顯示最終結果

#### Step by Step 模式
1. 點擊 "Step by Step" 按鈕
2. 每次點擊會顯示一次 merge 操作
3. 左側 Voronoi diagram 顯示為藍色
4. 右側 Voronoi diagram 顯示為綠色
5. 新增的 merging edges 顯示為紅色
6. 可以在任何步驟點擊 "Run" 繼續執行到結束

#### Reset
- 點擊 "Reset" 按鈕重置演算法狀態，但保留所有點

### 4. 儲存結果
1. 執行完演算法後，點選：File -> Save Output File
2. 輸出檔案格式符合規格要求

### 5. 載入輸出檔案
1. 點選：File -> Load Output File
2. 可以顯示之前儲存的 Voronoi diagram

## 檔案格式

### 輸入檔案格式
```
# 註解行（以 # 開頭）
x1 y1
x2 y2
x3 y3
...
```

範例：
```
103 200
193 64
193 370
283 200
```

### 輸出檔案格式
```
P x1 y1
P x2 y2
...
E x1 y1 x2 y2
E x1 y1 x2 y2
...
```

說明：
- P 開頭：座標點，按 lexical order 排序
- E 開頭：線段，格式為起點和終點座標
  - 滿足 x1 ≤ x2，或 x1 = x2 且 y1 ≤ y2
  - 按 lexical order 排序

範例：
```
P 103 200
P 193 64
P 193 370
P 283 200
E 0 34 193 161
E 0 363 193 261
E 193 161 193 261
E 193 161 437 0
E 193 261 600 476
```

## 演算法說明

### Divide-and-Conquer 方法

1. **DIVIDE（分割）**
   - 將點集按 x 座標排序
   - 從中間分成左右兩半

2. **CONQUER（征服）**
   - 遞迴解決左右兩側的 Voronoi diagram
   - Base cases：
     - 1 點：整個平面
     - 2 點：一條中垂線
     - 3 點：三條中垂線相交

3. **MERGE（合併）**
   - 找到左右兩側 convex hull 的上下切線
   - 從下切線開始構造 hyperplane Voronoi diagram
   - 刪除被 merging edge 切斷的舊邊

### 時間複雜度
- O(n log n)

## 功能特色

✅ 完整的 GUI 介面（600x600 畫布）
✅ Run 和 Step-by-step 兩種執行模式
✅ 滑鼠直接點擊輸入
✅ 讀取/儲存輸入檔案
✅ 讀取/顯示輸出檔案
✅ 支援 4 點以上的一般情況
✅ 輸出格式符合規格要求（lexical order）

## 注意事項

1. 畫布大小為 600x600
2. 座標範圍：0 ≤ x, y ≤ 600
3. 至少需要 2 個點才能執行演算法
4. 輸出檔案中的座標會轉換為整數
5. 特殊情況（如多點共線）的處理尚未完全實作

## 開發狀態

### 已實作
- ✅ 基本幾何類別
- ✅ Divide-and-Conquer 框架
- ✅ Base cases (1, 2, 3 點)
- ✅ GUI 介面
- ✅ 檔案讀寫
- ✅ Step-by-step 模式

### 待完善
- ⚠️ Merge 步驟的完整實作（特別是 hyperplane Voronoi diagram 的構造）
- ⚠️ 上下切線的精確計算
- ⚠️ 舊邊的刪除邏輯
- ⚠️ 特殊情況處理（共線、重複點等）

## 依賴套件
- Python 3.7+
- tkinter（通常內建於 Python）
- 無需其他第三方套件

## 作者
Muci - 2025 CSE544 Final Project
