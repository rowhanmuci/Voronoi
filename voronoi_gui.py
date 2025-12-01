"""
Voronoi Diagram - GUI 主程式
支援滑鼠點擊輸入、檔案讀寫、Run/Step-by-step 模式
"""
import tkinter as tk
from tkinter import filedialog, messagebox, Menu
from typing import List, Optional
from voronoi_geometry import Point, Edge, VoronoiDiagram
from voronoi_algorithm import VoronoiDC


class VoronoiGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Voronoi Diagram - Divide and Conquer")
        self.root.geometry("900x700")
        
        # 畫布設定
        self.canvas_width = 600
        self.canvas_height = 600
        
        # 資料
        self.points: List[Point] = []
        self.current_vd: Optional[VoronoiDiagram] = None
        self.algorithm = VoronoiDC(self.canvas_width, self.canvas_height)
        self.current_step = 0
        self.step_mode = False
        
        # 建立 UI
        self._create_menu()
        self._create_canvas()
        self._create_controls()
        self._create_status_bar()
    
    def _create_menu(self):
        """創建選單列"""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        # File 選單
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Input File", command=self.load_input_file)
        file_menu.add_command(label="Save Output File", command=self.save_output_file)
        file_menu.add_command(label="Load Output File", command=self.load_output_file)
        file_menu.add_separator()
        file_menu.add_command(label="Clear All", command=self.clear_all)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
    
    def _create_canvas(self):
        """創建畫布"""
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg='white',
            borderwidth=2,
            relief='solid'
        )
        self.canvas.pack()
        
        # 綁定滑鼠點擊事件
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        
        # 添加座標軸標籤
        label_frame = tk.Frame(canvas_frame)
        label_frame.pack(pady=5)
        tk.Label(label_frame, text=f"Canvas: {self.canvas_width} x {self.canvas_height}").pack()
    
    def _create_controls(self):
        """創建控制面板"""
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH)
        
        # 標題
        tk.Label(control_frame, text="Control Panel", font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Run 按鈕
        self.run_btn = tk.Button(
            control_frame,
            text="Run",
            command=self.run_algorithm,
            bg='lightgreen',
            font=('Arial', 12),
            width=20,
            height=2
        )
        self.run_btn.pack(pady=5)
        
        # Step by Step 按鈕
        self.step_btn = tk.Button(
            control_frame,
            text="Step by Step",
            command=self.step_algorithm,
            bg='lightblue',
            font=('Arial', 12),
            width=20,
            height=2
        )
        self.step_btn.pack(pady=5)
        
        # Reset 按鈕
        self.reset_btn = tk.Button(
            control_frame,
            text="Reset",
            command=self.reset_algorithm,
            bg='lightyellow',
            font=('Arial', 12),
            width=20
        )
        self.reset_btn.pack(pady=5)
        
        # Next Test 按鈕
        self.next_test_btn = tk.Button(
            control_frame,
            text="Next Test",
            command=self.next_test_group,
            bg='lightcyan',
            font=('Arial', 12),
            width=20
        )
        self.next_test_btn.pack(pady=5)
        
        # 分隔線
        tk.Frame(control_frame, height=2, bg='gray').pack(fill=tk.X, pady=20)
        
        # 點數資訊
        tk.Label(control_frame, text="Points Information", font=('Arial', 12, 'bold')).pack(pady=5)
        
        self.points_info = tk.Label(
            control_frame,
            text="Points: 0",
            font=('Arial', 10),
            justify=tk.LEFT,
            anchor='w'
        )
        self.points_info.pack(pady=5, fill=tk.X)
        
        # 步驟資訊
        self.step_info = tk.Label(
            control_frame,
            text="Step: 0 / 0",
            font=('Arial', 10),
            justify=tk.LEFT,
            anchor='w'
        )
        self.step_info.pack(pady=5, fill=tk.X)
        
        # 點列表
        tk.Label(control_frame, text="Point List:", font=('Arial', 10, 'bold')).pack(pady=5)
        
        # 滾動列表
        list_frame = tk.Frame(control_frame)
        list_frame.pack(pady=5, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.points_listbox = tk.Listbox(
            list_frame,
            font=('Courier', 9),
            yscrollcommand=scrollbar.set
        )
        self.points_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.points_listbox.yview)
    
    def _create_status_bar(self):
        """創建狀態列"""
        self.status_bar = tk.Label(
            self.root,
            text="Ready. Click on canvas to add points or load input file.",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def on_canvas_click(self, event):
        """處理畫布點擊事件"""
        x, y = event.x, event.y
        
        # 檢查是否在邊界內
        if 0 <= x <= self.canvas_width and 0 <= y <= self.canvas_height:
            point = Point(x, y)
            self.points.append(point)
            self.draw_point(point)
            self.update_points_list()
            self.status_bar.config(text=f"Added point: ({x}, {y})")
    
    def draw_point(self, point: Point, color='red', size=4):
        """在畫布上繪製點"""
        x, y = point.x, point.y
        self.canvas.create_oval(
            x - size, y - size, x + size, y + size,
            fill=color, outline='black', width=1
        )
    
    def draw_edge(self, edge: Edge, color='blue', width=2):
        """在畫布上繪製邊"""
        if edge.start and edge.end:
            self.canvas.create_line(
                edge.start.x, edge.start.y,
                edge.end.x, edge.end.y,
                fill=color, width=width
            )
    
    def draw_convex_hull(self, hull: list, color='purple', width=2, dash=(5, 3)):
        """在畫布上繪製凸包"""
        if len(hull) < 2:
            return
        
        n = len(hull)
        for i in range(n):
            p1 = hull[i]
            p2 = hull[(i + 1) % n]
            self.canvas.create_line(
                p1.x, p1.y, p2.x, p2.y,
                fill=color, width=width, dash=dash
            )
    
    def draw_voronoi_diagram(self, vd: VoronoiDiagram, edge_color='blue', point_color='red'):
        """繪製完整的 Voronoi diagram"""
        # 繪製邊
        for edge in vd.edges:
            self.draw_edge(edge, color=edge_color)
        
        # 繪製頂點
        for vertex in vd.vertices:
            self.draw_point(vertex, color='green', size=3)
        
        # 繪製站點
        for site in vd.sites:
            self.draw_point(site, color=point_color, size=5)
    
    def clear_canvas(self):
        """清空畫布"""
        self.canvas.delete('all')
    
    def clear_all(self):
        """清空所有資料"""
        self.points.clear()
        self.current_vd = None
        self.current_step = 0
        self.step_mode = False
        self.clear_canvas()
        self.update_points_list()
        self.status_bar.config(text="Cleared all data.")
    
    def update_points_list(self):
        """更新點列表顯示"""
        self.points_listbox.delete(0, tk.END)
        for i, p in enumerate(self.points):
            self.points_listbox.insert(tk.END, f"{i+1}. ({p.x:.1f}, {p.y:.1f})")
        
        self.points_info.config(text=f"Points: {len(self.points)}")
    
    def run_algorithm(self):
        """執行完整演算法"""
        if len(self.points) < 2:
            messagebox.showwarning("Warning", "Need at least 2 points!")
            return
        
        # 從當前步驟繼續執行到結束
        if self.step_mode and self.current_step < len(self.algorithm.steps):
            # 如果在 step-by-step 模式中，繼續執行剩餘步驟
            while self.current_step < len(self.algorithm.steps):
                self.step_algorithm()
        else:
            # 重新執行完整演算法
            self.current_step = 0
            self.step_mode = False
            self.clear_canvas()
            
            self.status_bar.config(text="Running algorithm...")
            self.root.update()
            
            # 建構 Voronoi diagram
            self.current_vd = self.algorithm.build(self.points)
            
            # 繪製結果
            self.draw_voronoi_diagram(self.current_vd)
            
            self.status_bar.config(text=f"Algorithm completed. {len(self.current_vd.edges)} edges generated.")
    
    def step_algorithm(self):
        """單步執行演算法"""
        if len(self.points) < 2:
            messagebox.showwarning("Warning", "Need at least 2 points!")
            return
        
        # 第一次按下 Step by Step
        if not self.step_mode:
            self.step_mode = True
            self.current_step = 0
            self.current_vd = self.algorithm.build(self.points)
        
        # 檢查是否還有步驟
        if self.current_step >= len(self.algorithm.steps):
            messagebox.showinfo("Info", "Algorithm completed!")
            return
        
        # 獲取當前步驟
        step_data = self.algorithm.steps[self.current_step]
        
        # 清空畫布並繪製當前狀態
        self.clear_canvas()
        
        # 繪製左側凸包（藍色虛線）
        if 'left_hull' in step_data and step_data['left_hull']:
            self.draw_convex_hull(step_data['left_hull'], color='blue', width=2, dash=(5, 3))
        
        # 繪製右側凸包（綠色虛線）
        if 'right_hull' in step_data and step_data['right_hull']:
            self.draw_convex_hull(step_data['right_hull'], color='green', width=2, dash=(5, 3))
        
        # 繪製左側 Voronoi（藍色）
        if step_data['left']:
            self.draw_voronoi_diagram(step_data['left'], edge_color='blue', point_color='blue')
        
        # 繪製右側 Voronoi（綠色）
        if step_data['right']:
            self.draw_voronoi_diagram(step_data['right'], edge_color='green', point_color='green')
        
        # 繪製合併後的凸包（紫色虛線）
        if 'merged_hull' in step_data and step_data['merged_hull']:
            self.draw_convex_hull(step_data['merged_hull'], color='purple', width=2, dash=(3, 2))
        
        # 繪製 hyperplane / dividing chain（紅色）
        if step_data['merged']:
            left_edge_set = set()
            right_edge_set = set()
            
            if step_data['left']:
                for edge in step_data['left'].edges:
                    if edge.start and edge.end:
                        left_edge_set.add((round(edge.start.x, 2), round(edge.start.y, 2),
                                          round(edge.end.x, 2), round(edge.end.y, 2)))
                        left_edge_set.add((round(edge.end.x, 2), round(edge.end.y, 2),
                                          round(edge.start.x, 2), round(edge.start.y, 2)))
            
            if step_data['right']:
                for edge in step_data['right'].edges:
                    if edge.start and edge.end:
                        right_edge_set.add((round(edge.start.x, 2), round(edge.start.y, 2),
                                           round(edge.end.x, 2), round(edge.end.y, 2)))
                        right_edge_set.add((round(edge.end.x, 2), round(edge.end.y, 2),
                                           round(edge.start.x, 2), round(edge.start.y, 2)))
            
            for edge in step_data['merged'].edges:
                if edge.start and edge.end:
                    edge_tuple = (round(edge.start.x, 2), round(edge.start.y, 2),
                                 round(edge.end.x, 2), round(edge.end.y, 2))
                    
                    if edge_tuple not in left_edge_set and edge_tuple not in right_edge_set:
                        self.draw_edge(edge, color='red', width=3)
        
        # 更新步驟資訊
        self.current_step += 1
        self.step_info.config(text=f"Step: {self.current_step} / {len(self.algorithm.steps)}")
        
        left_count = len(step_data['left'].sites) if step_data['left'] else 0
        right_count = len(step_data['right'].sites) if step_data['right'] else 0
        self.status_bar.config(
            text=f"Step {self.current_step}: Merging L({left_count} pts, blue) + R({right_count} pts, green) | "
                 f"Convex hulls shown as dashed lines | Hyperplane in red"
        )
    
    def reset_algorithm(self):
        """重置演算法狀態"""
        self.current_step = 0
        self.step_mode = False
        self.clear_canvas()
        
        # 重新繪製所有點
        for point in self.points:
            self.draw_point(point)
        
        self.step_info.config(text="Step: 0 / 0")
        self.status_bar.config(text="Algorithm reset.")
    
    def load_input_file(self):
        """載入輸入檔案（支援多組測試資料）"""
        filename = filedialog.askopenfilename(
            title="Select Input File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # 嘗試多種編碼
            lines = None
            encodings = ['utf-8', 'cp950', 'big5', 'gbk', 'latin-1']
            
            for encoding in encodings:
                try:
                    with open(filename, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    break  # 成功讀取，跳出迴圈
                except (UnicodeDecodeError, LookupError):
                    continue
            
            if lines is None:
                messagebox.showerror("Error", "無法讀取檔案，請檢查檔案編碼")
                return
            
            # 解析檔案，讀取所有組測試資料
            test_groups = []
            i = 0
            
            while i < len(lines):
                line = lines[i].strip()
                i += 1
                
                # 跳過註解和空行
                if not line or line.startswith('#'):
                    continue
                
                # 讀取點數 n
                try:
                    n = int(line)
                except ValueError:
                    continue
                
                # 如果 n = 0，結束讀取
                if n == 0:
                    break
                
                # 讀取 n 個點（使用 while 迴圈）
                points_in_group = []
                points_read = 0
                
                while points_read < n and i < len(lines):
                    point_line = lines[i].strip()
                    i += 1
                    
                    # 跳過註解和空行
                    if not point_line or point_line.startswith('#'):
                        continue
                    
                    parts = point_line.split()
                    if len(parts) >= 2:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            point = Point(x, y)
                            points_in_group.append(point)
                            points_read += 1
                        except ValueError:
                            continue
                
                # 儲存這一組測試資料
                if points_in_group:
                    test_groups.append(points_in_group)
            
            # 儲存測試資料組並開始顯示第一組
            if test_groups:
                self.test_groups = test_groups
                self.current_test_index = 0
                self.show_current_test_group()
            else:
                messagebox.showwarning("Warning", "No valid test data found in file")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def show_current_test_group(self):
        """顯示當前測試組"""
        if not hasattr(self, 'test_groups') or not self.test_groups:
            return
        
        if self.current_test_index >= len(self.test_groups):
            messagebox.showinfo("Info", "已完成所有測試組")
            return
        
        # 清空並載入當前測試組
        self.clear_all()
        current_group = self.test_groups[self.current_test_index]
        self.points = current_group.copy()
        
        # 繪製所有點
        for point in self.points:
            self.draw_point(point)
        
        self.update_points_list()
        
        # 更新狀態列
        total_groups = len(self.test_groups)
        current_num = self.current_test_index + 1
        self.status_bar.config(
            text=f"Test {current_num}/{total_groups}: Loaded {len(self.points)} points. "
                 f"Run algorithm, then press 'Next Test' for next group."
        )
        
        # 顯示訊息框告知使用者
        coords_str = "\n".join([f"  ({p.x}, {p.y})" for p in self.points])
        messagebox.showinfo(
            "Test Data Loaded",
            f"測試組 {current_num}/{total_groups}\n"
            f"點數: {len(self.points)}\n"
            f"座標:\n{coords_str}\n\n"
            f"請執行演算法後按 'Next Test' 繼續下一組"
        )
    
    def next_test_group(self):
        """載入下一組測試資料"""
        if not hasattr(self, 'test_groups') or not self.test_groups:
            messagebox.showwarning("Warning", "No test data loaded. Please load input file first.")
            return
        
        self.current_test_index += 1
        
        if self.current_test_index >= len(self.test_groups):
            messagebox.showinfo("Info", "所有測試組已完成！")
            self.current_test_index = len(self.test_groups) - 1  # 停在最後一組
            return
        
        self.show_current_test_group()
    
    def save_output_file(self):
        """儲存輸出檔案"""
        if not self.current_vd:
            messagebox.showwarning("Warning", "No Voronoi diagram to save. Run algorithm first!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Output File",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # 使用 UTF-8 without BOM
            with open(filename, 'w', encoding='utf-8') as f:
                # 輸出座標點 (按 lexical order 排序)
                sorted_points = sorted(self.points, key=lambda p: (p.x, p.y))
                for p in sorted_points:
                    f.write(f"P {int(p.x)} {int(p.y)}\n")
                
                # 輸出線段 (按 lexical order 排序)
                edges_data = []
                for edge in self.current_vd.edges:
                    if edge.start and edge.end:
                        tuple_data = edge.to_tuple()
                        if tuple_data:
                            edges_data.append(tuple_data)
                
                # 排序
                edges_data.sort()
                
                for x1, y1, x2, y2 in edges_data:
                    f.write(f"E {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n")
            
            self.status_bar.config(text=f"Saved output to {filename}")
            messagebox.showinfo("Success", f"Output saved to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {str(e)}")
    
    def load_output_file(self):
        """載入輸出檔案並顯示"""
        filename = filedialog.askopenfilename(
            title="Select Output File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # 嘗試多種編碼，優先使用 utf-8-sig 處理 BOM
            lines = None
            encodings = ['utf-8-sig', 'utf-8', 'cp950', 'big5', 'gbk', 'latin-1']
            
            for encoding in encodings:
                try:
                    with open(filename, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            
            if lines is None:
                messagebox.showerror("Error", "無法讀取檔案，請檢查檔案編碼")
                return
            
            # 清空現有資料
            self.clear_all()
            
            points = []
            edges = []
            
            # 解析檔案
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 移除可能的 BOM
                if line.startswith('\ufeff'):
                    line = line[1:]
                
                parts = line.split()
                if len(parts) < 3:
                    continue
                
                # 檢查第一個字元（可能有 BOM）
                type_char = parts[0].lstrip('\ufeff')
                
                if type_char == 'P' and len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    points.append(Point(x, y))
                elif type_char == 'E' and len(parts) >= 5:
                    x1, y1 = float(parts[1]), float(parts[2])
                    x2, y2 = float(parts[3]), float(parts[4])
                    edge = Edge(Point(x1, y1), Point(x2, y2))
                    edges.append(edge)
            
            # 更新資料
            self.points = points
            
            # 建立 Voronoi diagram
            self.current_vd = VoronoiDiagram()
            self.current_vd.sites = points
            self.current_vd.edges = edges
            
            # 繪製
            self.draw_voronoi_diagram(self.current_vd)
            self.update_points_list()
            
            self.status_bar.config(text=f"Loaded output file: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")


def main():
    root = tk.Tk()
    app = VoronoiGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()