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
        
        # 繪製左側（藍色）
        if step_data['left']:
            self.draw_voronoi_diagram(step_data['left'], edge_color='blue', point_color='blue')
        
        # 繪製右側（綠色）
        if step_data['right']:
            self.draw_voronoi_diagram(step_data['right'], edge_color='green', point_color='green')
        
        # 繪製合併結果（紅色）
        if step_data['merged']:
            for edge in step_data['merged'].edges:
                if edge not in step_data['left'].edges and edge not in step_data['right'].edges:
                    self.draw_edge(edge, color='red', width=3)
        
        # 更新步驟資訊
        self.current_step += 1
        self.step_info.config(text=f"Step: {self.current_step} / {len(self.algorithm.steps)}")
        self.status_bar.config(text=f"Step {self.current_step}: Merging left and right Voronoi diagrams")
    
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
        """載入輸入檔案"""
        filename = filedialog.askopenfilename(
            title="Select Input File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # 清空現有資料
            self.clear_all()
            
            # 解析檔案
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
                    messagebox.showinfo("Info", "讀入點數為零，檔案測試停止")
                    break
                
                # 讀取 n 個點
                points_in_group = []
                for j in range(n):
                    if i >= len(lines):
                        break
                    
                    point_line = lines[i].strip()
                    i += 1
                    
                    # 跳過註解
                    if point_line.startswith('#'):
                        j -= 1
                        continue
                    
                    parts = point_line.split()
                    if len(parts) >= 2:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            point = Point(x, y)
                            points_in_group.append(point)
                        except ValueError:
                            continue
                
                # 添加這一組點
                self.points.extend(points_in_group)
                
                # 只讀取第一組資料（如果要支援多組測試，可以改成迴圈）
                # 這裡先實作單組測試
                break
            
            # 繪製所有點
            for point in self.points:
                self.draw_point(point)
            
            self.update_points_list()
            self.status_bar.config(text=f"Loaded {len(self.points)} points from {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
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
            with open(filename, 'w') as f:
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
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # 清空現有資料
            self.clear_all()
            
            points = []
            edges = []
            
            # 解析檔案
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if parts[0] == 'P' and len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    points.append(Point(x, y))
                elif parts[0] == 'E' and len(parts) >= 5:
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