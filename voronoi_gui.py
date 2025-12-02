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
        
        self.canvas_width = 600
        self.canvas_height = 600
        
        self.points: List[Point] = []
        self.current_vd: Optional[VoronoiDiagram] = None
        self.algorithm = VoronoiDC(self.canvas_width, self.canvas_height)
        self.current_step = 0
        self.step_mode = False
        
        self._create_menu()
        self._create_canvas()
        self._create_controls()
        self._create_status_bar()
    
    def _create_menu(self):
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
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
        
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        
        label_frame = tk.Frame(canvas_frame)
        label_frame.pack(pady=5)
        tk.Label(label_frame, text=f"Canvas: {self.canvas_width} x {self.canvas_height}").pack()
    
    def _create_controls(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH)
        
        tk.Label(control_frame, text="Control Panel", font=('Arial', 14, 'bold')).pack(pady=10)
        
        self.run_btn = tk.Button(
            control_frame, text="Run", command=self.run_algorithm,
            bg='lightgreen', font=('Arial', 12), width=20, height=2
        )
        self.run_btn.pack(pady=5)
        
        self.step_btn = tk.Button(
            control_frame, text="Step by Step", command=self.step_algorithm,
            bg='lightblue', font=('Arial', 12), width=20, height=2
        )
        self.step_btn.pack(pady=5)
        
        self.reset_btn = tk.Button(
            control_frame, text="Reset", command=self.reset_algorithm,
            bg='lightyellow', font=('Arial', 12), width=20
        )
        self.reset_btn.pack(pady=5)
        
        self.next_test_btn = tk.Button(
            control_frame, text="Next Test", command=self.next_test_group,
            bg='lightcyan', font=('Arial', 12), width=20
        )
        self.next_test_btn.pack(pady=5)
        
        tk.Frame(control_frame, height=2, bg='gray').pack(fill=tk.X, pady=20)
        
        tk.Label(control_frame, text="Points Information", font=('Arial', 12, 'bold')).pack(pady=5)
        
        self.points_info = tk.Label(control_frame, text="Points: 0", font=('Arial', 10), justify=tk.LEFT, anchor='w')
        self.points_info.pack(pady=5, fill=tk.X)
        
        self.step_info = tk.Label(control_frame, text="Step: 0 / 0", font=('Arial', 10), justify=tk.LEFT, anchor='w')
        self.step_info.pack(pady=5, fill=tk.X)
        
        tk.Label(control_frame, text="Point List:", font=('Arial', 10, 'bold')).pack(pady=5)
        
        list_frame = tk.Frame(control_frame)
        list_frame.pack(pady=5, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.points_listbox = tk.Listbox(list_frame, font=('Courier', 9), yscrollcommand=scrollbar.set)
        self.points_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.points_listbox.yview)
    def _render_left_vd(self, step_data):
        """渲染左側 Voronoi"""
        vd = step_data['vd']
        # 只畫邊（藍色）
        for edge in vd.edges:
            self.draw_edge(edge, color='blue', width=2)
        
        # 突顯左側正在處理的點（大一點，藍色）
        for site in step_data['sites']:
            self.draw_point(site, color='blue', size=6)

    def _render_right_vd(self, step_data):
        """渲染右側 Voronoi"""
        vd = step_data['vd']
        # 只畫邊（綠色）
        for edge in vd.edges:
            self.draw_edge(edge, color='green', width=2)
        
        # 突顯右側正在處理的點（大一點，綠色）
        for site in step_data['sites']:
            self.draw_point(site, color='green', size=6)

    def _render_left_hull(self, step_data):
        """渲染左側 Convex Hull"""
        # 畫凸包（虛線紫色）
        self.draw_convex_hull(step_data['hull'], color='purple', width=2, dash=(5, 3))

    def _render_right_hull(self, step_data):
        """渲染右側 Convex Hull"""
        # 畫凸包（虛線紫色）
        self.draw_convex_hull(step_data['hull'], color='purple', width=2, dash=(5, 3))

    def _render_merged_hull(self, step_data):
        """渲染合併後的 Convex Hull"""
        # 畫合併凸包（實線紫色）
        self.draw_convex_hull(step_data['hull'], color='purple', width=3)

    def _render_tangents(self, step_data):
        """渲染上下切線"""
        # 畫合併凸包（淡色）
        self.draw_convex_hull(step_data['merged_hull'], color='lightgray', width=1)
        
        # 畫上切線（橙色粗線）
        upper_left, upper_right = step_data['upper_tangent']
        self.canvas.create_line(
            upper_left.x, upper_left.y,
            upper_right.x, upper_right.y,
            fill='orange', width=4
        )
        
        # 畫下切線（橙色粗線）
        lower_left, lower_right = step_data['lower_tangent']
        self.canvas.create_line(
            lower_left.x, lower_left.y,
            lower_right.x, lower_right.y,
            fill='orange', width=4
        )

    def _render_hyperplane(self, step_data):
        """渲染 Hyper Plane"""
        # 畫左右 VD（淡色）
        left_vd = step_data['left_vd']
        for edge in left_vd.edges:
            self.draw_edge(edge, color='lightblue', width=1)
        
        right_vd = step_data['right_vd']
        for edge in right_vd.edges:
            self.draw_edge(edge, color='lightgreen', width=1)
        
        # 畫 Hyper Plane（紅色粗線）
        for edge in step_data['hyperplane']:
            self.draw_edge(edge, color='red', width=3)

    def _render_after_elimination(self, step_data):
        """渲染消線後的結果"""
        # 畫被刪除的邊（灰色虛線）
        original_left = step_data['original_left_edges']
        current_left = step_data['left_edges']
        
        # 簡化比較：用端點坐標
        current_left_coords = set()
        for e in current_left:
            if e.start and e.end:
                coord = ((e.start.x, e.start.y), (e.end.x, e.end.y))
                current_left_coords.add(coord)
        
        for edge in original_left:
            if edge.start and edge.end:
                coord = ((edge.start.x, edge.start.y), (edge.end.x, edge.end.y))
                if coord not in current_left_coords:
                    self.draw_edge(edge, color='gray', width=1, dash=(2, 2))
        
        # 同樣處理右側
        original_right = step_data['original_right_edges']
        current_right = step_data['right_edges']
        
        current_right_coords = set()
        for e in current_right:
            if e.start and e.end:
                coord = ((e.start.x, e.start.y), (e.end.x, e.end.y))
                current_right_coords.add(coord)
        
        for edge in original_right:
            if edge.start and edge.end:
                coord = ((edge.start.x, edge.start.y), (edge.end.x, edge.end.y))
                if coord not in current_right_coords:
                    self.draw_edge(edge, color='gray', width=1, dash=(2, 2))
        
        # 畫保留的邊
        for edge in current_left:
            self.draw_edge(edge, color='blue', width=2)
        for edge in current_right:
            self.draw_edge(edge, color='green', width=2)
        
        # 畫 Hyper Plane
        for edge in step_data['hyperplane']:
            self.draw_edge(edge, color='red', width=3)

    def _render_merged_result(self, step_data):
        """渲染最終合併結果"""
        merged_vd = step_data['merged_vd']
        
        # 畫所有邊（深綠色）
        for edge in merged_vd.edges:
            self.draw_edge(edge, color='darkgreen', width=2)
        
        # 畫合併後的 convex hull
        if 'merged_hull' in step_data:
            self.draw_convex_hull(step_data['merged_hull'], color='gray', width=1, dash=(2, 2))

    def draw_edge(self, edge, color='blue', width=2, dash=None):
        """畫邊，支援虛線"""
        if edge.start and edge.end:
            if dash:
                self.canvas.create_line(
                    edge.start.x, edge.start.y,
                    edge.end.x, edge.end.y,
                    fill=color, width=width, dash=dash
                )
            else:
                self.canvas.create_line(
                    edge.start.x, edge.start.y,
                    edge.end.x, edge.end.y,
                    fill=color, width=width
                )
    def _create_status_bar(self):
        self.status_bar = tk.Label(
            self.root, text="Ready. Click on canvas to add points or load input file.",
            bd=1, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def on_canvas_click(self, event):
        x, y = event.x, event.y
        if 0 <= x <= self.canvas_width and 0 <= y <= self.canvas_height:
            point = Point(x, y)
            self.points.append(point)
            self.draw_point(point)
            self.update_points_list()
            self.status_bar.config(text=f"Added point: ({x}, {y})")
    
    def draw_point(self, point: Point, color='red', size=4):
        x, y = point.x, point.y
        self.canvas.create_oval(x - size, y - size, x + size, y + size, fill=color, outline='black', width=1)
    
    
    def draw_convex_hull(self, hull: list, color='purple', width=2, dash=None):
        """畫 convex hull
        
        Args:
            hull: Point 列表
            color: 顏色
            width: 線寬
            dash: 虛線樣式
        """
        if len(hull) < 3:
            return
        
        for i in range(len(hull)):
            p1 = hull[i]
            p2 = hull[(i + 1) % len(hull)]
            if dash:
                self.canvas.create_line(
                    p1.x, p1.y, p2.x, p2.y,
                    fill=color, width=width, dash=dash
                )
            else:
                self.canvas.create_line(
                    p1.x, p1.y, p2.x, p2.y,
                    fill=color, width=width
                )
    
    def draw_voronoi_diagram(self, vd: VoronoiDiagram, edge_color='blue', point_color='red'):
        for edge in vd.edges:
            self.draw_edge(edge, color=edge_color)
        for vertex in vd.vertices:
            self.draw_point(vertex, color='green', size=3)
        for site in vd.sites:
            self.draw_point(site, color=point_color, size=5)
    
    def clear_canvas(self):
        self.canvas.delete('all')
    
    def clear_all(self):
        self.points.clear()
        self.current_vd = None
        self.current_step = 0
        self.step_mode = False
        self.clear_canvas()
        self.update_points_list()
        self.status_bar.config(text="Cleared all data.")
    
    def update_points_list(self):
        self.points_listbox.delete(0, tk.END)
        for i, p in enumerate(self.points):
            self.points_listbox.insert(tk.END, f"{i+1}. ({p.x:.1f}, {p.y:.1f})")
        self.points_info.config(text=f"Points: {len(self.points)}")
    
    def run_algorithm(self):
        if len(self.points) < 2:
            messagebox.showwarning("Warning", "Need at least 2 points!")
            return
        
        if self.step_mode and self.current_step < len(self.algorithm.steps):
            while self.current_step < len(self.algorithm.steps):
                self.step_algorithm()
        else:
            self.current_step = 0
            self.step_mode = False
            self.clear_canvas()
            
            self.status_bar.config(text="Running algorithm...")
            self.root.update()
            
            self.current_vd = self.algorithm.build(self.points)
            self.draw_voronoi_diagram(self.current_vd)
            
            self.status_bar.config(text=f"Algorithm completed. {len(self.current_vd.edges)} edges generated.")

    def _draw_all_sites(self):
        """永遠顯示所有點（所有演算法點，不只當前步驟的）"""
        for point in self.points:
            self.draw_point(point, color='black', size=4)

    def step_algorithm(self):
        """Step-by-step 顯示演算法過程"""
        if len(self.points) < 2:
            messagebox.showwarning("Warning", "Need at least 2 points!")
            return
        
        # 第一次按或重新開始
        if not self.step_mode or self.current_step >= len(self.algorithm.steps):
            self.step_mode = True
            self.current_step = 0
            
            # 執行演算法
            self.current_vd = self.algorithm.build(self.points)
            
            if len(self.algorithm.steps) == 0:
                messagebox.showinfo("Info", "No merge steps (too few points)")
                self.clear_canvas()
                self._draw_all_sites()  # 永遠顯示所有點
                self.draw_voronoi_diagram(self.current_vd)
                return
            
            # 重置狀態欄
            self.step_info.config(text=f"Step: 1 / {len(self.algorithm.steps)}")
        
        # 已經完成所有步驟
        if self.current_step >= len(self.algorithm.steps):
            # 顯示最終結果（消線後的圖）
            self.clear_canvas()
            self._draw_all_sites()  # 永遠顯示所有點
            
            # 取得最後一個 merge 的結果（消線後）
            last_step = self.algorithm.steps[-1]
            if last_step['type'] == 'show_merged_result':
                merged_vd = last_step['merged_vd']
                
                # 畫消線後的邊（深綠色）
                for edge in merged_vd.edges:
                    self.draw_edge(edge, color='darkgreen', width=2)
                
                # 畫最終 convex hull（紫色實線）
                if 'merged_hull' in last_step:
                    self.draw_convex_hull(last_step['merged_hull'], color='purple', width=2)
            
            self.status_bar.config(text="Algorithm completed! Click 'Step' again to restart.")
            self.step_info.config(text=f"Completed ({len(self.algorithm.steps)} steps)")
            
            # 下次點擊會重新開始
            self.step_mode = False
            return
        
        # 取得當前步驟
        step_data = self.algorithm.steps[self.current_step]
        step_type = step_data['type']
        
        # 清空畫布
        self.clear_canvas()
        
        # **永遠先畫所有點**
        self._draw_all_sites()
        
        # 根據步驟類型渲染
        if step_type == 'show_left_vd':
            self._render_left_vd(step_data)
        elif step_type == 'show_right_vd':
            self._render_right_vd(step_data)
        elif step_type == 'show_left_hull':
            self._render_left_hull(step_data)
        elif step_type == 'show_right_hull':
            self._render_right_hull(step_data)
        elif step_type == 'show_merged_hull':
            self._render_merged_hull(step_data)
        elif step_type == 'show_tangents':
            self._render_tangents(step_data)
        elif step_type == 'show_hyperplane':
            self._render_hyperplane(step_data)
        elif step_type == 'show_after_elimination':
            self._render_after_elimination(step_data)
        elif step_type == 'show_merged_result':
            self._render_merged_result(step_data)
        
        # 更新狀態
        self.current_step += 1
        self.step_info.config(text=f"Step: {self.current_step} / {len(self.algorithm.steps)}")
        self.status_bar.config(text=f"Step {self.current_step}: {step_data['description']}")
    
    def reset_algorithm(self):
        self.current_step = 0
        self.step_mode = False
        self.clear_canvas()
        
        for point in self.points:
            self.draw_point(point)
        
        self.step_info.config(text="Step: 0 / 0")
        self.status_bar.config(text="Algorithm reset.")
    
    def load_input_file(self):
        filename = filedialog.askopenfilename(
            title="Select Input File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            lines = None
            encodings = ['utf-8', 'cp950', 'big5', 'gbk', 'latin-1']
            
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
            
            test_groups = []
            i = 0
            
            while i < len(lines):
                line = lines[i].strip()
                i += 1
                
                if not line or line.startswith('#'):
                    continue
                
                try:
                    n = int(line)
                except ValueError:
                    continue
                
                if n == 0:
                    break
                
                points_in_group = []
                points_read = 0
                
                while points_read < n and i < len(lines):
                    point_line = lines[i].strip()
                    i += 1
                    
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
                
                if points_in_group:
                    test_groups.append(points_in_group)
            
            if test_groups:
                self.test_groups = test_groups
                self.current_test_index = 0
                self.show_current_test_group()
            else:
                messagebox.showwarning("Warning", "No valid test data found in file")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def show_current_test_group(self):
        if not hasattr(self, 'test_groups') or not self.test_groups:
            return
        
        if self.current_test_index >= len(self.test_groups):
            messagebox.showinfo("Info", "已完成所有測試組")
            return
        
        self.clear_all()
        current_group = self.test_groups[self.current_test_index]
        self.points = current_group.copy()
        
        for point in self.points:
            self.draw_point(point)
        
        self.update_points_list()
        
        total_groups = len(self.test_groups)
        current_num = self.current_test_index + 1
        self.status_bar.config(
            text=f"Test {current_num}/{total_groups}: Loaded {len(self.points)} points. "
                 f"Run algorithm, then press 'Next Test' for next group."
        )
        
        coords_str = "\n".join([f"  ({p.x}, {p.y})" for p in self.points[:10]])
        if len(self.points) > 10:
            coords_str += f"\n  ... and {len(self.points) - 10} more"
        messagebox.showinfo(
            "Test Data Loaded",
            f"測試組 {current_num}/{total_groups}\n"
            f"點數: {len(self.points)}\n"
            f"座標:\n{coords_str}\n\n"
            f"請執行演算法後按 'Next Test' 繼續下一組"
        )
    
    def next_test_group(self):
        if not hasattr(self, 'test_groups') or not self.test_groups:
            messagebox.showwarning("Warning", "No test data loaded. Please load input file first.")
            return
        
        self.current_test_index += 1
        
        if self.current_test_index >= len(self.test_groups):
            messagebox.showinfo("Info", "所有測試組已完成！")
            self.current_test_index = len(self.test_groups) - 1
            return
        
        self.show_current_test_group()
    
    def save_output_file(self):
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
            with open(filename, 'w', encoding='utf-8') as f:
                sorted_points = sorted(self.points, key=lambda p: (p.x, p.y))
                for p in sorted_points:
                    f.write(f"P {int(p.x)} {int(p.y)}\n")
                
                edges_data = []
                for edge in self.current_vd.edges:
                    if edge.start and edge.end:
                        tuple_data = edge.to_tuple()
                        if tuple_data:
                            edges_data.append(tuple_data)
                
                edges_data.sort()
                
                for x1, y1, x2, y2 in edges_data:
                    f.write(f"E {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n")
            
            self.status_bar.config(text=f"Saved output to {filename}")
            messagebox.showinfo("Success", f"Output saved to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {str(e)}")
    
    def load_output_file(self):
        filename = filedialog.askopenfilename(
            title="Select Output File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
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
            
            self.clear_all()
            
            points = []
            edges = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('\ufeff'):
                    line = line[1:]
                
                parts = line.split()
                if len(parts) < 3:
                    continue
                
                type_char = parts[0].lstrip('\ufeff')
                
                if type_char == 'P' and len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    points.append(Point(x, y))
                elif type_char == 'E' and len(parts) >= 5:
                    x1, y1 = float(parts[1]), float(parts[2])
                    x2, y2 = float(parts[3]), float(parts[4])
                    edge = Edge(Point(x1, y1), Point(x2, y2))
                    edges.append(edge)
            
            self.points = points
            
            self.current_vd = VoronoiDiagram()
            self.current_vd.sites = points
            self.current_vd.edges = edges
            
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