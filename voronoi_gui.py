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
    
    def draw_edge(self, edge: Edge, color='blue', width=2):
        if edge.start and edge.end:
            self.canvas.create_line(edge.start.x, edge.start.y, edge.end.x, edge.end.y, fill=color, width=width)
    
    def draw_convex_hull(self, hull: list, color='purple', width=2, dash=(5, 3)):
        if len(hull) < 2:
            return
        n = len(hull)
        for i in range(n):
            p1 = hull[i]
            p2 = hull[(i + 1) % n]
            self.canvas.create_line(p1.x, p1.y, p2.x, p2.y, fill=color, width=width, dash=dash)
    
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
    
    def step_algorithm(self):
        if len(self.points) < 2:
            messagebox.showwarning("Warning", "Need at least 2 points!")
            return
        
        if not self.step_mode:
            self.step_mode = True
            self.current_step = 0
            self.current_vd = self.algorithm.build(self.points)
        
        if self.current_step >= len(self.algorithm.steps):
            messagebox.showinfo("Info", "Algorithm completed!")
            return
        
        step_data = self.algorithm.steps[self.current_step]
        self.clear_canvas()
        
        # 繪製 Convex Hulls (虛線)
        if 'left_hull' in step_data:
            self.draw_convex_hull(step_data['left_hull'], color='gray', width=1, dash=(4, 4))
        if 'right_hull' in step_data:
            self.draw_convex_hull(step_data['right_hull'], color='gray', width=1, dash=(4, 4))
        
        # 建立 hyperplane 集合
        hyperplane_edges = step_data.get('hyperplane', [])
        hp_set = set()
        for e in hyperplane_edges:
            if e.start and e.end:
                hp_set.add((round(e.start.x, 2), round(e.start.y, 2), round(e.end.x, 2), round(e.end.y, 2)))
                hp_set.add((round(e.end.x, 2), round(e.end.y, 2), round(e.start.x, 2), round(e.start.y, 2)))
        
        left_sites = set(step_data['left_sites'])
        
        if step_data['merged']:
            for edge in step_data['merged'].edges:
                if not edge.start or not edge.end:
                    continue
                
                edge_tup = (round(edge.start.x, 2), round(edge.start.y, 2), round(edge.end.x, 2), round(edge.end.y, 2))
                
                if edge_tup in hp_set:
                    self.draw_edge(edge, color='red', width=3)
                else:
                    is_left = False
                    if edge.site_left and edge.site_left in left_sites:
                        is_left = True
                    elif edge.site_right and edge.site_right in left_sites:
                        is_left = True
                    
                    if is_left:
                        self.draw_edge(edge, color='blue', width=2)
                    else:
                        self.draw_edge(edge, color='green', width=2)
        
        # 繪製站點
        if step_data['merged']:
            for site in step_data['merged'].sites:
                color = 'blue' if site in left_sites else 'green'
                self.draw_point(site, color=color, size=4)
        
        self.current_step += 1
        self.step_info.config(text=f"Step: {self.current_step} / {len(self.algorithm.steps)}")
        self.status_bar.config(text=f"Step {self.current_step}: Merging... Red=Hyperplane, Gray=Hull")
    
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