"""
Voronoi Diagram - Divide and Conquer 演算法
"""
from typing import List, Tuple, Optional
from voronoi_geometry import Point, Edge, VoronoiDiagram, perpendicular_bisector, line_intersection, convex_hull
import math


class VoronoiDC:
    """使用 Divide and Conquer 方法建構 Voronoi Diagram"""
    
    def __init__(self, canvas_width: int = 600, canvas_height: int = 600):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.steps = []  # 儲存每一步的狀態 (用於 step-by-step 模式)
    
    def build(self, points: List[Point]) -> VoronoiDiagram:
        """
        建構 Voronoi diagram
        """
        if len(points) < 2:
            vd = VoronoiDiagram()
            vd.sites = points.copy()
            return vd
        
        # 按 x 座標排序
        sorted_points = sorted(points, key=lambda p: (p.x, p.y))
        
        # 清空步驟記錄
        self.steps = []
        
        # 執行分治法
        vd = self._divide_conquer(sorted_points)
        
        return vd
    
    def _divide_conquer(self, points: List[Point]) -> VoronoiDiagram:
        """
        遞迴的分治法
        """
        n = len(points)
        
        # Base cases
        if n == 1:
            return self._voronoi_one_point(points[0])
        elif n == 2:
            return self._voronoi_two_points(points[0], points[1])
        elif n == 3:
            return self._voronoi_three_points(points[0], points[1], points[2])
        
        # Divide
        mid = n // 2
        left_points = points[:mid]
        right_points = points[mid:]
        
        # Conquer
        left_vd = self._divide_conquer(left_points)
        right_vd = self._divide_conquer(right_points)
        
        # 記錄步驟（合併前的狀態）
        self.steps.append({
            'left': left_vd,
            'right': right_vd,
            'merged': None
        })
        
        # Merge
        merged_vd = self._merge(left_vd, right_vd)
        
        # 更新最後一步的合併結果
        if self.steps:
            self.steps[-1]['merged'] = merged_vd
        
        return merged_vd
    
    def _voronoi_one_point(self, p: Point) -> VoronoiDiagram:
        """一個點的 Voronoi diagram（整個平面）"""
        vd = VoronoiDiagram()
        vd.add_site(p)
        return vd
    
    def _voronoi_two_points(self, p1: Point, p2: Point) -> VoronoiDiagram:
        """兩個點的 Voronoi diagram（一條中垂線）"""
        vd = VoronoiDiagram()
        vd.add_site(p1)
        vd.add_site(p2)
        
        # 處理重複點的情況
        if abs(p1.x - p2.x) < 1e-9 and abs(p1.y - p2.y) < 1e-9:
            return vd  # 重複點，沒有邊
        
        # 計算中垂線
        a, b, c = perpendicular_bisector(p1, p2)
        
        # 計算中垂線與畫布邊界的交點
        intersections = self._line_canvas_intersections(a, b, c)
        
        if len(intersections) >= 2:
            edge = Edge(intersections[0], intersections[1])
            vd.add_edge(edge)
        
        return vd
    
    def _voronoi_three_points(self, p1: Point, p2: Point, p3: Point) -> VoronoiDiagram:
        """三個點的 Voronoi diagram"""
        vd = VoronoiDiagram()
        vd.add_site(p1)
        vd.add_site(p2)
        vd.add_site(p3)
        
        # 檢查是否三點共線
        area = abs((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))
        if area < 1e-6:
            # 三點共線，退化為兩條平行的中垂線
            # 找出中間的點
            points = sorted([p1, p2, p3], key=lambda p: (p.x, p.y))
            
            # 中間點與左右兩點的中垂線
            a1, b1, c1 = perpendicular_bisector(points[0], points[1])
            intersections1 = self._line_canvas_intersections(a1, b1, c1)
            if len(intersections1) >= 2:
                edge1 = Edge(intersections1[0], intersections1[1])
                vd.add_edge(edge1)
            
            a2, b2, c2 = perpendicular_bisector(points[1], points[2])
            intersections2 = self._line_canvas_intersections(a2, b2, c2)
            if len(intersections2) >= 2:
                edge2 = Edge(intersections2[0], intersections2[1])
                vd.add_edge(edge2)
            
            return vd
        
        # 計算三條中垂線
        line12 = perpendicular_bisector(p1, p2)
        line13 = perpendicular_bisector(p1, p3)
        line23 = perpendicular_bisector(p2, p3)
        
        # 計算 Voronoi vertex（外心，三條中垂線的交點）
        center = line_intersection(*line12, *line13)
        
        if center and self._point_in_canvas_extended(center):
            vd.add_vertex(center)
            
            # 從 center 向三個方向延伸的邊
            # 每條邊沿著對應的中垂線，延伸到畫布邊界
            
            # 邊 1: 沿著 p1-p2 的中垂線
            edge1_pts = self._extend_from_center_to_boundary(center, *line12, p3)
            if edge1_pts:
                edge1 = Edge(center, edge1_pts)
                vd.add_edge(edge1)
            
            # 邊 2: 沿著 p1-p3 的中垂線
            edge2_pts = self._extend_from_center_to_boundary(center, *line13, p2)
            if edge2_pts:
                edge2 = Edge(center, edge2_pts)
                vd.add_edge(edge2)
            
            # 邊 3: 沿著 p2-p3 的中垂線
            edge3_pts = self._extend_from_center_to_boundary(center, *line23, p1)
            if edge3_pts:
                edge3 = Edge(center, edge3_pts)
                vd.add_edge(edge3)
        else:
            # 外心在畫布外，需要特殊處理
            # 這種情況下，三條中垂線可能都穿過畫布
            for line in [line12, line13, line23]:
                intersections = self._line_canvas_intersections(*line)
                if len(intersections) >= 2:
                    edge = Edge(intersections[0], intersections[1])
                    vd.add_edge(edge)
        
        return vd
    
    def _extend_from_center_to_boundary(self, center: Point, a: float, b: float, c: float, 
                                       avoid_point: Point) -> Optional[Point]:
        """
        從 center 沿著直線 ax + by + c = 0 延伸到畫布邊界
        延伸方向：遠離 avoid_point
        """
        # 找到直線與邊界的所有交點
        intersections = self._line_canvas_intersections(a, b, c)
        
        if not intersections:
            return None
        
        # 選擇離 avoid_point 較遠的那個交點
        best_pt = None
        max_dist = -1
        
        for pt in intersections:
            # 確保這個點在從 center 出發的正確方向上
            # 計算 center 到 pt 的向量
            dx = pt.x - center.x
            dy = pt.y - center.y
            
            # 計算 center 到 avoid_point 的向量
            ax_vec = avoid_point.x - center.x
            ay_vec = avoid_point.y - center.y
            
            # 如果兩個向量的點積為負，說明方向相反（這是我們要的）
            dot_product = dx * ax_vec + dy * ay_vec
            
            if dot_product < 0:  # 方向遠離 avoid_point
                dist = pt.distance_to(avoid_point)
                if dist > max_dist:
                    max_dist = dist
                    best_pt = pt
        
        return best_pt
    
    def _point_in_canvas_extended(self, p: Point) -> bool:
        """判斷點是否在畫布的擴展範圍內（允許一定的誤差範圍）"""
        margin = 10
        return -margin <= p.x <= self.canvas_width + margin and -margin <= p.y <= self.canvas_height + margin
    
    def _merge(self, left_vd: VoronoiDiagram, right_vd: VoronoiDiagram) -> VoronoiDiagram:
        """
        合併左右兩個 Voronoi diagram
        這是演算法的核心部分
        """
        merged = VoronoiDiagram()
        
        # 合併所有 sites
        merged.sites = left_vd.sites + right_vd.sites
        
        # 暫時保留所有邊（後續會刪除被切斷的）
        merged.edges = left_vd.edges.copy() + right_vd.edges.copy()
        merged.vertices = left_vd.vertices.copy() + right_vd.vertices.copy()
        
        # 如果只有兩個點，直接返回
        if len(merged.sites) == 2:
            return self._voronoi_two_points(merged.sites[0], merged.sites[1])
        
        # 計算左右兩側的凸包
        left_hull = convex_hull(left_vd.sites)
        right_hull = convex_hull(right_vd.sites)
        
        # 找到上下切線（這是簡化版本，實際需要更複雜的演算法）
        # 這裡暫時用最右邊的左側點和最左邊的右側點
        left_rightmost = max(left_vd.sites, key=lambda p: (p.x, -p.y))
        right_leftmost = min(right_vd.sites, key=lambda p: (p.x, p.y))
        
        # 構造 hyperplane（這是簡化版本）
        # 計算中垂線並與畫布交界
        a, b, c = perpendicular_bisector(left_rightmost, right_leftmost)
        intersections = self._line_canvas_intersections(a, b, c)
        
        if len(intersections) >= 2:
            hyperplane_edge = Edge(intersections[0], intersections[1])
            merged.add_edge(hyperplane_edge)
            
            # TODO: 實作完整的 hyperplane Voronoi diagram 構造
            # 這需要：
            # 1. 找到正確的上下切線
            # 2. 從下切線開始向上構造 merging edges
            # 3. 刪除被切斷的舊邊
        
        return merged
    
    def _line_canvas_intersections(self, a: float, b: float, c: float) -> List[Point]:
        """
        計算直線與畫布邊界的交點
        直線方程: ax + by + c = 0
        """
        intersections = []
        epsilon = 1e-9
        
        # 與左邊界 x=0 的交點
        if abs(b) > epsilon:
            y = -c / b
            if -epsilon <= y <= self.canvas_height + epsilon:
                intersections.append(Point(0, max(0, min(self.canvas_height, y))))
        
        # 與右邊界 x=canvas_width 的交點
        if abs(b) > epsilon:
            y = -(a * self.canvas_width + c) / b
            if -epsilon <= y <= self.canvas_height + epsilon:
                pt = Point(self.canvas_width, max(0, min(self.canvas_height, y)))
                # 檢查是否重複
                if not any(abs(pt.x - p.x) < epsilon and abs(pt.y - p.y) < epsilon for p in intersections):
                    intersections.append(pt)
        
        # 與上邊界 y=0 的交點
        if abs(a) > epsilon:
            x = -c / a
            if -epsilon <= x <= self.canvas_width + epsilon:
                pt = Point(max(0, min(self.canvas_width, x)), 0)
                if not any(abs(pt.x - p.x) < epsilon and abs(pt.y - p.y) < epsilon for p in intersections):
                    intersections.append(pt)
        
        # 與下邊界 y=canvas_height 的交點
        if abs(a) > epsilon:
            x = -(b * self.canvas_height + c) / a
            if -epsilon <= x <= self.canvas_width + epsilon:
                pt = Point(max(0, min(self.canvas_width, x)), self.canvas_height)
                if not any(abs(pt.x - p.x) < epsilon and abs(pt.y - p.y) < epsilon for p in intersections):
                    intersections.append(pt)
        
        # 如果只有一個交點，可能是直線經過角落，需要檢查角落
        if len(intersections) < 2:
            corners = [
                Point(0, 0),
                Point(self.canvas_width, 0),
                Point(0, self.canvas_height),
                Point(self.canvas_width, self.canvas_height)
            ]
            for corner in corners:
                # 檢查角落是否在直線上
                if abs(a * corner.x + b * corner.y + c) < epsilon:
                    if not any(abs(corner.x - p.x) < epsilon and abs(corner.y - p.y) < epsilon for p in intersections):
                        intersections.append(corner)
        
        return intersections[:2]  # 最多返回兩個交點
    
    def _point_in_canvas(self, p: Point) -> bool:
        """判斷點是否在畫布內"""
        return 0 <= p.x <= self.canvas_width and 0 <= p.y <= self.canvas_height


print("Divide-and-Conquer 演算法載入完成！")