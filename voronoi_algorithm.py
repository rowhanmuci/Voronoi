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
        
        # 計算三條中垂線
        line12 = perpendicular_bisector(p1, p2)
        line13 = perpendicular_bisector(p1, p3)
        line23 = perpendicular_bisector(p2, p3)
        
        # 計算 Voronoi vertex（三條中垂線的交點）
        center = line_intersection(*line12, *line13)
        
        if center:
            vd.add_vertex(center)
            
            # 從 center 出發的三條射線
            # 射線方向：沿著中垂線，遠離第三個點
            
            # 邊 1: center 到 p1-p2 中垂線的邊界
            intersections = self._line_canvas_intersections(*line12)
            for pt in intersections:
                if self._point_in_canvas(pt):
                    # 判斷方向
                    dist_to_p3 = pt.distance_to(p3)
                    if dist_to_p3 > center.distance_to(p3):
                        edge = Edge(center, pt)
                        vd.add_edge(edge)
                        break
            
            # 類似處理另外兩條邊
            intersections = self._line_canvas_intersections(*line13)
            for pt in intersections:
                if self._point_in_canvas(pt):
                    dist_to_p2 = pt.distance_to(p2)
                    if dist_to_p2 > center.distance_to(p2):
                        edge = Edge(center, pt)
                        vd.add_edge(edge)
                        break
            
            intersections = self._line_canvas_intersections(*line23)
            for pt in intersections:
                if self._point_in_canvas(pt):
                    dist_to_p1 = pt.distance_to(p1)
                    if dist_to_p1 > center.distance_to(p1):
                        edge = Edge(center, pt)
                        vd.add_edge(edge)
                        break
        
        return vd
    
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
        
        # 與左邊界 x=0 的交點
        if abs(a) > 1e-9:
            y = -c / b if abs(b) > 1e-9 else None
            if y is not None and 0 <= y <= self.canvas_height:
                intersections.append(Point(0, y))
        
        # 與右邊界 x=canvas_width 的交點
        if abs(a) > 1e-9:
            y = -(a * self.canvas_width + c) / b if abs(b) > 1e-9 else None
            if y is not None and 0 <= y <= self.canvas_height:
                intersections.append(Point(self.canvas_width, y))
        
        # 與上邊界 y=0 的交點
        if abs(b) > 1e-9:
            x = -c / a if abs(a) > 1e-9 else None
            if x is not None and 0 <= x <= self.canvas_width:
                pt = Point(x, 0)
                if not any(abs(pt.x - p.x) < 1e-6 and abs(pt.y - p.y) < 1e-6 for p in intersections):
                    intersections.append(pt)
        
        # 與下邊界 y=canvas_height 的交點
        if abs(b) > 1e-9:
            x = -(b * self.canvas_height + c) / a if abs(a) > 1e-9 else None
            if x is not None and 0 <= x <= self.canvas_width:
                pt = Point(x, self.canvas_height)
                if not any(abs(pt.x - p.x) < 1e-6 and abs(pt.y - p.y) < 1e-6 for p in intersections):
                    intersections.append(pt)
        
        return intersections[:2]  # 最多返回兩個交點
    
    def _point_in_canvas(self, p: Point) -> bool:
        """判斷點是否在畫布內"""
        return 0 <= p.x <= self.canvas_width and 0 <= p.y <= self.canvas_height


print("Divide-and-Conquer 演算法載入完成！")
