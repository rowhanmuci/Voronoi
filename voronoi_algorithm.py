"""
Voronoi Diagram - Divide and Conquer 演算法
完整實作包含 merge 演算法
"""
from typing import List, Tuple, Optional
from voronoi_geometry import Point, Edge, VoronoiDiagram, perpendicular_bisector, line_intersection, convex_hull, ccw, circumcenter, merge_convex_hulls, _find_upper_tangent_idx, _find_lower_tangent_idx
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
        
        # 按 x 座標排序，x 相同則按 y 排序
        sorted_points = sorted(points, key=lambda p: (p.x, p.y))
        
        # 移除重複點
        unique_points = [sorted_points[0]]
        for p in sorted_points[1:]:
            if abs(p.x - unique_points[-1].x) > 1e-9 or abs(p.y - unique_points[-1].y) > 1e-9:
                unique_points.append(p)
        
        # 清空步驟記錄
        self.steps = []
        
        # 執行分治法
        vd, _ = self._divide_conquer(unique_points)
        
        return vd
    
    def _divide_conquer(self, points: List[Point]) -> Tuple[VoronoiDiagram, List[Point]]:
        """
        遞迴的分治法
        返回: (VoronoiDiagram, convex_hull)
        """
        n = len(points)
        
        # Base cases
        if n == 1:
            vd = self._voronoi_one_point(points[0])
            return vd, points.copy()
        elif n == 2:
            vd = self._voronoi_two_points(points[0], points[1])
            return vd, points.copy()
        elif n == 3:
            vd = self._voronoi_three_points(points[0], points[1], points[2])
            hull = convex_hull(points)
            return vd, hull
        
        # Divide
        mid = n // 2
        left_points = points[:mid]
        right_points = points[mid:]
        
        # Conquer
        left_vd, left_hull = self._divide_conquer(left_points)
        right_vd, right_hull = self._divide_conquer(right_points)
        
        # 合併凸包 (O(n) 時間複雜度)
        merged_hull = merge_convex_hulls(left_hull, right_hull)
        
        # 記錄步驟（合併前的狀態）
        self.steps.append({
            'left': self._copy_vd(left_vd),
            'right': self._copy_vd(right_vd),
            'left_hull': left_hull.copy(),
            'right_hull': right_hull.copy(),
            'merged_hull': merged_hull.copy(),
            'merged': None
        })
        
        # Merge
        merged_vd = self._merge(left_vd, right_vd, left_hull, right_hull)
        
        # 更新最後一步的合併結果
        if self.steps:
            self.steps[-1]['merged'] = self._copy_vd(merged_vd)
        
        return merged_vd, merged_hull
    
    def _copy_vd(self, vd: VoronoiDiagram) -> VoronoiDiagram:
        """深拷貝 VoronoiDiagram"""
        new_vd = VoronoiDiagram()
        new_vd.sites = vd.sites.copy()
        new_vd.vertices = vd.vertices.copy()
        new_vd.edges = [Edge(e.start, e.end) for e in vd.edges]
        return new_vd
    
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
            edge.site_left = p1
            edge.site_right = p2
            vd.add_edge(edge)
        
        return vd
    
    def _voronoi_three_points(self, p1: Point, p2: Point, p3: Point) -> VoronoiDiagram:
        """三個點的 Voronoi diagram"""
        vd = VoronoiDiagram()
        vd.add_site(p1)
        vd.add_site(p2)
        vd.add_site(p3)
        
        # 按 x 座標排序
        points = sorted([p1, p2, p3], key=lambda p: (p.x, p.y))
        p1, p2, p3 = points[0], points[1], points[2]
        
        # 檢查是否三點共線
        cross = ccw(p1, p2, p3)
        if abs(cross) < 1e-6:
            # 三點共線，兩條平行的中垂線
            a1, b1, c1 = perpendicular_bisector(p1, p2)
            intersections1 = self._line_canvas_intersections(a1, b1, c1)
            if len(intersections1) >= 2:
                edge1 = Edge(intersections1[0], intersections1[1])
                edge1.site_left = p1
                edge1.site_right = p2
                vd.add_edge(edge1)
            
            a2, b2, c2 = perpendicular_bisector(p2, p3)
            intersections2 = self._line_canvas_intersections(a2, b2, c2)
            if len(intersections2) >= 2:
                edge2 = Edge(intersections2[0], intersections2[1])
                edge2.site_left = p2
                edge2.site_right = p3
                vd.add_edge(edge2)
            
            return vd
        
        # 計算外心（Voronoi vertex）
        center = circumcenter(p1, p2, p3)
        
        if center is None:
            return vd
        
        # 三條中垂線
        line12 = perpendicular_bisector(p1, p2)
        line13 = perpendicular_bisector(p1, p3)
        line23 = perpendicular_bisector(p2, p3)
        
        if self._point_in_canvas_extended(center):
            vd.add_vertex(center)
        
        # 建立三條射線（從外心出發）
        # 每條射線沿著對應的中垂線，方向遠離第三個點
        
        # 邊 1: p1-p2 的中垂線，遠離 p3
        end1 = self._extend_ray_to_boundary(center, *line12, p3)
        if end1:
            edge1 = Edge(center, end1)
            edge1.site_left = p1
            edge1.site_right = p2
            vd.add_edge(edge1)
        
        # 邊 2: p1-p3 的中垂線，遠離 p2
        end2 = self._extend_ray_to_boundary(center, *line13, p2)
        if end2:
            edge2 = Edge(center, end2)
            edge2.site_left = p1
            edge2.site_right = p3
            vd.add_edge(edge2)
        
        # 邊 3: p2-p3 的中垂線，遠離 p1
        end3 = self._extend_ray_to_boundary(center, *line23, p1)
        if end3:
            edge3 = Edge(center, end3)
            edge3.site_left = p2
            edge3.site_right = p3
            vd.add_edge(edge3)
        
        return vd
    
    def _extend_ray_to_boundary(self, start: Point, a: float, b: float, c: float, 
                                 away_from: Point) -> Optional[Point]:
        """
        從 start 沿著直線 ax + by + c = 0 延伸到邊界
        方向：遠離 away_from
        """
        intersections = self._line_canvas_intersections(a, b, c)
        
        if not intersections:
            return None
        
        best_pt = None
        max_dist = -1
        
        for pt in intersections:
            # 計算方向向量
            dx = pt.x - start.x
            dy = pt.y - start.y
            
            # 計算到 away_from 的向量
            ax = away_from.x - start.x
            ay = away_from.y - start.y
            
            # 點積為負表示方向相反（我們要的方向）
            dot = dx * ax + dy * ay
            
            if dot < 0:  # 方向遠離 away_from
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > max_dist:
                    max_dist = dist
                    best_pt = pt
        
        # 如果找不到，可能 start 在邊界外，選離 away_from 最遠的
        if best_pt is None and intersections:
            best_pt = max(intersections, key=lambda p: p.distance_to(away_from))
        
        return best_pt
    
    def _merge(self, left_vd: VoronoiDiagram, right_vd: VoronoiDiagram,
               left_hull: List[Point] = None, right_hull: List[Point] = None) -> VoronoiDiagram:
        """
        合併左右兩個 Voronoi diagram
        使用 dividing chain 演算法
        """
        merged = VoronoiDiagram()
        merged.sites = left_vd.sites + right_vd.sites
        
        # 計算凸包（如果沒有提供）
        if left_hull is None:
            left_hull = convex_hull(left_vd.sites)
        if right_hull is None:
            right_hull = convex_hull(right_vd.sites)
        
        # 找上下切線
        upper_left, upper_right = self._find_upper_tangent(left_hull, right_hull)
        lower_left, lower_right = self._find_lower_tangent(left_hull, right_hull)
        
        # 建立邊與 site 的關聯
        left_edge_sites = self._build_edge_site_map(left_vd)
        right_edge_sites = self._build_edge_site_map(right_vd)
        
        # 追蹤 dividing chain
        chain_edges, left_edges, right_edges = self._trace_chain(
            left_vd, right_vd,
            left_edge_sites, right_edge_sites,
            upper_left, upper_right,
            lower_left, lower_right
        )
        
        # 合併結果
        merged.edges = left_edges + right_edges + chain_edges
        
        # 收集頂點
        for edge in merged.edges:
            if edge.start and self._point_in_canvas(edge.start):
                if edge.start not in merged.vertices:
                    merged.vertices.append(edge.start)
            if edge.end and self._point_in_canvas(edge.end):
                if edge.end not in merged.vertices:
                    merged.vertices.append(edge.end)
        
        return merged
    
    def _build_edge_site_map(self, vd: VoronoiDiagram) -> dict:
        """
        建立每條邊對應的兩個 site
        回傳 {edge_index: (site1, site2)}
        """
        edge_map = {}
        
        for i, edge in enumerate(vd.edges):
            if edge.site_left and edge.site_right:
                edge_map[i] = (edge.site_left, edge.site_right)
            else:
                # 嘗試推斷
                sites = self._infer_edge_sites(edge, vd.sites)
                if sites:
                    edge_map[i] = sites
        
        return edge_map
    
    def _infer_edge_sites(self, edge: Edge, sites: List[Point]) -> Optional[Tuple[Point, Point]]:
        """
        根據邊的位置推斷它屬於哪兩個 site
        """
        if not edge.start or not edge.end:
            return None
        
        # 邊的中點
        mid_x = (edge.start.x + edge.end.x) / 2
        mid_y = (edge.start.y + edge.end.y) / 2
        mid = Point(mid_x, mid_y)
        
        # 找最近的兩個 site
        distances = [(s.distance_to(mid), s) for s in sites]
        distances.sort(key=lambda x: x[0])
        
        if len(distances) >= 2:
            return (distances[0][1], distances[1][1])
        return None
    
    def _find_upper_tangent(self, left_hull: List[Point], right_hull: List[Point]) -> Tuple[Point, Point]:
        """找上切線"""
        if len(left_hull) == 0 or len(right_hull) == 0:
            return left_hull[0] if left_hull else Point(0, 0), right_hull[0] if right_hull else Point(0, 0)
        
        # 左凸包最右邊的點
        left_idx = max(range(len(left_hull)), key=lambda i: (left_hull[i].x, -left_hull[i].y))
        # 右凸包最左邊的點
        right_idx = min(range(len(right_hull)), key=lambda i: (right_hull[i].x, right_hull[i].y))
        
        n_left = len(left_hull)
        n_right = len(right_hull)
        
        done = False
        while not done:
            done = True
            
            # 左凸包逆時針
            while n_left > 1:
                next_idx = (left_idx - 1 + n_left) % n_left
                if ccw(right_hull[right_idx], left_hull[left_idx], left_hull[next_idx]) > 1e-9:
                    left_idx = next_idx
                    done = False
                else:
                    break
            
            # 右凸包順時針
            while n_right > 1:
                next_idx = (right_idx + 1) % n_right
                if ccw(left_hull[left_idx], right_hull[right_idx], right_hull[next_idx]) < -1e-9:
                    right_idx = next_idx
                    done = False
                else:
                    break
        
        return left_hull[left_idx], right_hull[right_idx]
    
    def _find_lower_tangent(self, left_hull: List[Point], right_hull: List[Point]) -> Tuple[Point, Point]:
        """找下切線"""
        if len(left_hull) == 0 or len(right_hull) == 0:
            return left_hull[0] if left_hull else Point(0, 0), right_hull[0] if right_hull else Point(0, 0)
        
        left_idx = max(range(len(left_hull)), key=lambda i: (left_hull[i].x, left_hull[i].y))
        right_idx = min(range(len(right_hull)), key=lambda i: (right_hull[i].x, -right_hull[i].y))
        
        n_left = len(left_hull)
        n_right = len(right_hull)
        
        done = False
        while not done:
            done = True
            
            # 左凸包順時針
            while n_left > 1:
                next_idx = (left_idx + 1) % n_left
                if ccw(right_hull[right_idx], left_hull[left_idx], left_hull[next_idx]) < -1e-9:
                    left_idx = next_idx
                    done = False
                else:
                    break
            
            # 右凸包逆時針
            while n_right > 1:
                next_idx = (right_idx - 1 + n_right) % n_right
                if ccw(left_hull[left_idx], right_hull[right_idx], right_hull[next_idx]) > 1e-9:
                    right_idx = next_idx
                    done = False
                else:
                    break
        
        return left_hull[left_idx], right_hull[right_idx]
    
    def _trace_chain(self, left_vd: VoronoiDiagram, right_vd: VoronoiDiagram,
                      left_edge_sites: dict, right_edge_sites: dict,
                      upper_left: Point, upper_right: Point,
                      lower_left: Point, lower_right: Point) -> Tuple[List[Edge], List[Edge], List[Edge]]:
        """
        追蹤 dividing chain 從上到下
        """
        chain_edges = []
        
        # 複製邊（以便修改）
        left_edges = [Edge(e.start, e.end) for e in left_vd.edges]
        right_edges = [Edge(e.start, e.end) for e in right_vd.edges]
        
        # 複製 site 資訊
        for i, e in enumerate(left_edges):
            if i in left_edge_sites:
                e.site_left, e.site_right = left_edge_sites[i]
        for i, e in enumerate(right_edges):
            if i in right_edge_sites:
                e.site_left, e.site_right = right_edge_sites[i]
        
        # 當前左右 site
        cur_left = upper_left
        cur_right = upper_right
        
        # 計算起始中垂線
        a, b, c = perpendicular_bisector(cur_left, cur_right)
        
        # 找起始點（上邊界）
        cur_point = self._find_boundary_intersection(a, b, c, 'top')
        
        if cur_point is None:
            intersections = self._line_canvas_intersections(a, b, c)
            if intersections:
                cur_point = min(intersections, key=lambda p: p.y)
        
        max_iter = 200
        iteration = 0
        
        while iteration < max_iter:
            iteration += 1
            
            # 當前中垂線
            a, b, c = perpendicular_bisector(cur_left, cur_right)
            
            # 找下一個交點
            next_point, hit_edge, hit_side, new_site = self._find_next_hit(
                cur_point, a, b, c,
                cur_left, cur_right,
                left_edges, right_edges,
                left_vd.sites, right_vd.sites
            )
            
            if next_point is None:
                # 到達下邊界
                boundary_pt = self._find_boundary_intersection(a, b, c, 'bottom')
                if boundary_pt is None:
                    intersections = self._line_canvas_intersections(a, b, c)
                    if intersections:
                        boundary_pt = max(intersections, key=lambda p: p.y)
                
                if boundary_pt and cur_point:
                    edge = Edge(cur_point, boundary_pt)
                    edge.site_left = cur_left
                    edge.site_right = cur_right
                    chain_edges.append(edge)
                break
            
            # 創建 chain 邊
            if cur_point and next_point:
                edge = Edge(cur_point, next_point)
                edge.site_left = cur_left
                edge.site_right = cur_right
                chain_edges.append(edge)
            
            # 裁剪碰到的邊
            if hit_edge is not None:
                self._trim_edge_at_point(hit_edge, next_point, cur_left if hit_side == 'left' else cur_right)
            
            # 更新 site
            if new_site:
                if hit_side == 'left':
                    cur_left = new_site
                else:
                    cur_right = new_site
            
            # 檢查終止條件
            if cur_left == lower_left and cur_right == lower_right:
                # 延伸到下邊界
                a, b, c = perpendicular_bisector(cur_left, cur_right)
                boundary_pt = self._find_boundary_intersection(a, b, c, 'bottom')
                if boundary_pt is None:
                    intersections = self._line_canvas_intersections(a, b, c)
                    if intersections:
                        boundary_pt = max(intersections, key=lambda p: p.y)
                
                if boundary_pt and next_point:
                    edge = Edge(next_point, boundary_pt)
                    edge.site_left = cur_left
                    edge.site_right = cur_right
                    chain_edges.append(edge)
                break
            
            cur_point = next_point
        
        # 移除被完全裁剪的邊
        left_edges = [e for e in left_edges if e.start and e.end and e.start.distance_to(e.end) > 1e-6]
        right_edges = [e for e in right_edges if e.start and e.end and e.start.distance_to(e.end) > 1e-6]
        
        return chain_edges, left_edges, right_edges
    
    def _find_boundary_intersection(self, a: float, b: float, c: float, side: str) -> Optional[Point]:
        """找直線與特定邊界的交點"""
        epsilon = 1e-9
        
        if side == 'top':
            # y = 0
            if abs(a) > epsilon:
                x = -c / a
                if 0 <= x <= self.canvas_width:
                    return Point(x, 0)
        elif side == 'bottom':
            # y = canvas_height
            if abs(a) > epsilon:
                x = -(b * self.canvas_height + c) / a
                if 0 <= x <= self.canvas_width:
                    return Point(x, self.canvas_height)
        elif side == 'left':
            # x = 0
            if abs(b) > epsilon:
                y = -c / b
                if 0 <= y <= self.canvas_height:
                    return Point(0, y)
        elif side == 'right':
            # x = canvas_width
            if abs(b) > epsilon:
                y = -(a * self.canvas_width + c) / b
                if 0 <= y <= self.canvas_height:
                    return Point(self.canvas_width, y)
        
        return None
    
    def _find_next_hit(self, cur_point: Point, a: float, b: float, c: float,
                        cur_left: Point, cur_right: Point,
                        left_edges: List[Edge], right_edges: List[Edge],
                        left_sites: List[Point], right_sites: List[Point]) -> Tuple[Optional[Point], Optional[Edge], Optional[str], Optional[Point]]:
        """
        沿中垂線往下找最近的交點
        """
        if cur_point is None:
            return None, None, None, None
        
        best_point = None
        best_edge = None
        best_side = None
        best_site = None
        best_y = float('inf')
        
        # 檢查左側邊
        for edge in left_edges:
            if not edge.start or not edge.end:
                continue
            
            intersection = self._line_segment_intersection(a, b, c, edge)
            if intersection and intersection.y > cur_point.y + 1e-6:
                if intersection.y < best_y:
                    # 找對應的另一個 site
                    other = self._get_other_site(edge, cur_left, left_sites)
                    if other:
                        best_y = intersection.y
                        best_point = intersection
                        best_edge = edge
                        best_side = 'left'
                        best_site = other
        
        # 檢查右側邊
        for edge in right_edges:
            if not edge.start or not edge.end:
                continue
            
            intersection = self._line_segment_intersection(a, b, c, edge)
            if intersection and intersection.y > cur_point.y + 1e-6:
                if intersection.y < best_y:
                    other = self._get_other_site(edge, cur_right, right_sites)
                    if other:
                        best_y = intersection.y
                        best_point = intersection
                        best_edge = edge
                        best_side = 'right'
                        best_site = other
        
        # 檢查是否先到達邊界
        boundary_pt = self._find_boundary_intersection(a, b, c, 'bottom')
        if boundary_pt is None:
            # 嘗試左右邊界
            boundary_pt = self._find_boundary_intersection(a, b, c, 'left')
            if boundary_pt is None or boundary_pt.y <= cur_point.y:
                boundary_pt = self._find_boundary_intersection(a, b, c, 'right')
        
        if boundary_pt and boundary_pt.y > cur_point.y + 1e-6:
            if best_point is None or boundary_pt.y < best_y:
                return None, None, None, None  # 到達邊界
        
        return best_point, best_edge, best_side, best_site
    
    def _line_segment_intersection(self, a: float, b: float, c: float, edge: Edge) -> Optional[Point]:
        """計算直線 ax+by+c=0 與線段的交點"""
        if not edge.start or not edge.end:
            return None
        
        x1, y1 = edge.start.x, edge.start.y
        x2, y2 = edge.end.x, edge.end.y
        
        d1 = a * x1 + b * y1 + c
        d2 = a * x2 + b * y2 + c
        
        # 同側不相交
        if d1 * d2 > 1e-9:
            return None
        
        # 平行
        if abs(d1 - d2) < 1e-9:
            return None
        
        t = d1 / (d1 - d2)
        
        if t < -1e-9 or t > 1 + 1e-9:
            return None
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return Point(x, y)
    
    def _get_other_site(self, edge: Edge, current_site: Point, sites: List[Point]) -> Optional[Point]:
        """取得邊的另一個 site"""
        # 先檢查已儲存的 site 資訊
        if edge.site_left and edge.site_right:
            if edge.site_left == current_site:
                return edge.site_right
            elif edge.site_right == current_site:
                return edge.site_left
        
        # 推斷
        if not edge.start or not edge.end:
            return None
        
        mid_x = (edge.start.x + edge.end.x) / 2
        mid_y = (edge.start.y + edge.end.y) / 2
        mid = Point(mid_x, mid_y)
        
        # 找最近的兩個 site
        distances = [(s.distance_to(mid), s) for s in sites]
        distances.sort(key=lambda x: x[0])
        
        for dist, site in distances[:3]:
            if site != current_site:
                # 驗證：檢查這兩個 site 的中垂線是否接近這條邊
                a, b, c = perpendicular_bisector(current_site, site)
                d1 = abs(a * edge.start.x + b * edge.start.y + c)
                d2 = abs(a * edge.end.x + b * edge.end.y + c)
                
                if d1 < 20 or d2 < 20:  # 允許誤差
                    return site
        
        return None
    
    def _trim_edge_at_point(self, edge: Edge, intersection: Point, reference_site: Point):
        """在交點處裁剪邊，保留靠近 reference_site 的部分"""
        if not edge.start or not edge.end:
            return
        
        dist_start = edge.start.distance_to(reference_site)
        dist_end = edge.end.distance_to(reference_site)
        
        if dist_start < dist_end:
            edge.end = intersection
        else:
            edge.start = intersection
    
    def _line_canvas_intersections(self, a: float, b: float, c: float) -> List[Point]:
        """計算直線與畫布邊界的交點"""
        intersections = []
        epsilon = 1e-9
        
        # 左邊界 x=0
        if abs(b) > epsilon:
            y = -c / b
            if -epsilon <= y <= self.canvas_height + epsilon:
                intersections.append(Point(0, max(0, min(self.canvas_height, y))))
        
        # 右邊界 x=canvas_width
        if abs(b) > epsilon:
            y = -(a * self.canvas_width + c) / b
            if -epsilon <= y <= self.canvas_height + epsilon:
                pt = Point(self.canvas_width, max(0, min(self.canvas_height, y)))
                if not any(abs(pt.x - p.x) < epsilon and abs(pt.y - p.y) < epsilon for p in intersections):
                    intersections.append(pt)
        
        # 上邊界 y=0
        if abs(a) > epsilon:
            x = -c / a
            if -epsilon <= x <= self.canvas_width + epsilon:
                pt = Point(max(0, min(self.canvas_width, x)), 0)
                if not any(abs(pt.x - p.x) < epsilon and abs(pt.y - p.y) < epsilon for p in intersections):
                    intersections.append(pt)
        
        # 下邊界 y=canvas_height
        if abs(a) > epsilon:
            x = -(b * self.canvas_height + c) / a
            if -epsilon <= x <= self.canvas_width + epsilon:
                pt = Point(max(0, min(self.canvas_width, x)), self.canvas_height)
                if not any(abs(pt.x - p.x) < epsilon and abs(pt.y - p.y) < epsilon for p in intersections):
                    intersections.append(pt)
        
        return intersections[:2]
    
    def _point_in_canvas(self, p: Point) -> bool:
        """判斷點是否在畫布內"""
        return -1e-6 <= p.x <= self.canvas_width + 1e-6 and -1e-6 <= p.y <= self.canvas_height + 1e-6
    
    def _point_in_canvas_extended(self, p: Point) -> bool:
        """判斷點是否在畫布的擴展範圍內"""
        margin = 50
        return -margin <= p.x <= self.canvas_width + margin and -margin <= p.y <= self.canvas_height + margin


print("Divide-and-Conquer 演算法載入完成！")