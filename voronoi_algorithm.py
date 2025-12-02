"""
Voronoi Diagram - Divide and Conquer 演算法
完整實作包含 merge 演算法

【修正記錄】
v7: 加入詳細的 step-by-step 記錄
    記錄每個合併階段的詳細資訊，包括：
    - 左右 VD
    - 左右 Convex Hull
    - 合併後的 Convex Hull（重要！）
    - 上下切線
    - Hyper Plane
    - 消線後的 VD
    - 最終合併結果
"""
from typing import List, Tuple, Optional
from voronoi_geometry import Point, Edge, VoronoiDiagram, perpendicular_bisector, line_intersection, convex_hull, ccw, circumcenter, merge_convex_hulls
import math


class VoronoiDC:
    """使用 Divide and Conquer 方法建構 Voronoi Diagram"""
    
    def __init__(self, canvas_width: int = 600, canvas_height: int = 600):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.steps = []  # 詳細步驟記錄
    
    def build(self, points: List[Point]) -> VoronoiDiagram:
        """建構 Voronoi diagram"""
        if len(points) < 2:
            vd = VoronoiDiagram()
            vd.sites = points.copy()
            return vd
        
        sorted_points = sorted(points, key=lambda p: (p.x, p.y))
        
        # 移除重複點
        unique_points = [sorted_points[0]]
        for p in sorted_points[1:]:
            if abs(p.x - unique_points[-1].x) > 1e-9 or abs(p.y - unique_points[-1].y) > 1e-9:
                unique_points.append(p)
        
        self.steps = []
        vd, _ = self._divide_conquer(unique_points)
        return vd
    
    def _divide_conquer(self, points: List[Point]) -> Tuple[VoronoiDiagram, List[Point]]:
        """遞迴分治法"""
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
        
        # === 開始記錄詳細步驟 ===
        merge_id = len(self.steps)  # 用於標識這是第幾次合併
        
        # Step 1: 顯示左邊的 Voronoi
        self.steps.append({
            'type': 'show_left_vd',
            'merge_id': merge_id,
            'vd': self._copy_vd(left_vd),
            'sites': left_vd.sites.copy(),
            'description': f'左側 Voronoi Diagram ({len(left_vd.sites)} 點)'
        })
        
        # Step 2: 顯示右邊的 Voronoi
        self.steps.append({
            'type': 'show_right_vd',
            'merge_id': merge_id,
            'vd': self._copy_vd(right_vd),
            'sites': right_vd.sites.copy(),
            'description': f'右側 Voronoi Diagram ({len(right_vd.sites)} 點)'
        })
        
        # Step 3: 顯示左邊的 Convex Hull
        self.steps.append({
            'type': 'show_left_hull',
            'merge_id': merge_id,
            'hull': left_hull.copy(),
            'sites': left_vd.sites.copy(),
            'description': '左側 Convex Hull'
        })
        
        # Step 4: 顯示右邊的 Convex Hull
        self.steps.append({
            'type': 'show_right_hull',
            'merge_id': merge_id,
            'hull': right_hull.copy(),
            'sites': right_vd.sites.copy(),
            'description': '右側 Convex Hull'
        })
        
        # 合併凸包
        merged_hull = merge_convex_hulls(left_hull, right_hull)
        
        # Step 5: 顯示合併後的 Convex Hull（重要！）
        self.steps.append({
            'type': 'show_merged_hull',
            'merge_id': merge_id,
            'hull': merged_hull.copy(),
            'left_sites': left_vd.sites.copy(),
            'right_sites': right_vd.sites.copy(),
            'description': '合併後的 Convex Hull'
        })
        
        # 從完整凸包中找出上下切線
        upper_left, upper_right, lower_left, lower_right = self._find_tangents_from_full_hull(
            merged_hull, left_vd.sites, right_vd.sites
        )
        
        # Step 6: 顯示上下切線
        self.steps.append({
            'type': 'show_tangents',
            'merge_id': merge_id,
            'upper_tangent': (upper_left, upper_right),
            'lower_tangent': (lower_left, lower_right),
            'merged_hull': merged_hull.copy(),
            'left_sites': left_vd.sites.copy(),
            'right_sites': right_vd.sites.copy(),
            'description': '上下切線'
        })
        
        # Merge
        chain_edges, left_edges, right_edges = self._trace_chain(
            left_vd, right_vd,
            upper_left, upper_right,
            lower_left, lower_right
        )
        
        # Step 7: 顯示 Hyper Plane
        self.steps.append({
            'type': 'show_hyperplane',
            'merge_id': merge_id,
            'hyperplane': [Edge(e.start, e.end) for e in chain_edges],
            'left_vd': self._copy_vd(left_vd),
            'right_vd': self._copy_vd(right_vd),
            'left_sites': left_vd.sites.copy(),
            'right_sites': right_vd.sites.copy(),
            'description': 'Hyper Plane 路徑'
        })
        
        # Step 8: 顯示消線後的 Voronoi
        self.steps.append({
            'type': 'show_after_elimination',
            'merge_id': merge_id,
            'left_edges': [Edge(e.start, e.end) for e in left_edges],
            'right_edges': [Edge(e.start, e.end) for e in right_edges],
            'hyperplane': [Edge(e.start, e.end) for e in chain_edges],
            'original_left_edges': [Edge(e.start, e.end) for e in left_vd.edges],
            'original_right_edges': [Edge(e.start, e.end) for e in right_vd.edges],
            'left_sites': left_vd.sites.copy(),
            'right_sites': right_vd.sites.copy(),
            'description': '消線後的結果'
        })
        
        # 建立最終合併結果
        merged_vd = VoronoiDiagram()
        merged_vd.sites = left_vd.sites + right_vd.sites
        merged_vd.edges = left_edges + right_edges + chain_edges
        
        for edge in merged_vd.edges:
            if edge.start and self._point_in_canvas(edge.start):
                if edge.start not in merged_vd.vertices:
                    merged_vd.vertices.append(edge.start)
            if edge.end and self._point_in_canvas(edge.end):
                if edge.end not in merged_vd.vertices:
                    merged_vd.vertices.append(edge.end)
        
        # Step 9: 最終合併結果
        self.steps.append({
            'type': 'show_merged_result',
            'merge_id': merge_id,
            'merged_vd': self._copy_vd(merged_vd),
            'merged_hull': merged_hull.copy(),  # 保留合併後的凸包！
            'all_sites': merged_vd.sites.copy(),
            'description': '合併完成'
        })
        
        return merged_vd, merged_hull
    
    def _copy_vd(self, vd: VoronoiDiagram) -> VoronoiDiagram:
        """深拷貝 VoronoiDiagram"""
        new_vd = VoronoiDiagram()
        new_vd.sites = vd.sites.copy()
        new_vd.vertices = vd.vertices.copy()
        new_vd.edges = []
        for e in vd.edges:
            new_edge = Edge(e.start, e.end)
            new_edge.site_left = e.site_left
            new_edge.site_right = e.site_right
            new_vd.edges.append(new_edge)
        return new_vd
    
    def _voronoi_one_point(self, p: Point) -> VoronoiDiagram:
        """一個點的 Voronoi diagram"""
        vd = VoronoiDiagram()
        vd.add_site(p)
        return vd
    
    def _voronoi_two_points(self, p1: Point, p2: Point) -> VoronoiDiagram:
        """兩個點的 Voronoi diagram（一條中垂線）"""
        vd = VoronoiDiagram()
        vd.add_site(p1)
        vd.add_site(p2)
        
        if abs(p1.x - p2.x) < 1e-9 and abs(p1.y - p2.y) < 1e-9:
            return vd
        
        a, b, c = perpendicular_bisector(p1, p2)
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
        
        points = sorted([p1, p2, p3], key=lambda p: (p.x, p.y))
        p1, p2, p3 = points[0], points[1], points[2]
        
        cross = ccw(p1, p2, p3)
        if abs(cross) < 1e-6:
            # 三點共線
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
        
        center = circumcenter(p1, p2, p3)
        if center is None:
            return vd
        
        line12 = perpendicular_bisector(p1, p2)
        line13 = perpendicular_bisector(p1, p3)
        line23 = perpendicular_bisector(p2, p3)
        
        if self._point_in_canvas_extended(center):
            vd.add_vertex(center)
        
        end1 = self._extend_ray_to_boundary(center, *line12, p3)
        if end1:
            edge1 = Edge(center, end1)
            edge1.site_left = p1
            edge1.site_right = p2
            vd.add_edge(edge1)
        
        end2 = self._extend_ray_to_boundary(center, *line13, p2)
        if end2:
            edge2 = Edge(center, end2)
            edge2.site_left = p1
            edge2.site_right = p3
            vd.add_edge(edge2)
        
        end3 = self._extend_ray_to_boundary(center, *line23, p1)
        if end3:
            edge3 = Edge(center, end3)
            edge3.site_left = p2
            edge3.site_right = p3
            vd.add_edge(edge3)
        
        return vd
    
    def _extend_ray_to_boundary(self, start: Point, a: float, b: float, c: float, 
                                 away_from: Point) -> Optional[Point]:
        """從 start 沿著直線延伸到邊界，方向遠離 away_from"""
        intersections = self._line_canvas_intersections(a, b, c)
        
        if not intersections:
            return None
        
        best_pt = None
        max_dist = -1
        
        for pt in intersections:
            dx = pt.x - start.x
            dy = pt.y - start.y
            ax = away_from.x - start.x
            ay = away_from.y - start.y
            dot = dx * ax + dy * ay
            
            if dot < 0:
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > max_dist:
                    max_dist = dist
                    best_pt = pt
        
        if best_pt is None and intersections:
            best_pt = max(intersections, key=lambda p: p.distance_to(away_from))
        
        return best_pt
    
    def _find_tangents_from_full_hull(self, full_hull: List[Point], 
                                       left_sites: List[Point], 
                                       right_sites: List[Point]) -> Tuple[Point, Point, Point, Point]:
        """
        從完整凸包中找出上下切線
        切線就是連接左右兩群點的凸包邊
        """
        n = len(full_hull)
        if n < 2:
            return (left_sites[0], right_sites[0], left_sites[0], right_sites[0])
        
        # 建立點集合以快速判斷
        left_set = set(left_sites)
        right_set = set(right_sites)
        
        upper_left = None
        upper_right = None
        lower_left = None
        lower_right = None
        min_y_sum = float('inf')
        max_y_sum = float('-inf')
        
        # 在凸包上找連接左右點群的邊
        for i in range(n):
            p1 = full_hull[i]
            p2 = full_hull[(i + 1) % n]
            
            # 檢查這條邊是否連接左右兩群
            if (p1 in left_set and p2 in right_set):
                y_sum = p1.y + p2.y
                # 上切線：y 座標和較小
                if y_sum < min_y_sum:
                    min_y_sum = y_sum
                    upper_left = p1
                    upper_right = p2
                # 下切線：y 座標和較大
                if y_sum > max_y_sum:
                    max_y_sum = y_sum
                    lower_left = p1
                    lower_right = p2
                    
            elif (p1 in right_set and p2 in left_set):
                y_sum = p1.y + p2.y
                # 上切線：y 座標和較小
                if y_sum < min_y_sum:
                    min_y_sum = y_sum
                    upper_left = p2
                    upper_right = p1
                # 下切線：y 座標和較大
                if y_sum > max_y_sum:
                    max_y_sum = y_sum
                    lower_left = p2
                    lower_right = p1
        
        # 如果沒找到，使用備用方案
        if upper_left is None or upper_right is None:
            # 找最上方的左右點
            upper_left = min(left_sites, key=lambda p: (p.y, -p.x))
            upper_right = min(right_sites, key=lambda p: (p.y, p.x))
        
        if lower_left is None or lower_right is None:
            # 找最下方的左右點
            lower_left = max(left_sites, key=lambda p: (p.y, -p.x))
            lower_right = max(right_sites, key=lambda p: (p.y, p.x))
        
        return upper_left, upper_right, lower_left, lower_right
    
    def _trace_chain(self, left_vd: VoronoiDiagram, right_vd: VoronoiDiagram,
                     upper_left: Point, upper_right: Point,
                     lower_left: Point, lower_right: Point) -> Tuple[List[Edge], List[Edge], List[Edge]]:
        """追蹤 dividing chain"""
        chain_edges = []
        
        # 正確複製邊的所有資訊
        left_edges = []
        for e in left_vd.edges:
            new_edge = Edge(e.start, e.end)
            new_edge.site_left = e.site_left
            new_edge.site_right = e.site_right
            left_edges.append(new_edge)
        
        right_edges = []
        for e in right_vd.edges:
            new_edge = Edge(e.start, e.end)
            new_edge.site_left = e.site_left
            new_edge.site_right = e.site_right
            right_edges.append(new_edge)
        
        cur_left = upper_left
        cur_right = upper_right
        
        a, b, c = perpendicular_bisector(cur_left, cur_right)
        cur_point = self._find_boundary_intersection(a, b, c, 'top')
        if cur_point is None:
            intersections = self._line_canvas_intersections(a, b, c)
            if intersections:
                cur_point = min(intersections, key=lambda p: p.y)
        
        max_iter = 200
        iteration = 0
        
        while iteration < max_iter:
            iteration += 1
            a, b, c = perpendicular_bisector(cur_left, cur_right)
            
            # 檢查所有邊
            next_point, hits = self._find_next_hits_all_edges(
                cur_point, a, b, c,
                cur_left, cur_right,
                left_edges, right_edges,
                left_vd.sites, right_vd.sites
            )
            
            if next_point is None:
                boundary_pt = self._find_chain_end_boundary(a, b, c, cur_point, cur_left, cur_right)
                
                if boundary_pt and cur_point:
                    edge = Edge(cur_point, boundary_pt)
                    edge.site_left, edge.site_right = cur_left, cur_right
                    chain_edges.append(edge)
                break
            
            edge = Edge(cur_point, next_point)
            edge.site_left, edge.site_right = cur_left, cur_right
            chain_edges.append(edge)
            
            if not hits:
                break
            
            # 處理所有同時發生的撞擊
            for hit_edge, hit_side, new_site in hits:
                if hit_side == 'left':
                    self._trim_edge_at_point(hit_edge, next_point, cur_left, cur_right)
                    cur_left = new_site
                else:
                    self._trim_edge_at_point(hit_edge, next_point, cur_right, cur_left)
                    cur_right = new_site
            
            if cur_left == lower_left and cur_right == lower_right:
                a, b, c = perpendicular_bisector(cur_left, cur_right)
                boundary_pt = self._find_chain_end_boundary(a, b, c, next_point, cur_left, cur_right)
                
                if boundary_pt and next_point:
                    edge = Edge(next_point, boundary_pt)
                    edge.site_left, edge.site_right = cur_left, cur_right
                    chain_edges.append(edge)
                break
            
            cur_point = next_point
        
        left_edges = [e for e in left_edges if e.start and e.end and e.start.distance_to(e.end) > 1e-6]
        right_edges = [e for e in right_edges if e.start and e.end and e.start.distance_to(e.end) > 1e-6]
        
        return chain_edges, left_edges, right_edges
    
    def _find_next_hits_all_edges(self, cur_point: Point, a: float, b: float, c: float,
                                   cur_left: Point, cur_right: Point,
                                   left_edges: List[Edge], right_edges: List[Edge],
                                   left_sites: List[Point], right_sites: List[Point]):
        """沿中垂線找最近交點，檢查所有邊"""
        if cur_point is None:
            return None, []
        
        hits = []
        best_dist = float('inf')
        best_point = None
        
        # 判斷是否是水平線
        is_horizontal = abs(a) < 1e-9
        
        if is_horizontal:
            if abs(cur_left.x - cur_right.x) > 1e-5:
                go_left = cur_left.x < cur_right.x
            else:
                mid_x = (cur_left.x + cur_right.x) / 2
                go_left = cur_point.x > mid_x
        
        def check_all_edges(edges, side, current_site, all_sites):
            """檢查所有邊"""
            nonlocal best_dist, best_point, hits
            
            for edge in edges:
                if not edge.start or not edge.end:
                    continue
                
                intersection = self._line_segment_intersection(a, b, c, edge)
                if intersection is None:
                    continue
                
                # 判斷是否在前進方向上
                if is_horizontal:
                    if go_left:
                        is_forward = intersection.x < cur_point.x - 1e-5
                        dist = cur_point.x - intersection.x
                    else:
                        is_forward = intersection.x > cur_point.x + 1e-5
                        dist = intersection.x - cur_point.x
                else:
                    is_forward = intersection.y > cur_point.y + 1e-5
                    dist = intersection.y - cur_point.y
                
                if is_forward and dist > 0:
                    other = self._get_other_site(edge, current_site, all_sites)
                    
                    if other:
                        if dist < best_dist - 1e-5:
                            best_dist = dist
                            best_point = intersection
                            hits = [(edge, side, other)]
                        elif abs(dist - best_dist) < 1e-5:
                            hits.append((edge, side, other))
        
        check_all_edges(left_edges, 'left', cur_left, left_sites)
        check_all_edges(right_edges, 'right', cur_right, right_sites)
        
        # 檢查邊界
        if is_horizontal:
            if go_left:
                boundary_pt = Point(0, cur_point.y)
                boundary_dist = cur_point.x
            else:
                boundary_pt = Point(self.canvas_width, cur_point.y)
                boundary_dist = self.canvas_width - cur_point.x
        else:
            boundary_pt = self._find_boundary_intersection(a, b, c, 'bottom')
            boundary_dist = boundary_pt.y - cur_point.y if boundary_pt else float('inf')
        
        if boundary_pt and boundary_dist > 1e-5:
            if boundary_dist < best_dist - 1e-5:
                return boundary_pt, []
        
        return best_point, hits
    
    def _find_chain_end_boundary(self, a: float, b: float, c: float, 
                                  cur_point: Point, cur_left: Point, cur_right: Point) -> Optional[Point]:
        """找 chain 結束時的邊界點"""
        is_horizontal = abs(a) < 1e-9
        
        if is_horizontal:
            if abs(cur_left.x - cur_right.x) > 1e-5:
                go_left = cur_left.x < cur_right.x
            else:
                mid_x = (cur_left.x + cur_right.x) / 2
                go_left = cur_point.x > mid_x
            
            if go_left:
                return Point(0, cur_point.y)
            else:
                return Point(self.canvas_width, cur_point.y)
        else:
            boundary_pt = self._find_boundary_intersection(a, b, c, 'bottom')
            if boundary_pt is None:
                intersections = self._line_canvas_intersections(a, b, c)
                if intersections:
                    candidates = [p for p in intersections if p.y > cur_point.y + 1e-5]
                    if candidates:
                        boundary_pt = min(candidates, key=lambda p: p.y)
            return boundary_pt
    
    def _find_boundary_intersection(self, a: float, b: float, c: float, side: str) -> Optional[Point]:
        """找直線與特定邊界的交點"""
        epsilon = 1e-9
        
        if side == 'top':
            if abs(a) > epsilon:
                x = -c / a
                if 0 <= x <= self.canvas_width:
                    return Point(x, 0)
        elif side == 'bottom':
            if abs(a) > epsilon:
                x = -(b * self.canvas_height + c) / a
                if 0 <= x <= self.canvas_width:
                    return Point(x, self.canvas_height)
        elif side == 'left':
            if abs(b) > epsilon:
                y = -c / b
                if 0 <= y <= self.canvas_height:
                    return Point(0, y)
        elif side == 'right':
            if abs(b) > epsilon:
                y = -(a * self.canvas_width + c) / b
                if 0 <= y <= self.canvas_height:
                    return Point(self.canvas_width, y)
        
        return None
    
    def _line_segment_intersection(self, a: float, b: float, c: float, edge: Edge) -> Optional[Point]:
        """計算直線與線段的交點"""
        if not edge.start or not edge.end:
            return None
        
        x1, y1 = edge.start.x, edge.start.y
        x2, y2 = edge.end.x, edge.end.y
        
        d1 = a * x1 + b * y1 + c
        d2 = a * x2 + b * y2 + c
        
        if d1 * d2 > 1e-9:
            return None
        
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
        if edge.site_left and edge.site_right:
            if edge.site_left == current_site:
                return edge.site_right
            elif edge.site_right == current_site:
                return edge.site_left
        
        if not edge.start or not edge.end:
            return None
        
        mid_x = (edge.start.x + edge.end.x) / 2
        mid_y = (edge.start.y + edge.end.y) / 2
        mid = Point(mid_x, mid_y)
        
        distances = [(s.distance_to(mid), s) for s in sites]
        distances.sort(key=lambda x: x[0])
        
        for dist, site in distances[:3]:
            if site != current_site:
                a, b, c = perpendicular_bisector(current_site, site)
                d1 = abs(a * edge.start.x + b * edge.start.y + c)
                d2 = abs(a * edge.end.x + b * edge.end.y + c)
                
                if d1 < 20 or d2 < 20:
                    return site
        
        return None
    
    def _trim_edge_at_point(self, edge: Edge, intersection: Point, site_keep: Point, site_discard: Point):
        """在交點處裁剪邊"""
        if not edge.start or not edge.end:
            return
        
        dist_start_keep = edge.start.distance_to(site_keep)
        dist_start_discard = edge.start.distance_to(site_discard)
        
        if dist_start_keep < dist_start_discard:
            edge.end = intersection
        else:
            edge.start = intersection
    
    def _line_canvas_intersections(self, a: float, b: float, c: float) -> List[Point]:
        """計算直線與畫布邊界的交點"""
        intersections = []
        epsilon = 1e-9
        
        if abs(b) > epsilon:
            y = -c / b
            if -epsilon <= y <= self.canvas_height + epsilon:
                intersections.append(Point(0, max(0, min(self.canvas_height, y))))
        
        if abs(b) > epsilon:
            y = -(a * self.canvas_width + c) / b
            if -epsilon <= y <= self.canvas_height + epsilon:
                pt = Point(self.canvas_width, max(0, min(self.canvas_height, y)))
                if not any(abs(pt.x - p.x) < epsilon and abs(pt.y - p.y) < epsilon for p in intersections):
                    intersections.append(pt)
        
        if abs(a) > epsilon:
            x = -c / a
            if -epsilon <= x <= self.canvas_width + epsilon:
                pt = Point(max(0, min(self.canvas_width, x)), 0)
                if not any(abs(pt.x - p.x) < epsilon and abs(pt.y - p.y) < epsilon for p in intersections):
                    intersections.append(pt)
        
        if abs(a) > epsilon:
            x = -(b * self.canvas_height + c) / a
            if -epsilon <= x <= self.canvas_width + epsilon:
                pt = Point(max(0, min(self.canvas_width, x)), self.canvas_height)
                if not any(abs(pt.x - p.x) < epsilon and abs(pt.y - p.y) < epsilon for p in intersections):
                    intersections.append(pt)
        
        return intersections[:2]
    
    def _point_in_canvas(self, p: Point) -> bool:
        return -1e-6 <= p.x <= self.canvas_width + 1e-6 and -1e-6 <= p.y <= self.canvas_height + 1e-6
    
    def _point_in_canvas_extended(self, p: Point) -> bool:
        margin = 50
        return -margin <= p.x <= self.canvas_width + margin and -margin <= p.y <= self.canvas_height + margin


print("Divide-and-Conquer 演算法載入完成！（v7: 詳細 step-by-step 記錄）")