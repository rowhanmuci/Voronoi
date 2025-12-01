"""
Voronoi Diagram - 幾何基礎類別
包含點、邊、DCEL 等基本資料結構
"""
import math
from typing import List, Tuple, Optional

class Point:
    """二維點"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"
    
    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9
    
    def __hash__(self):
        return hash((round(self.x, 6), round(self.y, 6)))
    
    def distance_to(self, other: 'Point') -> float:
        """計算到另一點的距離"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __lt__(self, other):
        """用於排序：先比較 x，再比較 y"""
        if abs(self.x - other.x) > 1e-9:
            return self.x < other.x
        return self.y < other.y


class Edge:
    """Voronoi diagram 的邊"""
    def __init__(self, start: Optional[Point] = None, end: Optional[Point] = None):
        self.start = start
        self.end = end
        self.twin = None
        self.next = None
        self.prev = None
        self.site_left = None
        self.site_right = None
    
    def __repr__(self):
        return f"Edge({self.start} -> {self.end})"
    
    def is_ray(self) -> bool:
        return self.start is None or self.end is None
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        if self.start and self.end:
            if self.start.x < self.end.x or (abs(self.start.x - self.end.x) < 1e-9 and self.start.y <= self.end.y):
                return (self.start.x, self.start.y, self.end.x, self.end.y)
            else:
                return (self.end.x, self.end.y, self.start.x, self.start.y)
        return None


class VoronoiDiagram:
    """Voronoi diagram 資料結構"""
    def __init__(self):
        self.sites: List[Point] = []
        self.edges: List[Edge] = []
        self.vertices: List[Point] = []
    
    def add_site(self, site: Point):
        self.sites.append(site)
    
    def add_edge(self, edge: Edge):
        self.edges.append(edge)
    
    def add_vertex(self, vertex: Point):
        if vertex not in self.vertices:
            self.vertices.append(vertex)


def perpendicular_bisector(p1: Point, p2: Point) -> Tuple[float, float, float]:
    """計算兩點的中垂線，返回 (a, b, c) 使得 ax + by + c = 0"""
    mx = (p1.x + p2.x) / 2
    my = (p1.y + p2.y) / 2
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    a = dx
    b = dy
    c = -dx * mx - dy * my
    return (a, b, c)


def line_intersection(a1: float, b1: float, c1: float, 
                      a2: float, b2: float, c2: float) -> Optional[Point]:
    """計算兩條直線的交點"""
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    x = (-c1 * b2 + c2 * b1) / det
    y = (-a1 * c2 + a2 * c1) / det
    return Point(x, y)


def ccw(a: Point, b: Point, c: Point) -> float:
    """計算三點的方向 (>0: 逆時針, <0: 順時針, =0: 共線)"""
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def convex_hull(points: List[Point]) -> List[Point]:
    """計算凸包 (Graham Scan)，返回逆時針順序"""
    if not points:
        return []
    if len(points) == 1:
        return points.copy()
    if len(points) == 2:
        return points.copy()

    start = min(points, key=lambda p: (p.y, p.x))
    
    def polar_angle_key(p: Point):
        dx = p.x - start.x
        dy = p.y - start.y
        return (math.atan2(dy, dx), dx*dx + dy*dy)
    
    sorted_points = [start] + sorted([p for p in points if p != start], key=polar_angle_key)
    
    hull = []
    for p in sorted_points:
        while len(hull) > 1 and ccw(hull[-2], hull[-1], p) < 0:
            hull.pop()
        hull.append(p)
    
    return hull


def circumcenter(p1: Point, p2: Point, p3: Point) -> Optional[Point]:
    """計算三點的外心"""
    area = abs((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))
    if area < 1e-9:
        return None
    line12 = perpendicular_bisector(p1, p2)
    line13 = perpendicular_bisector(p1, p3)
    return line_intersection(*line12, *line13)


def merge_convex_hulls(left_hull: List[Point], right_hull: List[Point]) -> List[Point]:
    """合併兩個凸包，O(n)"""
    if not left_hull:
        return right_hull.copy()
    if not right_hull:
        return left_hull.copy()
    
    upper_left_idx, upper_right_idx = _find_upper_tangent_idx(left_hull, right_hull)
    lower_left_idx, lower_right_idx = _find_lower_tangent_idx(left_hull, right_hull)
    
    merged = []
    n_left = len(left_hull)
    n_right = len(right_hull)
    
    idx = lower_left_idx
    while True:
        merged.append(left_hull[idx])
        if idx == upper_left_idx:
            break
        idx = (idx - 1 + n_left) % n_left
    
    idx = upper_right_idx
    while True:
        merged.append(right_hull[idx])
        if idx == lower_right_idx:
            break
        idx = (idx - 1 + n_right) % n_right
    
    return merged


def _find_upper_tangent_idx(left_hull: List[Point], right_hull: List[Point]) -> Tuple[int, int]:
    """找上切線索引"""
    left_idx = 0
    for i in range(len(left_hull)):
        if left_hull[i].x > left_hull[left_idx].x or \
           (abs(left_hull[i].x - left_hull[left_idx].x) < 1e-9 and left_hull[i].y < left_hull[left_idx].y):
            left_idx = i
    
    right_idx = 0
    for i in range(len(right_hull)):
        if right_hull[i].x < right_hull[right_idx].x or \
           (abs(right_hull[i].x - right_hull[right_idx].x) < 1e-9 and right_hull[i].y < right_hull[right_idx].y):
            right_idx = i
    
    n_left, n_right = len(left_hull), len(right_hull)
    
    done = False
    while not done:
        done = True
        while n_left > 1:
            next_idx = (left_idx - 1 + n_left) % n_left
            if ccw(right_hull[right_idx], left_hull[left_idx], left_hull[next_idx]) > 1e-9:
                left_idx = next_idx
                done = False
            else:
                break
        while n_right > 1:
            next_idx = (right_idx + 1) % n_right
            if ccw(left_hull[left_idx], right_hull[right_idx], right_hull[next_idx]) < -1e-9:
                right_idx = next_idx
                done = False
            else:
                break
    
    return left_idx, right_idx


def _find_lower_tangent_idx(left_hull: List[Point], right_hull: List[Point]) -> Tuple[int, int]:
    """找下切線索引"""
    left_idx = 0
    for i in range(len(left_hull)):
        if left_hull[i].x > left_hull[left_idx].x or \
           (abs(left_hull[i].x - left_hull[left_idx].x) < 1e-9 and left_hull[i].y > left_hull[left_idx].y):
            left_idx = i
    
    right_idx = 0
    for i in range(len(right_hull)):
        if right_hull[i].x < right_hull[right_idx].x or \
           (abs(right_hull[i].x - right_hull[right_idx].x) < 1e-9 and right_hull[i].y > right_hull[right_idx].y):
            right_idx = i
    
    n_left, n_right = len(left_hull), len(right_hull)
    
    done = False
    while not done:
        done = True
        while n_left > 1:
            next_idx = (left_idx + 1) % n_left
            if ccw(right_hull[right_idx], left_hull[left_idx], left_hull[next_idx]) < -1e-9:
                left_idx = next_idx
                done = False
            else:
                break
        while n_right > 1:
            next_idx = (right_idx - 1 + n_right) % n_right
            if ccw(left_hull[left_idx], right_hull[right_idx], right_hull[next_idx]) > 1e-9:
                right_idx = next_idx
                done = False
            else:
                break
    
    return left_idx, right_idx


print("幾何基礎類別載入完成！")