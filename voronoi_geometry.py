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
        self.start = start  # 起點（可能為 None，代表射線）
        self.end = end      # 終點（可能為 None，代表射線）
        self.twin = None    # DCEL 中的對偶邊
        self.next = None    # DCEL 中的下一條邊
        self.prev = None    # DCEL 中的前一條邊
        self.site = None    # 這條邊所屬的 site
    
    def __repr__(self):
        return f"Edge({self.start} -> {self.end})"
    
    def is_ray(self) -> bool:
        """判斷是否為射線"""
        return self.start is None or self.end is None
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        """轉換為 (x1, y1, x2, y2) 格式，用於輸出"""
        if self.start and self.end:
            # 確保 x1 <= x2，或 x1 == x2 且 y1 <= y2
            if self.start.x < self.end.x or (abs(self.start.x - self.end.x) < 1e-9 and self.start.y <= self.end.y):
                return (self.start.x, self.start.y, self.end.x, self.end.y)
            else:
                return (self.end.x, self.end.y, self.start.x, self.start.y)
        return None


class VoronoiDiagram:
    """Voronoi diagram 資料結構"""
    def __init__(self):
        self.sites: List[Point] = []      # 所有的站點
        self.edges: List[Edge] = []        # 所有的邊
        self.vertices: List[Point] = []    # 所有的頂點
    
    def add_site(self, site: Point):
        """添加站點"""
        self.sites.append(site)
    
    def add_edge(self, edge: Edge):
        """添加邊"""
        self.edges.append(edge)
    
    def add_vertex(self, vertex: Point):
        """添加頂點"""
        if vertex not in self.vertices:
            self.vertices.append(vertex)


def perpendicular_bisector(p1: Point, p2: Point) -> Tuple[float, float, float]:
    """
    計算兩點的中垂線
    返回: (a, b, c) 使得 ax + by + c = 0
    中垂線通過中點，垂直於 p1p2
    """
    # 中點
    mx = (p1.x + p2.x) / 2
    my = (p1.y + p2.y) / 2
    
    # 方向向量
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    
    # 中垂線方向向量為 (-dy, dx)
    # 中垂線方程: -dy * (x - mx) + dx * (y - my) = 0
    # 整理為: -dy * x + dx * y + (dy * mx - dx * my) = 0
    a = -dy
    b = dx
    c = dy * mx - dx * my
    
    return (a, b, c)


def line_intersection(a1: float, b1: float, c1: float, 
                      a2: float, b2: float, c2: float) -> Optional[Point]:
    """
    計算兩條直線的交點
    L1: a1*x + b1*y + c1 = 0
    L2: a2*x + b2*y + c2 = 0
    """
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:  # 平行
        return None
    
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return Point(x, y)


def ccw(a: Point, b: Point, c: Point) -> float:
    """
    計算三點的方向
    > 0: 逆時針
    < 0: 順時針
    = 0: 共線
    """
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def convex_hull(points: List[Point]) -> List[Point]:
    """
    計算凸包 (使用 Graham Scan)
    返回按逆時針順序排列的凸包頂點
    """
    if len(points) <= 2:
        return points.copy()
    
    # 找到最下面最左邊的點
    start = min(points, key=lambda p: (p.y, p.x))
    
    # 按極角排序
    def polar_angle_key(p: Point):
        dx = p.x - start.x
        dy = p.y - start.y
        return (math.atan2(dy, dx), dx*dx + dy*dy)
    
    sorted_points = [start] + sorted([p for p in points if p != start], key=polar_angle_key)
    
    # Graham Scan
    hull = []
    for p in sorted_points:
        while len(hull) > 1 and ccw(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    
    return hull


print("幾何基礎類別載入完成！")
