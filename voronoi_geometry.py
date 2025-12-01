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
        self.site_left = None   # 左側的 site
        self.site_right = None  # 右側的 site
    
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
    
    # p1 到 p2 的方向向量
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    
    # 中垂線的法向量就是 (dx, dy)（垂直於線段的向量）
    # 中垂線方程: dx*(x - mx) + dy*(y - my) = 0
    # 展開: dx*x + dy*y - dx*mx - dy*my = 0
    a = dx
    b = dy
    c = -dx * mx - dy * my
    
    return (a, b, c)


def line_intersection(a1: float, b1: float, c1: float, 
                      a2: float, b2: float, c2: float) -> Optional[Point]:
    """
    計算兩條直線的交點
    L1: a1*x + b1*y + c1 = 0
    L2: a2*x + b2*y + c2 = 0
    
    使用 Cramer's rule 求解:
    [a1  b1] [x]   [-c1]
    [a2  b2] [y] = [-c2]
    
    det = a1*b2 - a2*b1
    x = (-c1*b2 + c2*b1) / det
    y = (-a1*c2 + a2*c1) / det
    """
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:  # 平行
        return None
    
    x = (-c1 * b2 + c2 * b1) / det
    y = (-a1 * c2 + a2 * c1) / det
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
    # [修正 1] 移除 len <= 2 的特殊判斷，統一走下面的排序邏輯
    # 這樣可以保證即使只有 2 點，也是依照「角度/逆時針」順序排列，而非單純的 X 軸順序
    if not points:
        return []
    if len(points) <= 2:
        # 對於 2 點，直接按座標排序可能不符合 CCW 邏輯（取決於相對位置）
        # 最安全的方法是讓它們像多點一樣，找最低點然後按角度排
        pass 

    # 找到最下面最左邊的點
    start = min(points, key=lambda p: (p.y, p.x))
    
    # 按極角排序
    def polar_angle_key(p: Point):
        dx = p.x - start.x
        dy = p.y - start.y
        # 使用 atan2 確保角度正確
        return (math.atan2(dy, dx), dx*dx + dy*dy)
    
    sorted_points = [start] + sorted([p for p in points if p != start], key=polar_angle_key)
    
    # Graham Scan
    hull = []
    for p in sorted_points:
        # [修正 2] 將 <= 0 改為 < 0
        # <= 0 會移除共線的點 (只留兩端)
        # < 0 則會保留共線的點 (這對 Voronoi 的切線搜尋較為穩健)
        while len(hull) > 1 and ccw(hull[-2], hull[-1], p) < 0:
            hull.pop()
        hull.append(p)
    
    return hull


def circumcenter(p1: Point, p2: Point, p3: Point) -> Optional[Point]:
    """
    計算三點的外心（外接圓圓心）
    外心是三條中垂線的交點
    """
    # 檢查是否共線
    area = abs((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))
    if area < 1e-9:
        return None
    
    line12 = perpendicular_bisector(p1, p2)
    line13 = perpendicular_bisector(p1, p3)
    
    return line_intersection(*line12, *line13)


def merge_convex_hulls(left_hull: List[Point], right_hull: List[Point]) -> List[Point]:
    """
    合併兩個已排序的凸包，時間複雜度 O(n)
    假設左凸包的所有點都在右凸包的左邊
    返回合併後的凸包（逆時針順序）
    """
    if not left_hull:
        return right_hull.copy()
    if not right_hull:
        return left_hull.copy()
    
    # 找上下切線
    upper_left_idx, upper_right_idx = _find_upper_tangent_idx(left_hull, right_hull)
    lower_left_idx, lower_right_idx = _find_lower_tangent_idx(left_hull, right_hull)
    
    # 構建合併後的凸包
    # 從左凸包的下切點開始，逆時針到上切點
    # 然後從右凸包的上切點，逆時針到下切點
    merged = []
    
    n_left = len(left_hull)
    n_right = len(right_hull)
    
    # 從左凸包下切點逆時針走到上切點
    idx = lower_left_idx
    while True:
        merged.append(left_hull[idx])
        if idx == upper_left_idx:
            break
        idx = (idx - 1 + n_left) % n_left
    
    # 從右凸包上切點逆時針走到下切點
    idx = upper_right_idx
    while True:
        merged.append(right_hull[idx])
        if idx == lower_right_idx:
            break
        idx = (idx - 1 + n_right) % n_right
    
    return merged


def _find_upper_tangent_idx(left_hull: List[Point], right_hull: List[Point]) -> Tuple[int, int]:
    """找上切線的索引，O(n)"""
    # 左凸包最右邊的點
    left_idx = 0
    for i in range(len(left_hull)):
        if left_hull[i].x > left_hull[left_idx].x or \
           (abs(left_hull[i].x - left_hull[left_idx].x) < 1e-9 and left_hull[i].y < left_hull[left_idx].y):
            left_idx = i
    
    # 右凸包最左邊的點
    right_idx = 0
    for i in range(len(right_hull)):
        if right_hull[i].x < right_hull[right_idx].x or \
           (abs(right_hull[i].x - right_hull[right_idx].x) < 1e-9 and right_hull[i].y < right_hull[right_idx].y):
            right_idx = i
    
    n_left = len(left_hull)
    n_right = len(right_hull)
    
    done = False
    while not done:
        done = True
        
        # 左凸包逆時針移動
        while n_left > 1:
            next_idx = (left_idx - 1 + n_left) % n_left
            if ccw(right_hull[right_idx], left_hull[left_idx], left_hull[next_idx]) > 1e-9:
                left_idx = next_idx
                done = False
            else:
                break
        
        # 右凸包順時針移動
        while n_right > 1:
            next_idx = (right_idx + 1) % n_right
            if ccw(left_hull[left_idx], right_hull[right_idx], right_hull[next_idx]) < -1e-9:
                right_idx = next_idx
                done = False
            else:
                break
    
    return left_idx, right_idx


def _find_lower_tangent_idx(left_hull: List[Point], right_hull: List[Point]) -> Tuple[int, int]:
    """找下切線的索引，O(n)"""
    # 左凸包最右邊的點
    left_idx = 0
    for i in range(len(left_hull)):
        if left_hull[i].x > left_hull[left_idx].x or \
           (abs(left_hull[i].x - left_hull[left_idx].x) < 1e-9 and left_hull[i].y > left_hull[left_idx].y):
            left_idx = i
    
    # 右凸包最左邊的點
    right_idx = 0
    for i in range(len(right_hull)):
        if right_hull[i].x < right_hull[right_idx].x or \
           (abs(right_hull[i].x - right_hull[right_idx].x) < 1e-9 and right_hull[i].y > right_hull[right_idx].y):
            right_idx = i
    
    n_left = len(left_hull)
    n_right = len(right_hull)
    
    done = False
    while not done:
        done = True
        
        # 左凸包順時針移動
        while n_left > 1:
            next_idx = (left_idx + 1) % n_left
            if ccw(right_hull[right_idx], left_hull[left_idx], left_hull[next_idx]) < -1e-9:
                left_idx = next_idx
                done = False
            else:
                break
        
        # 右凸包逆時針移動
        while n_right > 1:
            next_idx = (right_idx - 1 + n_right) % n_right
            if ccw(left_hull[left_idx], right_hull[right_idx], right_hull[next_idx]) > 1e-9:
                right_idx = next_idx
                done = False
            else:
                break
    
    return left_idx, right_idx


print("幾何基礎類別載入完成！")