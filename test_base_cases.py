"""
測試 Voronoi Base Cases
"""
from voronoi_geometry import Point
from voronoi_algorithm import VoronoiDC

def test_two_points():
    """測試兩點情況"""
    print("=" * 60)
    print("測試：兩點")
    print("=" * 60)
    
    # 測試 1: 一般情況
    print("\n測試 1: 一般兩點 (289, 290) 和 (342, 541)")
    points = [Point(289, 290), Point(342, 541)]
    algo = VoronoiDC(600, 600)
    vd = algo.build(points)
    
    print(f"站點數: {len(vd.sites)}")
    print(f"邊數: {len(vd.edges)}")
    for i, edge in enumerate(vd.edges):
        if edge.start and edge.end:
            print(f"  Edge {i}: ({edge.start.x:.1f}, {edge.start.y:.1f}) -> ({edge.end.x:.1f}, {edge.end.y:.1f})")
    
    # 測試 2: 水平
    print("\n測試 2: 水平兩點 (200, 200) 和 (400, 200)")
    points = [Point(200, 200), Point(400, 200)]
    vd = algo.build(points)
    
    print(f"站點數: {len(vd.sites)}")
    print(f"邊數: {len(vd.edges)}")
    for i, edge in enumerate(vd.edges):
        if edge.start and edge.end:
            print(f"  Edge {i}: ({edge.start.x:.1f}, {edge.start.y:.1f}) -> ({edge.end.x:.1f}, {edge.end.y:.1f})")
    
    # 測試 3: 垂直
    print("\n測試 3: 垂直兩點 (200, 200) 和 (200, 400)")
    points = [Point(200, 200), Point(200, 400)]
    vd = algo.build(points)
    
    print(f"站點數: {len(vd.sites)}")
    print(f"邊數: {len(vd.edges)}")
    for i, edge in enumerate(vd.edges):
        if edge.start and edge.end:
            print(f"  Edge {i}: ({edge.start.x:.1f}, {edge.start.y:.1f}) -> ({edge.end.x:.1f}, {edge.end.y:.1f})")
    
    # 測試 4: 重複點
    print("\n測試 4: 重複點 (200, 200) 和 (200, 200)")
    points = [Point(200, 200), Point(200, 200)]
    vd = algo.build(points)
    
    print(f"站點數: {len(vd.sites)}")
    print(f"邊數: {len(vd.edges)}")

def test_three_points():
    """測試三點情況"""
    print("\n" + "=" * 60)
    print("測試：三點")
    print("=" * 60)
    
    algo = VoronoiDC(600, 600)
    
    # 測試 1: 一般三角形
    print("\n測試 1: 銳角三角形 (147, 190), (164, 361), (283, 233)")
    points = [Point(147, 190), Point(164, 361), Point(283, 233)]
    vd = algo.build(points)
    
    print(f"站點數: {len(vd.sites)}")
    print(f"頂點數: {len(vd.vertices)}")
    print(f"邊數: {len(vd.edges)}")
    for i, vertex in enumerate(vd.vertices):
        print(f"  Vertex {i}: ({vertex.x:.1f}, {vertex.y:.1f})")
    for i, edge in enumerate(vd.edges):
        if edge.start and edge.end:
            print(f"  Edge {i}: ({edge.start.x:.1f}, {edge.start.y:.1f}) -> ({edge.end.x:.1f}, {edge.end.y:.1f})")
    
    # 測試 2: 三點共線（水平）
    print("\n測試 2: 三點共線（水平）(200, 200), (300, 200), (400, 200)")
    points = [Point(200, 200), Point(300, 200), Point(400, 200)]
    vd = algo.build(points)
    
    print(f"站點數: {len(vd.sites)}")
    print(f"頂點數: {len(vd.vertices)}")
    print(f"邊數: {len(vd.edges)}")
    for i, edge in enumerate(vd.edges):
        if edge.start and edge.end:
            print(f"  Edge {i}: ({edge.start.x:.1f}, {edge.start.y:.1f}) -> ({edge.end.x:.1f}, {edge.end.y:.1f})")
    
    # 測試 3: 三點共線（垂直）
    print("\n測試 3: 三點共線（垂直）(200, 200), (200, 300), (200, 400)")
    points = [Point(200, 200), Point(200, 300), Point(200, 400)]
    vd = algo.build(points)
    
    print(f"站點數: {len(vd.sites)}")
    print(f"頂點數: {len(vd.vertices)}")
    print(f"邊數: {len(vd.edges)}")
    for i, edge in enumerate(vd.edges):
        if edge.start and edge.end:
            print(f"  Edge {i}: ({edge.start.x:.1f}, {edge.start.y:.1f}) -> ({edge.end.x:.1f}, {edge.end.y:.1f})")

def test_four_points():
    """測試四點情況（原始測試案例）"""
    print("\n" + "=" * 60)
    print("測試：四點（菱形）")
    print("=" * 60)
    
    algo = VoronoiDC(600, 600)
    
    print("\n測試: 菱形 (193, 64), (193, 370), (103, 200), (283, 200)")
    points = [Point(193, 64), Point(193, 370), Point(103, 200), Point(283, 200)]
    vd = algo.build(points)
    
    print(f"站點數: {len(vd.sites)}")
    print(f"頂點數: {len(vd.vertices)}")
    print(f"邊數: {len(vd.edges)}")
    
    print("\n頂點:")
    for i, vertex in enumerate(vd.vertices):
        print(f"  V{i}: ({vertex.x:.1f}, {vertex.y:.1f})")
    
    print("\n邊:")
    for i, edge in enumerate(vd.edges):
        if edge.start and edge.end:
            print(f"  Edge {i}: ({edge.start.x:.1f}, {edge.start.y:.1f}) -> ({edge.end.x:.1f}, {edge.end.y:.1f})")

if __name__ == "__main__":
    test_two_points()
    test_three_points()
    test_four_points()
