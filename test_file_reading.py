"""
測試檔案讀取功能
"""

def read_test_file(filename):
    """讀取測試檔案"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    test_groups = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        
        # 跳過註解和空行
        if not line or line.startswith('#'):
            continue
        
        # 讀取點數 n
        try:
            n = int(line)
        except ValueError:
            continue
        
        # 如果 n = 0，結束
        if n == 0:
            print("讀入點數為零，檔案測試停止")
            break
        
        # 讀取 n 個點
        points = []
        for j in range(n):
            if i >= len(lines):
                break
            
            point_line = lines[i].strip()
            i += 1
            
            # 跳過註解
            if point_line.startswith('#'):
                j -= 1
                continue
            
            parts = point_line.split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    points.append((x, y))
                except ValueError:
                    continue
        
        if points:
            test_groups.append({
                'n': n,
                'points': points
            })
    
    return test_groups


if __name__ == "__main__":
    # 測試單組檔案
    print("=" * 60)
    print("測試 test_input.txt (單組測試)")
    print("=" * 60)
    groups = read_test_file('test_input.txt')
    
    for i, group in enumerate(groups):
        print(f"\n測試組 {i+1}:")
        print(f"  點數: {group['n']}")
        print(f"  座標:")
        for j, (x, y) in enumerate(group['points']):
            print(f"    點 {j+1}: ({x}, {y})")
    
    # 測試多組檔案
    print("\n" + "=" * 60)
    print("測試 test_input_multi.txt (多組測試)")
    print("=" * 60)
    groups = read_test_file('test_input_multi.txt')
    
    for i, group in enumerate(groups):
        print(f"\n測試組 {i+1}:")
        print(f"  點數: {group['n']}")
        print(f"  座標:")
        for j, (x, y) in enumerate(group['points']):
            print(f"    點 {j+1}: ({x}, {y})")
