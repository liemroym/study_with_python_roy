def 階乗(数):
    if 数 == 1: return 1
    return 数 * 階乗(数-1)

def 主要():
    print(階乗(4))

主要()