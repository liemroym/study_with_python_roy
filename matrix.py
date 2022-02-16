def matrix():
    m = int(input("Masukkan M: "))
    n = int(input("Masukkan N: "))

    result = []
    matrix = []

    for i in range(m):
        for j in range(n):
            value = int(input("Masukkan nilai ke [{}], [{}]: ".format(i+1, j+1)))    
            matrix += [value]

        result += [matrix]
        matrix = []
    print(*result, sep='\n')
            
matrix()