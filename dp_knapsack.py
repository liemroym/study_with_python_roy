
def dp(list, y, k):
    if (k <= len(list)):
        f : list[int] = [0]*(y+1)

        print(f)
        for i in range(k):
            oldf = f.copy()
            for j in range(y+1):
                f[j] = max(oldf[j], list[i][1] + (-999999 if j-list[i][0] < 0 else oldf[j-list[i][0]]))

            print(f)
    else:
        print("k > n")

if __name__ == "__main__":
    # d p([(2, 65), (3, 80), (1, 30)], 5, 3)
    dp([(3, 25), (2, 20), (1, 15), (4, 40), (5, 50)], 6, 5)