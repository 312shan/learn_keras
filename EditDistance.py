# 参考：https://www.geeksforgeeks.org/edit-distance-dp-5/
def edit_dist(a, b, m, n):
    if m == 0:
        return n
    if n == 0:
        return m
    if a[m - 1] == b[n - 1]:
        return edit_dist(a, b, m - 1, n - 1)
    else:
        return min(edit_dist(a, b, m - 1, n),  # remove(移除 a[m-1]，n 不变因为b[n-1]没有匹配)
                   edit_dist(a, b, m, n - 1),  # insert(在 a[m-1] 后面插入一个元素来和 b[n-1]匹配，所以 m 不变，b[n-1]已经被匹配，所以n-1给到下一步)
                   edit_dist(a, b, m - 1, n - 1)  # replace(replace 当然构成a[m-1] b[n-1]匹配，m-1,n-1 给到下一步)
                   ) + 1


def edit_dist_dp(a, b, m, n):
    dp = [[0] * (n + 1) for _ in range(m + 1)]  # 小心，这里需要构造的是 m+1*n+1 的矩阵哟
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif a[i - 1] == b[j - 1]:  # cur last is same
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    import numpy as np
    print(np.asarray(dp))
    return dp[m][n]


a = "sunday"
b = "saturday"
d = edit_dist(a, b, len(a), len(b))
print(d)
d = edit_dist_dp(a, b, len(a), len(b))
print(d)

"""
[[0 1 2 3 4 5 6 7 8]
 [1 0 1 2 3 4 5 6 7]
 [2 1 1 2 2 3 4 5 6]
 [3 2 2 2 3 3 4 5 6]
 [4 3 3 3 3 4 3 4 5]
 [5 4 3 4 4 4 4 3 4]
 [6 5 4 4 5 5 5 4 3]]
3

Process finished with exit code 0
从打印的矩阵可以看出，结合代码，可以发现，求解每一行其实只依赖于上一行 j 、j-1 的信息 和 i-1 的信息，所以dp还可以优化为一个 m*2 的矩阵
"""
