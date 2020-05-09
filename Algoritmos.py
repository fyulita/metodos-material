import numpy as np


def gauss_no_perm(A):
    n = len(A)
    ans = A.copy()

    for i in range(n - 1):
        for j in range(i + 1, n):
            f = ans[j][i] / ans[i][i]
            for k in range(i, n):
                ans[j][k] = ans[j][k] - f * ans[i][k]

    return ans


def gauss(A):
    def swap(M, i, j):
        temp = M[i].copy()
        M[i] = M[j]
        M[j] = temp

    def column_slice(M, col, begin, end):
        ans = []
        for i in range(len(M)):
            if i < begin or i >= end:
                ans.append(-1)
            else:
                ans.append(np.abs(M[i][col]))

        return ans

    n = len(A)
    ans = A.copy()
    row = 0
    column = 0
    while row < n and column < n:
        i_max = np.argmax(column_slice(ans, column, row, n))
        if ans[i_max][column] == 0:
            column += 1
        else:
            swap(ans, row, i_max)
            for i in range(row + 1, n):
                f = ans[i][column] / ans[row][column]
                for j in range(column + 1, n):
                    ans[i][j] = ans[i][j] - f * ans[row][j]
            row += 1
            column += 1

    return ans


def solve(A, b):
    def swap(M, i, j):
        temp = M[i].copy()
        M[i] = M[j]
        M[j] = temp

    def column_slice(M, col, begin, end):
        ans = []
        for i in range(len(M)):
            if i < begin or i >= end:
                ans.append(-1)
            else:
                ans.append(np.abs(M[i][col]))

        return ans

    n = len(A)
    A_work = A.copy()
    ans = b.copy()
    row = 0
    column = 0
    while row < n and column < n:
        i_max = np.argmax(column_slice(A_work, column, row, n))
        if A_work[i_max][column] == 0:
            column += 1
        else:
            swap(A_work, row, i_max)
            swap(ans, row, i_max)
            for i in range(row + 1, n):
                f = A_work[i][column] / A_work[row][column]
                for j in range(column, n):
                    A_work[i][j] = A_work[i][j] - f * A_work[row][j]
                ans[i] = ans[i] - f * ans[row]
            row += 1
            column += 1

    k = n - 1
    while k >= 0:
        sum = ans[k]
        for l in range(k + 1, n):
            sum = sum - A_work[k][l] * ans[l]
        ans[k] = sum / A_work[k][k]
        k -= 1

    return ans


def LU(A):
    n = len(A)
    L = np.identity(n)
    U = A.copy()

    for i in range(n - 1):
        temp = np.identity(n)
        for j in range(i + 1, n):
            f = U[j][i] / U[i][i]
            for k in range(i, n):
                U[j][k] = U[j][k] - f * U[i][k]
            temp[j][i] = -f
        L = np.dot(temp, L)

    L = np.linalg.inv(L)

    return U, L


def cholesky(A):
    n = len(A)
    L = np.zeros([n, n])
    L[0][0] = np.sqrt(A[0][0])
    for i in range(1, n):
        L[i][0] = A[i][0] / L[0][0]
    for j in range(1, n):
        sum1 = 0
        for k in range(j):
            sum1 = sum1 + L[j][k] ** 2
        L[j][j] = np.sqrt(A[j][j] - sum1)
        for i in range(j + 1, n):
            sum2 = 0
            for l in range(j):
                sum2 = sum2 + L[i][l] * L[j][l]
            L[i][j] = (A[i][j] - sum2) / L[j][j]

    return L


A = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 4.0, 9.0, 16.0],
    [1.0, 8.0, 27.0, 64.0],
    [1.0, 16.0, 81.0, 256.0]
])

a = np.array([2.0, 10.0, 44.0, 190.0])

B = np.array([
    [2.0, 1.0, 2.0],
    [1.0, 2.0, 3.0],
    [4.0, 1.0, 2.0]
])

b = np.array([1.0, 0.0, 1.0])

print(cholesky(A))
