import numpy as np


def is_matrix(A):
    return not (len(A.shape) != 2 or A.shape[0] != A.shape[1])


def is_inversible(A):
    assert (is_matrix(A))
    return not np.isclose(np.abs(np.linalg.det(A)), 0)


def are_LI(v, w):
    assert (len(v) == len(w))
    n = len(v)
    ans = False

    f = 0
    for i in range(n):
        if (v[i] == 0 and w[i] != 0) or (w[i] == 0 and v[i] != 0):
            ans = True
        elif (v[i] != 0 and w[i] != 0) and f == 0:
            f = v[i] / w[i]
        elif (v[i] != 0 and w[i] != 0) and (v[i] / w[i] != f):
            ans = True

    return ans


def has_one_solution(A):
    assert (is_matrix(A))
    n = len(A)
    ans = True

    for i in range(n):
        for j in range(i + 1, n):
            if not are_LI(A[i], A[j]):
                ans = False

    return ans


def is_symmetric(A):
    assert (is_matrix(A))
    return np.allclose(A, A.T)


def is_diagonal(A):
    assert (is_matrix(A))
    n = len(A)
    ans = True
    for i in range(n):
        for j in range(n):
            if i != j and not np.isclose(A[i, j], 0):
                ans = False

    return ans


def is_orthogonal(A):
    assert (is_inversible(A))
    return np.allclose(A.T, np.linalg.inv(A))


def is_diagonally_dominant(A):
    assert (is_matrix(A))
    n = len(A)
    ans = True
    for i in range(n):
        sum = 0
        for j in range(n):
            if i != j:
                sum = sum + np.abs(A[i, j])
        if np.abs(A[i, i]) < sum:
            ans = False

    return ans


def is_diagonally_strictly_dominant(A):
    assert (is_matrix(A))
    n = len(A)
    ans = True
    for i in range(n):
        sum = 0
        for j in range(n):
            if i != j:
                sum = sum + np.abs(A[i, j])
        if np.abs(A[i, i]) <= sum:
            ans = False

    return ans


def has_LU(A):
    assert (is_matrix(A))
    n = len(A)
    ans = True
    for i in range(1, n):
        B = A[:i, :i]
        ans = is_inversible(B)

    return ans


def gauss_no_perm(A):
    assert (is_matrix(A))
    n = len(A)
    ans = A.copy()

    for i in range(n - 1):
        for j in range(i + 1, n):
            f = ans[j, i] / ans[i, i]
            for k in range(i, n):
                ans[j, k] = ans[j, k] - f * ans[i, k]

    return ans


def gauss(A):
    assert (is_matrix(A))

    def swap(M, i, j):
        temp = M[i].copy()
        M[i] = M[j]
        M[j] = temp

    n = len(A)
    ans = A.copy()
    row = 0
    column = 0
    while row < n and column < n:
        i_max = np.argmax(ans[row:, column]) + row
        if ans[i_max, column] == 0:
            column += 1
        else:
            swap(ans, row, i_max)
            for i in range(row + 1, n):
                f = ans[i, column] / ans[row, column]
                for j in range(column, n):
                    ans[i, j] = ans[i, j] - f * ans[row, j]
            row += 1
            column += 1

    return ans


def solve(A, b):
    assert (has_one_solution(A))

    def swap(M, i, j):
        temp = M[i].copy()
        M[i] = M[j]
        M[j] = temp

    n = len(A)
    A_work = A.copy()
    ans = b.copy()
    row = 0
    column = 0
    while row < n and column < n:
        i_max = np.argmax(A_work[row:, column]) + row
        if A_work[i_max, column] == 0:
            column += 1
        else:
            swap(A_work, row, i_max)
            swap(ans, row, i_max)
            for i in range(row + 1, n):
                f = A_work[i, column] / A_work[row, column]
                for j in range(column, n):
                    A_work[i, j] = A_work[i, j] - f * A_work[row, j]
                ans[i] = ans[i] - f * ans[row]
            row += 1
            column += 1

    k = n - 1
    while k >= 0:
        sum = ans[k]
        for l in range(k + 1, n):
            sum = sum - A_work[k, l] * ans[l]
        ans[k] = sum / A_work[k, k]
        k -= 1

    return ans


def LU(A):
    assert (has_LU(A))
    n = len(A)
    L = np.identity(n)
    U = A.copy()

    for i in range(n - 1):
        temp = np.identity(n)
        for j in range(i + 1, n):
            f = U[j, i] / U[i, i]
            for k in range(i, n):
                U[j, k] = U[j, k] - f * U[i, k]
            temp[j, i] = -f
        L = temp @ L

    L = np.linalg.inv(L)

    return L, U


def LDL(A):
    assert (has_LU(A))
    assert (is_symmetric(A))

    L, U = LU(A)
    D = np.zeros(U.shape)
    for i in range(len(D)):
        D[i, i] = U[i, i]

    return L, D


def cholesky(A):
    L, D = LDL(A)
    D = np.sqrt(D)
    L = L @ D
    return L
