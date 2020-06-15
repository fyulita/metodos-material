import numpy as np


def is_squared(A):
    return A.shape[0] == A.shape[1]


def is_inversible(A):
    assert is_squared(A)
    return not np.isclose(np.abs(np.linalg.det(A)), 0)


def are_LI(v, w):
    assert len(v) == len(w)
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


def is_symmetric(A):
    assert is_squared(A)
    return np.allclose(A, A.T)


def is_antisymmetric(A):
    assert is_squared(A)
    return np.allclose(A, -A.T)


def is_diagonal(A):
    assert is_squared(A)
    n = len(A)
    ans = True
    for i in range(n):
        for j in range(n):
            if i != j and not np.isclose(A[i, j], 0):
                ans = False

    return ans


def is_orthogonal(A):
    assert is_inversible(A)
    return np.allclose(A.T, np.linalg.inv(A))


def is_diagonally_dominant(A):
    assert is_squared(A)
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
    assert is_squared(A)
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
    assert is_squared(A)
    n = len(A)
    ans = True
    for i in range(1, n):
        B = A[:i, :i]
        ans = is_inversible(B)

    return ans


def norm_1(x):
    n = len(x)
    sum = 0
    for i in range(n):
        sum = sum + np.abs(x[i])

    return sum


def norm_2(x):
    n = len(x)
    sum = 0
    for i in range(n):
        sum = sum + x[i] ** 2

    return np.sqrt(sum)


def norm_infty(x):
    n = len(x)
    max = 0
    for i in range(n):
        if np.abs(x[i]) > max:
            max = x[i]

    return max


def norm_p(x, p):
    n = len(x)
    sum = 0
    for i in range(n):
        sum = sum + np.abs(x[i]) ** p

    return sum ** (1 / p)


def gauss_no_perm(A):
    assert is_squared(A)
    n = len(A)
    ans = A.copy()

    for i in range(n - 1):
        for j in range(i + 1, n):
            f = ans[j, i] / ans[i, i]
            for k in range(i, n):
                ans[j, k] = ans[j, k] - f * ans[i, k]

    return ans


def gauss(A):
    assert is_squared(A)

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


def has_one_solution(A):
    triangle = gauss(A)
    n = len(triangle)
    ans = True
    for i in range(n):
        if norm_infty(triangle[i]) == 0:
            ans = False

    return ans


def LU(A):
    assert has_LU(A)
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
    assert has_LU(A)
    assert is_symmetric(A)

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


def QR_Givens(A):
    n = len(A)
    R = A.copy()
    Q = np.eye(n)
    i = 0
    j = 1
    while i < j < n:
        if R[j, i] == 0:
            if j == n - 1:
                i += 1
                j = i + 1
            else:
                j += 1
        else:
            W = np.eye(n)
            v = np.array([R[i, i], R[j, i]])
            norm = norm_2(v)
            W[i, i] = v[0] / norm
            W[i, j] = v[1] / norm
            W[j, i] = -v[1] / norm
            W[j, j] = v[0] / norm
            R = W @ R
            Q = W @ Q
            if j == n - 1:
                i += 1
                j = i + 1
            else:
                j += 1
    Q = Q.T

    return Q, R


def QR_Householder(A):
    n = len(A)
    R = A.copy()
    Q = np.eye(n)
    for i in range(n - 1):
        v = R[i:, i]
        w = np.zeros(len(v))
        w[0] = norm_2(v)
        u = (v - w) / norm_2(v - w)
        if len(u) != n:
            u = np.append(np.zeros(n - len(u)), u)
        W = np.eye(n) - 2 * np.outer(u, u)
        R = W @ R
        Q = W @ Q
    Q = Q.T

    return Q, R


def solve(A, b):
    assert has_one_solution(A)
    n = len(A)
    Q, R = QR_Givens(A)
    ans = Q.T @ b

    k = n - 1
    while k >= 0:
        sum = ans[k]
        for l in range(k + 1, n):
            sum = sum - R[k, l] * ans[l]
        ans[k] = sum / R[k, k]
        k -= 1

    return ans


def power_iteration(A, niter=50000, eps=1e-16):
    assert is_symmetric(A)
    eigenvector = np.random.rand(A.shape[1])
    old = np.ones(A.shape[1])

    i = 0
    while i < niter and not np.allclose(eigenvector, old, atol=eps):
        old = eigenvector
        eigenvector = A @ eigenvector
        eigenvector = eigenvector / np.linalg.norm(eigenvector)
        i += 1

    eigenvalue = np.dot(eigenvector, A @ eigenvector) / np.linalg.norm(eigenvector)

    return eigenvalue, eigenvector


def eigen(A, num=None, **kwargs):
    assert is_symmetric(A)
    if num is None or num > A.shape[0]:
        num = A.shape[0]

    A = A.copy()
    eigenvalues = []
    eigenvectors = np.zeros((A.shape[0], num))

    for i in range(num):
        l, v = power_iteration(A, **kwargs)
        eigenvalues.append(l)
        eigenvectors[:, i] = v

        A = A - l * np.outer(v, v)

    return np.array(eigenvalues), eigenvectors


def singular_values(A):
    lambs, U = eigen(A @ A.T)
    lambs, V = eigen(A.T @ A)
    Sigma = U.T @ A @ V

    return U, Sigma, V


def jacobi(A, b, niter=10000):
    assert is_diagonally_strictly_dominant(A)
    x = np.zeros(A.shape[1])

    D = np.diagflat(np.diag(A))
    L = -(np.tril(A) - D)
    U = -(np.triu(A) - D)

    T = np.linalg.inv(D) @ (L + U)
    c = np.linalg.inv(D) @ b

    for i in range(niter):
        x = T @ x + c
    return x


def gauss_siedel(A, b, niter=5000):
    assert is_diagonally_strictly_dominant(A)
    x = np.zeros(A.shape[1])

    D = np.diagflat(np.diag(A))
    L = -(np.tril(A) - D)
    U = -(np.triu(A) - D)

    T = np.linalg.inv(D - L) @ U
    c = np.linalg.inv(D - L) @ b

    for i in range(niter):
        x = T @ x + c
    return x
