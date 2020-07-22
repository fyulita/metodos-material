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
    ans = A[0, 0] > 0
    i = 1
    B = A[:i + 1, :i + 1]
    while i < n and ans:
        B = A[:i + 1, :i + 1]
	ans = np.linalg.det(np.linalg.inv(B)) > 0
	i += 1

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


def lagrange_polynomial(x_s, y_s, x):
    def L(x_s, k, x):
        x_k = x_s[k]
        x_s = np.delete(x_s, k)
        L = np.prod(x - x_s) / np.prod(x_k - x_s)

        return L

    P = 0
    for k in range(len(y_s)):
        P = P + y_s[k] * L(x_s, k, x)

    return P


def divided_differences(x_s, y_s, x):
    def brackets(x_s, y_s):
        if len(y_s) == 1:
            return y_s[0]
        else:
            return (brackets(x_s[1:], y_s[1:]) - brackets(x_s[:-1], y_s[:-1])) / (x_s[-1] - x_s[0])

    P = y_s[0]
    for k in range(1, len(x_s) - 1):
        P = P + brackets(x_s[:k + 1], y_s[:k + 1]) * np.prod(x - x_s[:k])

    return P


def lineal_interpolation(x_s, y_s, x):
    if x <= x_s[0]:
        return ((y_s[1] - y_s[0]) / (x_s[1] - x_s[0])) * (x - x_s[0]) + y_s[0]
    elif x >= x_s[-1]:
        return ((y_s[-1] - y_s[-2]) / (x_s[-1] - x_s[-2])) * (x - x_s[-1]) + y_s[-1]
    else:
        i = 0
        while not (x_s[i] <= x < x_s[i + 1]):
            i += 1
        return ((y_s[i + 1] - y_s[i]) / (x_s[i + 1] - x_s[i])) * (x - x_s[i]) + y_s[i]


def symmetric_differences(x, eps, f, **kwargs):
    return (f(x + eps, **kwargs) - f(x - eps, **kwargs)) / (2 * eps)


def trapezoidal(a, b, n, f, **kwargs):
    intervals = np.linspace(a, b, n)
    h = (b - a) / n
    sum = 0
    for i in intervals[1:n - 2]:
        sum = sum + f(i, **kwargs)

    return h / 2 * (f(intervals[0], **kwargs) + 2 * sum + f(intervals[-1], **kwargs))


def simpson(a, b, n, f, **kwargs):
    if n % 2 != 0:
        n += 1
    intervals = np.linspace(a, b, n)
    h = (b - a) / n
    sum = 0
    for i in range(int(n / 2 - 2)):
        sum = sum + f(intervals[2 * i], **kwargs) + 4 * f(intervals[2 * i + 1], **kwargs) + f(intervals[2 * i + 2], **kwargs)

    return h / 3 * sum


def bisection(left, right, N, eps, f, **kwargs):
    mid = (left + right) / 2
    f_mid = f(mid, **kwargs)
    i = 0
    while i < N and (np.abs(f_mid) < eps or mid < eps):
        mid = (left + right) / 2
        f_left = f(left, **kwargs)
        f_mid = f(mid, **kwargs)

        if f_mid * f_left > 0:
            left = mid
        else:
            right = mid

        i += 1

    return mid


def newton(x, eps, N, f, df, **kwargs):
    fx = f(x, **kwargs)
    i = 0
    while i < N and np.abs(fx) > eps:
        x = x - f(x, **kwargs) / df(x, **kwargs)
        fx = f(x, **kwargs)
        i += 1

    return x
