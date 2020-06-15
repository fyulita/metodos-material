from Algoritmos import *
import traceback


'''############# Test Functions #############'''

def Test(func):
    global tests

    tests.append(func)

    return func


def run_tests():
    exceptions = []
    for test in tests:
        try:
            print("Running {} ... ".format(test.__name__), end='')
            test()
            print("OK")
        except AssertionError as e:
            error_msg = traceback.format_exc()
            exceptions.append((test, error_msg))
            print("ERROR")

    if len(exceptions) > 0:
        print("\nErrors:\n")
        for (test, error_msg) in exceptions:
            print("In {}".format(test.__name__))
            print(error_msg)
    else:
        print("\n\nAll tests passed!")


'''############# Tests #############'''
tests = []

'''##### has_LU Tests #####'''

@Test
def has_LU_1():
    A = np.identity(3)

    assert has_LU(A)


@Test
def has_LU_2():
    A = np.zeros((3, 3))

    assert not has_LU(A)


@Test
def has_LU_3():
    A = np.ones((3, 3))

    assert not has_LU(A)


@Test
def has_LU_4():
    A = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])

    assert not has_LU(A)


'''##### is_orthogonal Tests #####'''


@Test
def is_orthogonal_1():
    A = np.eye(3)

    assert is_orthogonal(A)


@Test
def is_orthogonal_2():
    angle = np.pi / 4
    A = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    assert is_orthogonal(A)


@Test
def is_orthogonal_3():
    A = np.array([
        [1, 0],
        [0, -1]
    ])

    assert is_orthogonal(A)


'''##### LU Tests #####'''


@Test
def LU_1():
    A = 3 * np.identity(3)

    L, U = LU(A)

    assert np.allclose(L, np.eye(3))
    assert np.allclose(U, 3 * np.eye(3))


@Test
def LU_2():
    L = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ])

    U = np.array([
        [1, 1, 1],
        [0, 2, 2],
        [0, 0, 3],
    ])

    A = L @ U

    L_obt, U_obt = LU(A)

    assert np.allclose(L, L_obt)
    assert np.allclose(U, U_obt)


'''##### cholesky Tests #####'''


@Test
def cholesky_1():
    A = np.eye(3)
    L = cholesky(A)

    assert np.allclose(A, L)


@Test
def cholesky_2():
    A = 4 * np.eye(3)
    L = cholesky(A)

    assert np.allclose(L, 2 * np.eye(3))


@Test
def cholesky_3():
    L1 = np.array([
        [1, 0, 0],
        [2, 2, 0],
        [4, 4, 4]
    ])

    A = L1 @ L1.T
    L = cholesky(A)
    assert np.allclose(L, L1)


'''##### QR_Givens Tests #####'''


@Test
def QR_Givens_1():
    A = np.array([
        [1, 1],
        [1, 2]
    ])
    Q_expected = np.array([
        [1, -1],
        [1, 1]
    ]) / np.sqrt(2)
    R_expected = np.array([
        [2, 3],
        [0, 1]
    ]) / np.sqrt(2)

    Q, R = QR_Givens(A)

    assert np.allclose(Q, Q_expected)
    assert np.allclose(R, R_expected)


@Test
def QR_Givens_2():
    A = np.array([
        [1, 2, 0],
        [0, 1, 1],
        [1, 0, 1]
    ])
    Q_expected = np.array([
        [1 / np.sqrt(2), 1 / np.sqrt(3), -1 / np.sqrt(6)],
        [0, 1 / np.sqrt(3), 2 / np.sqrt(6)],
        [1 / np.sqrt(2), -1 / np.sqrt(3), 1 / np.sqrt(6)]
    ])
    R_expected = np.array([
        [np.sqrt(2), np.sqrt(2), 1 / np.sqrt(2)],
        [0, np.sqrt(3), 0],
        [0, 0, np.sqrt(6) / 2]
    ])

    Q, R = QR_Givens(A)

    assert np.allclose(Q, Q_expected)
    assert np.allclose(R, R_expected)


@Test
def QR_Givens_3():
    A = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ])
    Q_expected = np.array([
        [1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(3)],
        [1 / np.sqrt(2), -1 / np.sqrt(6), -1 / np.sqrt(3)],
        [0, 2 / np.sqrt(6), -1 / np.sqrt(3)]
    ])
    R_expected = np.array([
        [np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2)],
        [0, 3 / np.sqrt(6), 1 / np.sqrt(6)],
        [0, 0, -2 / np.sqrt(3)]
    ])
    
    Q, R = QR_Givens(A)
    
    assert np.allclose(Q_expected, Q)
    assert np.allclose(R_expected, R)


@Test
def QR_Givens_4():
    A = np.array([
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41]
    ])
    Q_expected = np.array([
        [6 / 7, -69 / 175, 58 / 175],
        [3 / 7, 158 / 175, -6 / 175],
        [-2 / 7, 6 / 35, 33 / 35]
    ])
    R_expected = np.array([
        [14, 21, -14],
        [0, 175, -70],
        [0, 0, -35]
    ])

    Q, R = QR_Givens(A)

    assert np.allclose(Q, Q_expected)
    assert np.allclose(R, R_expected)


'''##### QR_Householder Tests #####'''


@Test
def QR_Householder_1():
    A = np.array([
        [1, 1],
        [1, 2]
    ])
    Q_expected = np.array([
        [1, 1],
        [1, -1]
    ]) / np.sqrt(2)
    R_expected = np.array([
        [2, 3],
        [0, -1]
    ]) / np.sqrt(2)

    Q, R = QR_Householder(A)

    assert np.allclose(Q, Q_expected)
    assert np.allclose(R, R_expected)


@Test
def QR_Householder_2():
    A = np.array([
        [1, 2, 0],
        [0, 1, 1],
        [1, 0, 1]
    ])
    Q_expected = np.array([
        [1 / np.sqrt(2), 1 / np.sqrt(3), -1 / np.sqrt(6)],
        [0, 1 / np.sqrt(3), 2 / np.sqrt(6)],
        [1 / np.sqrt(2), -1 / np.sqrt(3), 1 / np.sqrt(6)]
    ])
    R_expected = np.array([
        [np.sqrt(2), np.sqrt(2), 1 / np.sqrt(2)],
        [0, np.sqrt(3), 0],
        [0, 0, np.sqrt(6) / 2]
    ])

    Q, R = QR_Householder(A)

    assert np.allclose(Q, Q_expected)
    assert np.allclose(R, R_expected)


@Test
def QR_Householder_3():
    A = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ])
    Q_expected = np.array([
        [1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(3)],
        [1 / np.sqrt(2), -1 / np.sqrt(6), -1 / np.sqrt(3)],
        [0, 2 / np.sqrt(6), -1 / np.sqrt(3)]
    ])
    R_expected = np.array([
        [np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2)],
        [0, 3 / np.sqrt(6), 1 / np.sqrt(6)],
        [0, 0, -2 / np.sqrt(3)]
    ])

    Q, R = QR_Householder(A)

    assert np.allclose(Q_expected, Q)
    assert np.allclose(R_expected, R)


@Test
def QR_Householder_4():
    A = np.array([
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41]
    ])
    Q_expected = np.array([
        [6 / 7, -69 / 175, 58 / 175],
        [3 / 7, 158 / 175, -6 / 175],
        [-2 / 7, 6 / 35, 33 / 35]
    ])
    R_expected = np.array([
        [14, 21, -14],
        [0, 175, -70],
        [0, 0, -35]
    ])

    Q, R = QR_Householder(A)

    assert np.allclose(Q, Q_expected)
    assert np.allclose(R, R_expected)


'''##### solve Tests #####'''


@Test
def solve_1():
    A = np.eye(5)
    b = np.arange(5)

    x = solve(A, b)

    assert np.allclose(x, b)


@Test
def solve_2():
    A = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 4.0, 9.0, 16.0],
        [1.0, 8.0, 27.0, 64.0],
        [1.0, 16.0, 81.0, 256.0]
    ])
    b = np.array([2.0, 10.0, 44.0, 190.0])
    x_expected = np.array([-1.0, 1.0, -1.0, 1.0])

    x = solve(A, b)

    assert np.allclose(x_expected, x)


@Test
def solve_3():
    A = np.array([
        [2.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        [4.0, 1.0, 2.0]
    ])
    b = np.array([1.0, 0.0, 1.0])
    x_expected = np.array([0, -3, 2])

    x = solve(A, b)

    assert np.allclose(x_expected, x)


@Test
def solve_4():
    A = np.array([
        [0.835, 0.667],
        [0.333, 0.266]
    ])
    b = np.array([0.168, 0.067])
    x_expected = np.array([1, -1])

    x = solve(A, b)

    assert np.allclose(x_expected, x)


'''##### power_iteration Tests #####'''


@Test
def power_iteration_1():
    A = np.diag([1, 2, -0.5])
    lamb_expected = 2
    v_expected = np.array([0, 1, 0])

    lamb, v = power_iteration(A)

    assert np.isclose(lamb, lamb_expected)
    assert np.allclose(v, v_expected)


@Test
def power_iteration_2():
    A = np.array([
        [1, -2],
        [-2, -3]
    ])
    lamb_expected = -1 - 2 * np.sqrt(2)
    v_expected = np.array([(np.sqrt(2) - 1) / np.sqrt((np.sqrt(2) - 1) ** 2 + 1), 1 / np.sqrt((np.sqrt(2) - 1) ** 2 + 1)])

    lamb, v = power_iteration(A)

    assert np.allclose(lamb, lamb_expected)
    assert np.allclose(v, v_expected)


@Test
def power_iteration_3():
    A = np.array([
        [2, 0, 2],
        [0, 0, 0],
        [2, 0, 2]
    ])
    lamb_exp = 4
    v_exp = np.array([1, 0, 1]) / np.sqrt(2)

    lamb, v = power_iteration(A)

    assert np.isclose(lamb_exp, lamb)
    assert np.allclose(v_exp, v)


'''##### eigen Tests #####'''


@Test
def eigen_1():
    A = np.diag([1, 2, -0.5])
    lamb_expected = np.array([2, 1, -0.5])
    v_expected = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    lamb, v = eigen(A)

    assert np.allclose(lamb, lamb_expected)
    assert np.allclose(v, v_expected)


@Test
def eigen_2():
    A = np.array([
        [1, -2],
        [-2, -3]
    ])
    lamb_expected = np.array([-1 - 2 * np.sqrt(2), -1 + 2 * np.sqrt(2)])
    v_expected = np.array([
        [(np.sqrt(2) - 1) / np.sqrt((np.sqrt(2) - 1) ** 2 + 1), (np.sqrt(2) + 1) / np.sqrt((np.sqrt(2) + 1) ** 2 + 1)],
        [1 / np.sqrt((np.sqrt(2) - 1) ** 2 + 1), -1 / np.sqrt((np.sqrt(2) + 1) ** 2 + 1)]
    ])

    lamb, v = eigen(A)

    assert np.allclose(lamb, lamb_expected)
    assert np.allclose(v, v_expected)


'''##### singular_values Tests #####'''


@Test
def singular_values_1():
    A = np.diag([2, 1])
    U_exp = np.eye(2)
    Sigma_exp = A
    V_exp = U_exp

    U, Sigma, V = singular_values(A)

    assert np.allclose(U_exp, U)
    assert np.allclose(Sigma_exp, Sigma)
    assert np.allclose(V_exp, V)


'''##### jacobi Tests #####'''


@Test
def jacobi_1():
    A = np.eye(5)
    b = np.arange(5)

    x = jacobi(A, b)

    assert np.allclose(x, b)


@Test
def jacobi_2():
    A = np.array([
        [2, 1, 0],
        [0, 4, 0],
        [2, 2, 8]
    ])
    b = np.array([1, 3, 3])
    x_expected = np.array([1 / 8, 3 / 4, 5 / 32])

    x = jacobi(A, b)

    assert np.allclose(x_expected, x)


@Test
def jacobi_3():
    A = np.array([
        [2, 1, 0, 0, 0],
        [0, 4, 0, 1, 2],
        [0, 0, 3, 2, 0],
        [0, 0, 0, 5, 0],
        [0, 0, 0, 0, 3]
    ])
    b = np.array([1, 1, 5, 0, 1])
    x_expected = np.array([11 / 24, 1 / 12, 5 / 3, 0, 1 / 3])

    x = jacobi(A, b)

    assert np.allclose(x_expected, x)


'''##### gauss_siedel Tests #####'''


@Test
def gauss_siedel_1():
    A = np.eye(5)
    b = np.arange(5)

    x = gauss_siedel(A, b)

    assert np.allclose(x, b)


@Test
def gauss_siedel_2():
    A = np.array([
        [2, 1, 0],
        [0, 4, 0],
        [2, 2, 8]
    ])
    b = np.array([1, 3, 3])
    x_expected = np.array([1 / 8, 3 / 4, 5 / 32])

    x = gauss_siedel(A, b)

    assert np.allclose(x_expected, x)


@Test
def gauss_siedel_3():
    A = np.array([
        [2, 1, 0, 0, 0],
        [0, 4, 0, 1, 2],
        [0, 0, 3, 2, 0],
        [0, 0, 0, 5, 0],
        [0, 0, 0, 0, 3]
    ])
    b = np.array([1, 1, 5, 0, 1])
    x_expected = np.array([11 / 24, 1 / 12, 5 / 3, 0, 1 / 3])

    x = gauss_siedel(A, b)

    assert np.allclose(x_expected, x)


run_tests()
