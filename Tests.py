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


run_tests()
