import pyquil.quil as pq
import numpy as np
from pyquil.gates import *
from grove.phaseestimation.phase_estimation import *
from grove.qft.fourier import *

def unitary_operator(a, N):
    """
    Creates the operator U such that U|y> = |a * y> for all y < N.
    Applying phase estimation using this operator on the input state
    |1> is equivalent to modular exponentiation.
    """
    n = int(2**np.ceil(np.log2(N + 1)))
    U = np.zeros(shape=(n, n))
    for i in range(n):
        col = np.zeros(n)
        if i < N:
            col[a*i % N] = 1
        else:
            col[i] = 1
        U[i] = col
    return U.T



def is_unitary(matrix):
    rows, cols = matrix.shape
    if rows != cols:
        return False
    return np.allclose(np.eye(rows), matrix.dot(matrix.T.conj()))

def order(a, N):
    """
    Program that calculates order of f(x) = a^x (mod N)
    """
    U = unitary_operator(a, N)

    L = int(np.log2(len(U)))
    t = 2*L + 3

    register_1 = range(t)
    register_2 = range(t, t+L)

    p = pq.Program()
    p.inst(map(H, register_1))
    p.inst(X(register_2[0]))

    for i in register_1:

        if i > 0:
            U = np.dot(U, U)
        cU = controlled(U)
        name = "CONTROLLED-U{0}".format(2 ** i)
        # define the gate
        p.defgate(name, cU)
        # apply it
        p.inst((name, i) + tuple(register_2))


    p += inverse_qft(register_1)


    for i in register_1:
        p.measure(i,  i)

    return p

def gcd(a, b):
    if b > a:
        return gcd(b, a)
    if b == 0:
        return a
    return gcd(b, a % b)

def test_exponentiation():
    # Test modular exponentiation on f(x) = 7^x (mod 15)
    U = unitary_operator(7, 15)
    assert is_unitary(U), "U must be unitary"

    # Matrices for repeated squaring
    U_2 = U.dot(U)
    U_4 = U_2.dot(U_2)
    U_8 = U_4.dot(U_4)

    one = np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    four = np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
    seven = np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
    thirteen = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])

    assert np.array_equal(U_2.dot(one), four), "7^3 (mod 15) = 4"
    assert np.array_equal(U_8.dot(U_2).dot(U).dot(one), thirteen), "7^11 (mod 15) = 13"
    assert np.array_equal(U_8.dot(U).dot(one), seven), "7^9 (mod 15) = 7"
    assert np.array_equal(U_8.dot(U_4).dot(U_2).dot(U).dot(one), thirteen), "7^15 (mod 15) = 13"



if __name__ == "__main__":
    import pyquil.forest as forest
    qvm = forest.Connection()
    classical_regs = [0]*10

    # let's find the order of f(x) = 7^x (mod 15)
    p = order(7, 15)

    print p

    print qvm.run(p, classical_regs)

    #test_exponentiation()





