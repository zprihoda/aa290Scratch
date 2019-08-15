"""
Implementing Examples from:
http://www1.aucegypt.edu/faculty/mharafa/MENG%20475/Finite%20Element%20Vibration%20Analysis%20Fall%202010.pdf
"""

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

def getRodMatrices(n,E,A,L,rho):
    K = np.diag([2.]*n) + np.diag([-1.]*(n-1), k=1) + np.diag([-1.]*(n-1), k=-1)
    K[0,0] = 1
    K[-1,-1] = 1
    K *= E*A/(L/n)

    C = 0.01*K

    M = np.diag([2/3.]*n) + np.diag([1/6.]*(n-1), k=1) + np.diag([1/6.]*(n-1), k=-1)
    M[0,0] = 1/3.
    M[-1,-1] = 1/3.
    M *= rho*A*(L/n)

    return K, C, M

def getNaturalFrequencies(K,M,L,plot=False):
    """
    obtain natural frequencies (eigenvalue problem)
    M @ ddu + K @ u = 0  (free response)
    assume u = a*sin(2*pi*w*t)  (a,w are constants, w is scalar, a is in R^n)
    --> K @ a = w^2 * M @ a
    --> (M^-1 @ K)@a = w^2 @ a
    a are the vibration modes
    w are the natural frequencies
    """

    n = M.shape[0]

    A = np.dot(npl.inv(M[1:,1:]), K[1:,1:])   # fixed-free rod
    lmbda,v = npl.eig(A)
    sort_idx = lmbda.argsort()

    w = np.sqrt(lmbda[sort_idx[:min(3,n-1)]])
    print(w/(2*np.pi))


    plt.figure()
    for i in sort_idx[:min(3,n-1)]:
        plt.plot(np.linspace(0,L,n),np.append([0],v[:,i]))
    plt.title("Mode Shapes")
    plt.xlabel("Axial Distance [m]")
    plt.ylabel("Normalized Axial Displacement")
    plt.legend(["Mode 1", "Mode 2", "Mode 3"])
    plt.grid()
    plt.xlim([0,L])
    plt.show()


def main():
    E = 80e9
    A = 0.01
    L = 8.
    rho = 7800.
    rho = 8000.

    n = 40

    K,C,M = getRodMatrices(n,E,A,L,rho)

    getNaturalFrequencies(K,M,L,plot=True)



if __name__ == "__main__":
    main()
