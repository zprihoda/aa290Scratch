"""
Implementing Examples from:
http://www1.aucegypt.edu/faculty/mharafa/MENG%20475/Finite%20Element%20Vibration%20Analysis%20Fall%202010.pdf

Another useful reference:
https://engineering.purdue.edu/~deadams/ME563/lecture1510.pdf

TODO: Beam example natural frequency not converging...
    Not Sure why... Fix this!
"""

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

def getRodMatrices(n,E,A,L,rho):

    dl = L/n

    K = np.diag([2.]*n) + np.diag([-1.]*(n-1), k=1) + np.diag([-1.]*(n-1), k=-1)
    K[0,0] = 1
    K[-1,-1] = 1
    K *= E*A/dl

    C = 0.01*K

    M = np.diag([2/3.]*n) + np.diag([1/6.]*(n-1), k=1) + np.diag([1/6.]*(n-1), k=-1)
    M[0,0] = 1/3.
    M[-1,-1] = 1/3.
    M *= rho*A*dl

    return K, C, M

def getRodNaturalFrequencies(K,M,L,plot=False):
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

    if plot:
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


def getBeamMatrices(n,E,w,h,L,rho):
    A = w*h
    dl = L/(n+1)
    dm = dl*A*rho
    I = 1./12 * dm/2 * (dl**2 + h**2)
    # I = 1./12 * dm/2 * (dl**2)

    K_e = E*I/dl**3 * np.array([
            [   12,    6*dl,   -12,    6*dl],
            [ 6*dl, 4*dl**2, -6*dl, 2*dl**2],
            [  -12,   -6*dl,    12,   -6*dl],
            [ 6*dl, 2*dl**2, -6*dl, 4*dl**2]
            ])

    M_e = rho*A*dl/420 * np.array([
            [    156,    22*dl,     54,   -13*dl],
            [  22*dl,  4*dl**2,  13*dl, -3*dl**2],
            [     54,    13*dl,    156,   -22*dl],
            [ -13*dl, -3*dl**2, -22*dl,  4*dl**2]
            ])

    C_e = 0.01*K_e

    K_tot = np.zeros([2*(n+1),2*(n+1)])
    M_tot = np.zeros([2*(n+1),2*(n+1)])
    for i in range(n):
        K_tot[2*i:2*i+4,2*i:2*i+4] += K_e
        M_tot[2*i:2*i+4,2*i:2*i+4] += M_e

    C_tot = 0.01*K_tot

    return K_tot, C_tot, M_tot


def getBeamNaturalFrequencies(K,M,L,plot=False):
    """
    obtain natural frequencies (eigenvalue problem)
    M @ ddu + K @ u = 0  (free response)
    assume u = a*sin(2*pi*w*t)  (a,w are constants, w is scalar, a is in R^n)
    --> K @ a = w^2 * M @ a
    --> (M^-1 @ K)@a = w^2 @ a
    a are the vibration modes
    w are the natural frequencies
    """

    n = int(M.shape[0]/2)

    A = np.dot(npl.inv(M[2:,2:]), K[2:,2:])   # fixed-free rod
    lmbda,v = npl.eig(A)
    sort_idx = lmbda.argsort()

    w = np.sqrt(lmbda[sort_idx[:min(3,n-1)]])
    print(w/(2*np.pi))

    if plot:
        plt.figure()
        for i in sort_idx[:min(3,n-1)]:
            plt.plot(np.linspace(0,L,n),np.append([0],v[::2,i]))
        plt.title("Mode Shapes")
        plt.xlabel("Axial Distance [m]")
        plt.ylabel("Normalized Axial Displacement")
        plt.legend(["Mode 1", "Mode 2", "Mode 3"])
        plt.grid()
        plt.xlim([0,L])
        plt.show()



def main():

    ## rod example
    # E = 80e9
    # A = 0.01
    # L = 8.
    # rho = 7800.
    # rho = 8000.

    # n = 40

    # K,C,M = getRodMatrices(n,E,A,L,rho)
    # getRodNaturalFrequencies(K,M,L,plot=True)


    ## Beam Example
    E = 80e9
    w = 20./1000
    h = 0.4/1000
    A = w*h
    L = 0.2
    rho = 2700

    n = 5

    K,C,M = getBeamMatrices(n,E,w,h,L,rho)
    getBeamNaturalFrequencies(K,M,L,plot=False)

if __name__ == "__main__":
    main()
