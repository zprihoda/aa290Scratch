import numpy as np
import numpy.linalg as npl

import matplotlib.pyplot as plt

"""
Test dynamics to ensure they converge as n -> inf
"""

def getTorsionMatrices(n,G,A,L,rho,I):
    dl = L/n

    K = np.diag([2.]*n) + np.diag([-1.]*(n-1), k=1) + np.diag([-1.]*(n-1), k=-1)
    K[0,0] = 1
    K[-1,-1] = 1
    K *= G*A/dl

    C = 0.01*K

    M = np.diag([2/3.]*n) + np.diag([1/6.]*(n-1), k=1) + np.diag([1/6.]*(n-1), k=-1)
    M[0,0] = 1/3.
    M[-1,-1] = 1/3.
    M *= I/n

    return K, C, M

def getDeflectionMatrices(n,E,A,L,rho,I):
    dl = L/n
    m = rho*A*L
    dm = m/n

    r = 1.5/100

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

    K_tot = np.zeros([2*n,2*n])
    M_tot = np.zeros([2*n,2*n])
    for i in range(n-1):
        K_tot[2*i:2*i+4,2*i:2*i+4] += K_e
        M_tot[2*i:2*i+4,2*i:2*i+4] += M_e

    C_tot = 0.01*K_tot

    return K_tot, C_tot, M_tot

def getNaturalFrequencies(K,M,start_idx=0, end_idx=None):
    """Obtain natural frequencies from eigenvalue problem"""

    if end_idx is None:
        M = M[start_idx:,start_idx:]
        K = K[start_idx:,start_idx:]
    else:
        M = M[start_idx:-end_idx,start_idx:-end_idx]
        K = K[start_idx:-end_idx,start_idx:-end_idx]
    n = M.shape[0]

    A = np.dot(npl.inv(M), K)   # fixed-free rod
    lmbda,v = npl.eig(A)
    sort_idx = lmbda.argsort()[:min(3,n)]

    w = np.sqrt(lmbda[sort_idx])

    if end_idx is None:
        end_idx = 0

    v = [np.hstack([[0]*start_idx,v[:,i],[0]*end_idx]) for i in sort_idx]

    return w,v


def main():
    # test torsion dynamics
    E = 50.            # K_lat
    G = 100./1000        # K_rot
    rho = 1000.

    r = 1.5/100
    delta = 1./1000

    L = 1.

    A = 2*np.pi*r*delta
    m = rho*A*L

    I_zz = m*r**2

    n_list = [5,10,15,20,30,40,50,75,100,125,150,175,200]
    wt_arr = []
    wd_arr = []
    for n in n_list:
        dl = L/n
        I_xx = 1./12 * m * (6*r**2 + L**2)

        # K,C,M = getTorsionMatrices(n,G,A,L,rho,I_zz)
        # wt,vt = getNaturalFrequencies(K,M,start_idx=1)
        # wt_arr.append(wt)

        K,C,M, = getDeflectionMatrices(n,E,A,L,rho,I_xx)
        wd,vd = getNaturalFrequencies(K,M,start_idx=2)
        wd_arr.append(wd)

    # wt_arr = np.array(wt_arr)
    # vt_arr = np.array(vt)

    wd_arr = np.array(wd_arr)
    vd_arr = np.array(vd)[:,::2]

    # fig,axes = plt.subplots(2,1)
    # axes[0].plot(np.linspace(0,L,n),vt_arr.T)
    # axes[0].legend(['Mode {}'.format(idx) for idx in np.arange(len(n_list))+1])
    # axes[0].set_xlabel(r'$x/L$')
    # axes[0].set_ylabel(r'$\theta$')
    # axes[0].grid()
    # axes[0].set_title('Torsional Modes and Natural Frequencies')
    # axes[1].semilogy(n_list,wt_arr,'-o')
    # axes[1].set_xlabel(r'$n$')
    # axes[1].set_ylabel(r'$w_n$')
    # axes[1].grid()
    # plt.tight_layout()

    fig,axes = plt.subplots(2,1)
    axes[0].plot(np.linspace(0,L,n),vd_arr.T)
    axes[0].legend(['Mode {}'.format(n) for n in np.arange(len(n_list))+1])
    axes[0].set_xlabel(r'$x/L$')
    axes[0].set_ylabel(r'$u$')
    axes[0].grid()
    axes[0].set_title('Deflection Modes and Natural Frequencies')
    axes[1].semilogy(n_list,wd_arr,'-o')
    axes[1].set_xlabel(r'$n$')
    axes[1].set_ylabel(r'$w_n$')
    axes[1].grid()
    plt.tight_layout()

    print('Theoretical Wn = ', 1.875**2 * np.sqrt(E*I_xx/(rho*A*L**4)))

    plt.show()


if __name__ == "__main__":
    main()
