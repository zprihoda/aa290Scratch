import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import matplotlib.pyplot as plt


class ReducedDynamics():
    def __init__(self, full_dyn, n_red):
        Af, Bf = full_dyn.A, full_dyn.B
        A_red, B_red, C_red = reduceDynamics(full_dyn, n_red)

        self.A = A_red
        self.B = B_red
        self.C = C_red

def reduceDynamics(dyn, n_red):
    """
    See pages 78 and 209-211 of Approximation of Large-Scale Dynamical Systems by Antoulas
    """

    n = dyn.A.shape[0]
    m = dyn.B.shape[1]

    # stable/unstable seperation
    A_s, B_s, C_s, V, l = stabSep(dyn)

    A_n = A_s[:l,:l]
    A_c = A_s[:l,l:]
    A_p = A_s[l:,l:]

    B_n = B_s[:l,:]
    B_p = B_s[l:,:]

    C_n = C_s[:,:l]
    C_p = C_s[:,l:]

    # Form new seperated system:
    # dX_n = A_n @ x_n + B_tilde @ u_tilde
    # y = C_n @ X_n + D_tilde @ u_tilde
    B_tilde = np.hstack([B_n, A_c])
    n_tilde = n_red - (n-l)

    # apply balanced reduction
    A_nr, B_tilde_r, C_nr = balancedReduction(A_n, B_tilde, C_n, n_tilde)
    l_r = A_nr.shape[0]
    B_nr = B_tilde_r[:,:m]
    A_cr = B_tilde_r[:,m:]

    # recombine system
    A_r = np.vstack([
        np.hstack([A_nr, A_cr]),
        np.hstack([np.zeros([n-l,l_r]),A_p])
        ])
    B_r = np.vstack([B_nr, B_p])
    C_r = np.hstack([C_nr,C_p])

    # TODO: determine what transformations needed for x_nr and x_p terms
    #   May include both the reduction similarity transform and the stabSep transforms...
    #   be very careful, maybe check with Joe...

    return A_r, B_r, C_r

def balancedReduction(Af, Bf, Cf, n_red, debug=0):
    # get P and Q (infinite grammians)
    if Cf is None:
        Cf = np.eye(Af.shape[0])

    P = spl.solve_lyapunov(Af, -Bf @ Bf.conj().T)
    Q = spl.solve_lyapunov(Af.conj().T, -Cf.conj().T @ Cf)

    # Obtain U, K, Sigma
    # note: Books wants P = U U^*, scipy returns P = U^* U
    #    therefore, we want U^* to be upper triangular -> U = lower traingular
    U = spl.cholesky(P, lower=True)

    lmbda,K = npl.eig(U.conj().T @ Q @ U)

    idx = lmbda.argsort()[::-1] # sort the eigenvalue outputs
    lmbda = lmbda[idx]
    K = K[:,idx]

    Sigma = np.diag(np.sqrt(lmbda))
    Sigma_nhalf = np.diag(1/lmbda**(1./4))

    # Obtain balanced similarity transform matrices
    T = np.sqrt(Sigma) @ K.conj().T @ npl.inv(U)
    T_inv = U @ K @ Sigma_nhalf

    if debug:
        # test matrices
        tmp1 = T @ P @ T.conj().T
        tmp2 = T_inv.conj().T @ Q @ T_inv
        print("Maxmimum Matrix Error: ",np.max(np.abs(tmp1-tmp2)))

        plt.plot(lmbda,'.')
        plt.xlabel('i')
        plt.ylabel(r'$\sigma_i$')
        plt.title('Hankel Singular Values')
        plt.grid()
        plt.show()

    # Obtain reduced transform matrices
    T = T[:n_red,:]
    T_inv = T_inv[:,:n_red]

    # obtain reduced matrices
    A_r = T @ Af @ T_inv
    B_r = T @ Bf
    C_r = Cf @ T_inv

    return A_r, B_r, C_r, T, T_inv

def stabSep(dyn):
    """
    See [1] for implementation details
    References:
    [1] : https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=78129&tag=1
    """

    A_s, V, l = spl.schur(dyn.A, sort='lhp')
    B_s = V.T @ dyn.B
    C_s = dyn.C @ V


    min_lmbda = np.min(npl.eigvals(A_s))
    threshold = min_lmbda/1e5
    l = sum(npl.eigvals(A_s) < threshold)    # remove close to unstable elements

    return A_s, B_s, C_s, V, l

if __name__ == "__main__":
    from dynamics import Dynamics

    # arguments
    n = 10
    n_r = 5     # reduced

    # Generate Dynamics Matrices
    while True:
        tmp = np.random.randn(n,n)
        A = -np.dot(tmp,tmp.T)  # guaranteed to be n.s.d.
        lmbda = npl.eig(A)[0]

        if all(lmbda < 0):  # check that we are n.d.
            break

    B = np.vstack([[1,0],
                   np.zeros([n-2,2]),
                   [0,1]])

    dyn_f = Dynamics(A,B)

    # obtain reduced Dynamics
    A_r, B_r, C_r, T, T_inv = balancedReduction(dyn_f.A, dyn_f.B, dyn_f.C, n_red=n_r)

    # Setup Simulation parameters
    x0 = np.zeros(n)
    x0[-1] = 5

    tf = 100
    dt = 0.01
    t_arr = np.arange(0, tf, dt)

    # simulate full dynamics
    x_arr = np.zeros([n,len(t_arr)])
    x = x0
    for i,t in enumerate(t_arr):
        dx = A@x
        x = x + dx*dt
        x_arr[:,i] = x

    # simulate reduced dynamics
    xr_arr = np.zeros([n,len(t_arr)])
    x = x0
    for i,t in enumerate(t_arr):
        x_r = T @ x
        dx_r = A_r@x_r
        x_r = x_r + dx_r*dt
        x = T_inv @ x_r
        xr_arr[:,i] = x

    # print results
    print('Maximum State Error:', np.max(np.abs(x_arr-xr_arr)))

    # plot results
    plt.plot(t_arr, x_arr[-1,:])
    plt.plot(t_arr, xr_arr[-1,:])
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Full vs Reduced Dynamics')
    plt.legend(['Full: n={:}'.format(n), 'Reduced: n={:}'.format(n_r)])
    plt.show()

