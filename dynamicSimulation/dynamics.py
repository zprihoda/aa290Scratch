"""
-------------------
Reference Material:
-------------------

Torsional:
Reference equations on torsional and bending vibrations for beam elements:
    http://www.wind.civil.aau.dk/lecture/7sem/notes/Lecture9.pdf
Duke Lectures on StructDyn
    http://people.duke.edu/~hpgavin/cee541/
Universite Libre de Bruxelles Course Notes:
    https://scmero.ulb.ac.be/Teaching/Courses/MECA-H-303/MECA-H-303-Lectures.pdf
Numerial Solution to Wave equation:
    http://www-users.math.umn.edu/~olver/num_/lnp.pdf

Bending / General FE methods:
http://www.solid.iei.liu.se/Education/TMHL08/Lectures/Lecture__8.pdf
http://homepages.cae.wisc.edu/~suresh/ME964Website/M964Notes/Notes/introfem.pdf
DYNAMIC FINITE ELEMENT METHODS - Lecture notes for SD2450 - Biomechanics and Neuronics

Finite Element for Vibrations:
http://www1.aucegypt.edu/faculty/mharafa/MENG%20475/Finite%20Element%20Vibration%20Analysis%20Fall%202010.pdf

----
TODO
----
How to handle state when we are extending
    ie. while extending how does our state (x) change
    We may be able to linearly interpolate to find new state:
        x_new_pos = L/n * np.arange... (?)

Implement dynamics for bending

Implement Dynamics class
    Calculate instance variables once:
        eg. Relation Matrix
    Store structural properties
    Cleanup inputs and outputs

Implement ability to fix one of the ends
    Need to look into how fixing an end affects our dynamics model
"""

import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import structuralProperties as structProp

class Dynamics():
    def __init__(self, arm, dt, n=None, noise_bending=None, noise_torsion=None):

        if n is None:
            self.n = arm.state.n
        else:
            self.n = n

        self.structProps = arm.structProps
        self.dt = dt
        self.L = arm.r

        self.A_torsion, self.B_torsion = self.getABTorsion()
        self.noise_torsion = noise_torsion

        self.A_bending, self.B_bending = self.getABDeflection()
        self.noise_bending = noise_bending


    def getTorsionMatrices(self):
        L = self.L
        n = self.n
        dl = L/n

        r = self.structProps['radius']
        rho = self.structProps['density']
        delta = self.structProps['thickness']
        G = self.structProps['k_rot']

        A = 2*np.pi*r*delta

        m = rho*A*L
        I = m*r**2

        K = np.diag([2.]*n) + np.diag([-1.]*(n-1), k=1) + np.diag([-1.]*(n-1), k=-1)
        K[0,0] = 1
        K[-1,-1] = 1
        K *= G*A/dl

        C = 0.01*K
        # C = 0.0*K

        M = np.diag([2/3.]*n) + np.diag([1/6.]*(n-1), k=1) + np.diag([1/6.]*(n-1), k=-1)
        M[0,0] = 1/3.
        M[-1,-1] = 1/3.
        M *= I/n

        return K, C, M

    def getABTorsion(self):
        dt = self.dt
        n = self.n

        # get torsion matrices
        K,C,M = self.getTorsionMatrices()
        M_inv = npl.inv(M)

        # form continuous dynamics
        A = np.vstack([np.hstack([np.zeros([n,n]), np.eye(n)]),
                       np.hstack([-M_inv@K, -M_inv@C])])
        B_tmp = np.vstack([np.array([1,0]),
                           np.zeros([n-2,2]),
                           np.array([0,1])
                          ])
        B = np.vstack([np.zeros([n,2]),
                       M_inv@B_tmp])

        # discretize dynamics
        t_arr = np.linspace(0,1,10)*dt
        y_arr = np.array([spl.expm(t*A) for t in t_arr])
        tmp = np.sum(y_arr,axis=0)*(t_arr[1]-t_arr[0])

        A_d = spl.expm(A*dt)
        B_d = np.dot(tmp,B)

        return A_d, B_d

    def getDeflectionMatrices(self):

        L = self.L
        n = self.n
        dl = L/n

        r = self.structProps['radius']
        rho = self.structProps['density']
        delta = self.structProps['thickness']
        E = self.structProps['k_lat']

        A = 2*np.pi*r*delta

        m = rho*A*L
        dm = m/n

        I = 1./12 * m * (6*r**2 + dl**2)

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

    def getABDeflection(self):
        # obtain properties from arm
        L = self.L

        n = self.n
        dl = L/n

        r = self.structProps['radius']
        rho = self.structProps['density']
        delta = self.structProps['thickness']
        E = self.structProps['k_lat']

        area = 2*np.pi*r*delta

        # get Bending matrices
        K,C,M = self.getDeflectionMatrices()
        M_inv = npl.inv(M)

        # obtain continous dynamics matrices
        A = np.vstack([np.hstack([np.zeros([2*n,2*n]), np.eye(2*n)]),
                       np.hstack([-M_inv@K, -M_inv@C])])
        B_tmp = np.vstack([np.array([1,0,0,0]),
                           np.array([0,1,0,0]),
                           np.zeros([2*(n-2),4]),
                           np.array([0,0,1,0]),
                           np.array([0,0,0,1])
                          ])
        B = np.vstack([np.zeros([2*n,4]),
                       M_inv@B_tmp])

        # discretize dynamics
        t_arr = np.linspace(0,1,10)*dt
        y_arr = np.array([spl.expm(t*A) for t in t_arr])
        tmp = np.sum(y_arr,axis=0)*(t_arr[1]-t_arr[0])

        A_d = spl.expm(A*dt)
        B_d = np.dot(tmp,B)

        return A_d, B_d


    def simulateTorsion(self, arm, M1, M2):
        """
        Simulate torsional dynamics

        Arguments:
            arm   : Arm object
            M1    : Applied moment at start of boom
            M2    : Applied moment at end of boom

        Output:
            Updated arm object
        """

        if self.n != arm.state.n:
            raise ValueError("Arm and dynamics must have the same number of elements")
        n = self.n
        noise = self.noise_torsion

        # get dynamics and control matrices
        A_d = self.A_torsion
        B_d = self.B_torsion

        # setup state
        theta = arm.state.rot_z
        w = arm.state.rate_z
        X = np.append(theta, w)

        # Update step
        u = np.array([M1,M2])
        X_new = np.dot(A_d,X) + np.dot(B_d,u)

        if noise is not None:
            mean = np.zeros(2*n)
            cov = np.diag([noise[0]]*n + [noise[1]]*n)
            X_new += np.random.multivariate_normal(mean,cov)

        theta_new = X_new[0:n]
        w_new = X_new[n:]

        arm.state.rot_z = theta_new
        arm.state.rate_z = w_new

        return arm

    def simulateBending(self, arm, F1, F2, M1, M2, bc_start=0, bc_end=0, noise=None):
        """
        Simulate torsional dynamics

        Arguments:
            arm   : Arm object
            F1    : Applied force at start of boom
            F2    : Applied force at end of boom
            M1    : Applied moment at start of boom
            M2    : Applied moment at end of boom
            dt    : Time step interval
            bc_start, bc_end: boundary conditions at the start and end of boom
                0: No boundary conditions (default)
                1: fixed deflection, free rotation (pin joint)
                2: fixed deflection, fixed rotation
            noise : (optional) var of noise to inject into dynamics
                    [sigma_delta^2, sigma_theta^2, sigma_delta_dot^2, sigma_w^2]
        Output:
            Updated arm object
        """

        if self.n != arm.state.n:
            raise ValueError("Arm and dynamics must have the same number of elements")
        n = self.n
        noise = self.noise_bending

        # get dynamics and control matrices
        A_d = self.A_bending
        B_d = self.B_bending

        # # setup state
        delta = arm.state.def_lat
        ddelta = arm.state.rate_lat
        X = np.append(delta, ddelta)

        # handle boundary conditions
        idx = range(2*n)
        idx_rm = []
        if bc_start == 1:
            idx_rm.extend([0])
        elif bc_start == 2:
            idx_rm.extend([0, 1])
        if bc_end == 1:
            idx_rm.extend([2*n-2])
        elif bc_end == 2:
            idx_rm.extend([2*n-2, 2*n-1])

        idx_keep = np.array(list(set(idx)-set(idx_rm)))
        idx_rm = np.array(idx_rm)
        idx_keep2 = np.append(idx_keep, idx_keep+2*n)

        A_d = A_d[idx_keep2[:,None],idx_keep2]
        B_d = B_d[idx_keep2,:]
        X = X[idx_keep2]

        # Update step
        u = np.array([F1, M1, F2, M2])
        X_new = np.dot(A_d,X) + np.dot(B_d,u)

        if noise is not None:
            n_keep = len(idx_keep)
            mean = np.zeros(4*n)[idx_keep2]
            tmp = np.array([noise[0], noise[1]]*n + [noise[2], noise[3]]*n)
            tmp = tmp[idx_keep2]
            cov = np.diag(tmp)
            X_new += np.random.multivariate_normal(mean,cov)

        delta_new = X_new[:len(X_new)//2]
        ddelta_new = X_new[len(X_new)//2:]

        # assign outputs
        arm.state.def_lat = np.zeros(2*n)
        arm.state.def_lat[idx_keep] = delta_new
        arm.state.rate_lat = np.zeros(2*n)
        arm.state.rate_lat[idx_keep] = ddelta_new

        return arm


    def dynamicsStep(self, arm, u):
        """
        Main Dynamics Function called in simulation

        Inputs:
            arm: current robot arm object
            u    : control commands
                u = [M1, ... , M6, F_r]
                M1-M6 = Command moments for each rotational motor
                F_r = Force in radial direction
            dt  : time step
            noise_torsion : (optional) var of noise to inject into torsion dynamics
                    [sigma_theta^2, sigma_w^2]
            noise_bending : (optional) var of noise to inject into bending dynamics
                    [sigma_delta^2, sigma_theta^2, sigma_delta_dot^2, sigma_w^2]
        """

        # Torsion simulation
        M1_cmd = u['rot'][0]
        M2_cmd = u['rot'][1]

        # TODO: Implement External forces
        M1_ext = 0
        M2_ext = 0

        M1 = M1_cmd + M1_ext
        M2 = M2_cmd + M2_ext

        arm = self.simulateTorsion(arm, M1, M2)

        # Bending Simulation
        # TODO: implement external forces
        M1 = u['lat'][0]
        M2 = u['lat'][1]
        F1 = 0
        F2 = 0

        arm = self.simulateBending(arm, F1, F2, M1, M2, bc_start=1)

        # Extension Simulation
        # TODO: Implement
        # F_cmd = u[-1]
        # F_ext = 0   # TODO
        # Fz = F_cmd + F_ext
        # arm.state = simulateExtension(arm.state, Fz)    # NOTE: simulate extension also remaps all other states to the new finite element model

        return arm


if __name__ == "__main__":
    # simple test
    from robotArm import Arm
    import copy

    # simulation parameters
    tf = 10.0
    dt = 1e-2

    # setup arm and dynamics
    arm = Arm(1,np.zeros(6), num_fe=10)
    arm.state.def_lat[1::2] = 1.0
    arm.state.def_lat[0::2] = arm.state.pos_z
    arm.state.rot_z[0] = 0.1

    dynamics = Dynamics(arm, dt)

    # setup arrays
    t_arr = np.arange(0,tf,dt)
    u_arr = np.zeros([2,len(t_arr)])
    u_arr = [{'rot':[0,0],'lat':[0,0]} for t in t_arr]

    # simulate
    state_list = []
    state_list.append(copy.copy(arm.state))
    for i in range(len(t_arr)-1):

        arm = dynamics.dynamicsStep(arm,u_arr[i])
        state_list.append(copy.copy(arm.state))

    import plotResults
    plotResults.plotAll(state_list, u_arr, t_arr)
    plotResults.animateTorsion(state_list,t_arr)
    plotResults.animateBending(state_list,t_arr)

