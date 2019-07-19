from boomDynamics import simulateTorsion

def dynamicsStep(arm, u, dt):
    """
    Inputs:
        arm: current robot arm object
        u    : control commands
            u = [M1, ... , M6, F_r]
            M1-M6 = Command moments for each rotational motor
            F_r = Force in radial direction
        dt  : time step
    """

    # Determine Command Forces
    M1_cmd = u[0]
    M2_cmd = u[1]

    # TODO: Determine external forces
    M1_ext = 0
    M2_ext = 0

    M1 = M1_cmd + M1_ext
    M2 = M2_cmd + M2_ext

    # Torsion simulation
    arm = simulateTorsion(arm, M1, M2, dt)


    # Bending Simulation
    # TODO: Implement

    # Extension Simulation
    # TODO: Implement
    # F_cmd = u[-1]
    # F_ext = 0   # TODO
    # Fz = F_cmd + F_ext
    # arm.state = simulateExtension(arm.state, Fz)    # NOTE: simulate extension also remaps all other states to the new finite element model

    return arm
