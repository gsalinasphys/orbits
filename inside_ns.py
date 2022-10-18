from scipy.optimize import root_scalar

from orbit2d import get_2dtrajectory


def enter_ns(d0: float, b: float, v0: float, k: float, radius: float):
    (thetai, thetaf), rs, ts = get_2dtrajectory(d0, b, v0, k)

    try:
        root_found = root_scalar(lambda thetas: rs(thetas)-radius, bracket=(thetai, (thetai+thetaf)/2.))
        if root_found.converged:
            return root_found.root
        else:
            return None
    except ValueError:
        return None
