import mpmath as mp
from scipy.optimize import root_scalar

from orbit2d import get_2dtrajectory

mp.mp.dps = 50

def enter_ns(r02d: mp.matrix, v02d: mp.matrix, k: float, radius: float, t0: float = 0.) -> tuple:
    (thetamin, _, thetamax), r, rdot, thetadot, t = get_2dtrajectory(r02d, v02d, k)
    thetaclose = (thetamin+thetamax) / 2.

    try:
        root_found = root_scalar(lambda theta: mp.norm(r(theta))-radius, bracket=(thetamin, thetaclose), xtol=1e-15, rtol=1e-15)
        if root_found.converged:
            return mp.mpf(root_found.root), r(root_found.root), rdot(root_found.root), thetadot(root_found.root), t(root_found.root)
        else:
            return None
    except ValueError as e:
        print(e)
        return None
