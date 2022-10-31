import mpmath as mp
from scipy.optimize import root_scalar

from orbit2d import get_2dtrajectory
from orbit3d import _twodim_params

mp.mp.dps = 50

def enter_ns(d0: float, b: float, v0: float, k: float, radius: float):
    (thetamin, thetamax), r, rdot, thetadot, t = get_2dtrajectory(d0, b, v0, k)
    theta0 = 0.5 * (thetamin+thetamax)

    try:
        root_found = root_scalar(lambda thetas: mp.norm(rs(thetas))-radius, bracket=(thetamin, theta0))
        if root_found.converged:
            return root_found.root, r(root_found.root), rdot(root_found.root), thetadot(root_found.root), t(root_found.root)
        else:
            return None
    except ValueError as e:
        print(e)
        return None
