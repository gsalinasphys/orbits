import mpmath as mp
import numpy as np
from scipy.optimize import root_scalar

from orbit3d import get_3dtrajectory

mp.mp.dps = 50

def enter_ns(rin: np.ndarray, vin: np.ndarray, k: float, radius: float):
    theta0, rs, vs, ts = get_3dtrajectory(rin, vin, k)

    try:
        root_found = root_scalar(lambda thetas: np.linalg.norm(rs(thetas))-radius, bracket=(0., theta0))
        if root_found.converged:
            return root_found.root, rs(root_found.root), vs(root_found.root), ts(root_found.root)
        else:
            return None
    except ValueError:
        return None
