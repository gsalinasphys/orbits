import mpmath as mp
import numpy as np
from plotly import express as px
from scipy.spatial.transform import Rotation

from orbit2d import get_2dtrajectory

mp.mp.dps = 50


def _rotation(rin: np.ndarray, vin: np.ndarray) -> np.ndarray:
    v0 = np.linalg.norm(vin)
    b = np.linalg.norm(np.cross(rin, vin)) / v0
    d0 = np.sqrt(np.linalg.norm(rin)**2 - b**2)

    r2d, v2d = np.array([d0, b, 0.]), np.array([-v0, 0., 0.])
    rotation, _ = Rotation.align_vectors(np.array([rin, vin]), np.array([r2d, v2d]))

    return (d0, b, v0), rotation

def get_3dtrajectory(rin: np.ndarray, vin: np.ndarray, k: float) -> tuple:
    (d0, b, v0), rotation = _rotation(rin, vin)
    d0, b, v0, k = mp.mpf(d0), mp.mpf(b), mp.mpf(v0), mp.mpf(k)
    (thetai, thetaf), rs, ts = get_2dtrajectory(d0, b, v0, k)

    x = np.vectorize(lambda theta: rs(theta) * mp.cos(theta))
    y = np.vectorize(lambda theta: rs(theta) * mp.sin(theta))

    r2d = lambda thetas: np.array([x(thetas), y(thetas), np.zeros(len(thetas))], dtype=float).T
    
    return (thetai, thetaf), \
        lambda thetas: rotation.apply(r2d(thetas)), \
        ts

def plot_3dtraj(rin: np.ndarray, vin: np.ndarray, k: float, n_points: int = 100, filepath: str = ''):
    (thetai, thetaf), xs, _ = get_3dtrajectory(rin, vin, k)
    thetas = mp.linspace(thetai, thetaf, n_points)

    fig = px.line_3d(x=xs(thetas).T[0], y=xs(thetas).T[1], z=xs(thetas).T[2])
    if filepath:
        fig.write_html(filepath)
    fig.show()
