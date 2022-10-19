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

    in_or_out = np.sign(np.dot(rin, vin))
    v2d = np.array([in_or_out*v0, 0., 0.])
    possible_r2d = np.array([d0, b, 0.]), np.array([d0, -b, 0.])
    (rotation1, _), (rotation2, _) = (Rotation.align_vectors(np.array([rin, vin]), np.array([r2d, v2d])) for r2d in possible_r2d)

    reconstructed_r2d = np.array([rotation1.apply(rin), rotation2.apply(rin)])
    which_rotation = np.where(np.isclose(abs(reconstructed_r2d[:, 1]), b))[0][0]
    rotation = np.array([rotation1, rotation2])[which_rotation]

    return (d0, (-1)**which_rotation*b, in_or_out*v0), rotation

def get_3dtrajectory(rin: np.ndarray, vin: np.ndarray, k: float) -> tuple:
    (d0, b, v0), rotation = _rotation(rin, vin)
    theta0, rs, rdots, thetadots, ts = get_2dtrajectory(d0, b, v0, k)

    x2d = np.vectorize(lambda theta: rs(theta) * mp.cos(theta))
    y2d = np.vectorize(lambda theta: rs(theta) * mp.sin(theta))
    r2d = lambda thetas: np.array([x2d(thetas), y2d(thetas), np.zeros_like(thetas)], dtype=float).T
    r3d = lambda thetas: rotation.apply(r2d(thetas))

    vx2d = np.vectorize(lambda theta: rdots(theta)*mp.cos(theta) - rs(theta)*thetadots(theta)*mp.sin(theta))
    vy2d = np.vectorize(lambda theta: rdots(theta)*mp.sin(theta) + rs(theta)*thetadots(theta)*mp.cos(theta))
    v2d = lambda thetas: np.array([vx2d(thetas), vy2d(thetas), np.zeros_like(thetas)], dtype=float).T
    v3d = lambda thetas: rotation.apply(v2d(thetas))
    
    return theta0, r3d, v3d, ts

def plot_3dtraj(rin: np.ndarray, vin: np.ndarray, k: float, n_points: int = 100, filepath: str = ''):
    theta0, xs, *_ = get_3dtrajectory(rin, vin, k)
    thetas = mp.linspace(0., 2*theta0, n_points)

    fig = px.line_3d(x=xs(thetas).T[0], y=xs(thetas).T[1], z=xs(thetas).T[2])
    if filepath:
        fig.write_html(filepath)
    fig.show()
