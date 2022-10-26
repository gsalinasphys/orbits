import mpmath as mp
import numpy as np
from plotly import express as px
from scipy.spatial.transform import Rotation

from orbit2d import get_2dtrajectory

mp.mp.dps = 50

def _twodim_params(rin: mp.matrix, vin: mp.matrix) -> tuple:
    v0 = mp.norm(vin)
    b = mp.norm(np.cross(rin, vin)) / v0
    d0 = mp.sqrt(mp.norm(rin)**2 - b**2)

    return d0, b, mp.sign(np.dot(rin, vin)) * v0

def _cross_matrix(v: mp.matrix) -> mp.matrix:
    return mp.matrix([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])

def _rotation_matrix(angle: float, unit_vector: mp.matrix) -> mp.matrix:
    return mp.cos(angle)*mp.eye(3) + mp.sin(angle)*_cross_matrix(unit_vector) + (1-mp.cos(angle))*unit_vector*unit_vector.T

def _rotate_r(rin: mp.matrix) -> mp.matrix:
    angle = -mp.atan(rin[2]/rin[1]) if rin[1] else mp.pi/2
    rotation = _rotation_matrix(angle, mp.matrix([1., 0, 0]))
    r2 = rotation * rin
    angle2 = -mp.atan(r2[1]/r2[0]) if r2[0] else mp.pi/2
    rotation2 = _rotation_matrix(angle2, mp.matrix([0., 0, 1]))
    r3 = rotation2 * r2
    factor = 1.
    if r3[0] < 0:
        factor *= _rotation_matrix(mp.pi, mp.matrix([0., 0, 1]))

    return factor * rotation2 * rotation

def _rotate(rin: mp.matrix, vin: mp.matrix) -> mp.matrix:
    rotation1 = _rotate_r(rin)
    r2, v2 = rotation1 * rin, rotation1 * vin
    angle = -mp.atan(v2[2]/v2[1]) if v2[1] else mp.pi/2
    rotation2 = _rotation_matrix(angle, mp.matrix([1., 0, 0]))
    v3 = rotation2 * v2
    angle2 = -mp.atan(v3[1]/v3[0]) if v3[0] else mp.pi/2
    rotation3 = _rotation_matrix(angle2, mp.matrix([0, 0, 1]))
    r3 = rotation3 * r2
    factor = 1.

    if r3[1] < 0:
        factor *= _rotation_matrix(mp.pi, mp.matrix([1., 0, 0]))

    return (factor * rotation3 * rotation2 * rotation1) ** (-1)

def get_3dtrajectory(rin: mp.matrix, vin: mp.matrix, k: float) -> tuple:
    rotation = _rotate(rin, vin)
    d0, b, v0 = _twodim_params(rin, vin)
    (thetamin, thetamax), rs, rdots, thetadots, ts = get_2dtrajectory(d0, b, v0, k)

    x2d = np.vectorize(lambda theta: rs(theta) * mp.cos(theta))
    y2d = np.vectorize(lambda theta: rs(theta) * mp.sin(theta))
    r2d = lambda thetas: np.array([x2d(thetas), y2d(thetas), mp.matrix([[0]*len(thetas)])]).T
    r3d = lambda thetas: np.array([rotation * mp.matrix(r2d_val) for r2d_val in r2d(thetas)])

    vx2d = np.vectorize(lambda theta: rdots(theta)*mp.cos(theta) - rs(theta)*thetadots(theta)*mp.sin(theta))
    vy2d = np.vectorize(lambda theta: rdots(theta)*mp.sin(theta) + rs(theta)*thetadots(theta)*mp.cos(theta))
    v2d = lambda thetas: np.array([vx2d(thetas), vy2d(thetas), mp.matrix([[0]*len(thetas)])]).T
    v3d = lambda thetas: np.array([rotation * mp.matrix(v2d_val) for v2d_val in v2d(thetas)])
    
    return (thetamin, thetamax), r3d, v3d, ts

def plot_3dtraj(rin: np.ndarray, vin: np.ndarray, k: float, n_points: int = 10_000, filepath: str = ''):
    (thetamin, thetamax), r3ds, *_ = get_3dtrajectory(rin, vin, k)
    thetas = mp.linspace(thetamin, thetamax, n_points)[1:-1]

    xs, ys, zs = r3ds(thetas).T

    fig = px.line_3d(x=np.array(xs, dtype=float), y=np.array(ys, dtype=float), z=np.array(zs, dtype=float))
    if filepath:
        fig.write_html(filepath)
    fig.show()
