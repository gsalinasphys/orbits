import mpmath as mp
import numpy as np
from plotly import express as px
from scipy.spatial.transform import Rotation

from orbit2d import get_2dtrajectory

mp.mp.dps = 50

def _twodim_params(r0: mp.matrix, v0: mp.matrix) -> tuple:
    v02d = mp.norm(v0)
    b = mp.norm(np.cross(r0, v0)) / v02d
    d0 = mp.sqrt(mp.norm(r0)**2 - b**2)

    return d0, b, mp.sign(np.dot(r0, v0)) * v02d

def _cross_matrix(v: mp.matrix) -> mp.matrix:
    return mp.matrix([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])

def _rotation_matrix(angle: float, unit_vector: mp.matrix) -> mp.matrix:
    return mp.cos(angle)*mp.eye(3) + mp.sin(angle)*_cross_matrix(unit_vector) + (1-mp.cos(angle))*unit_vector*unit_vector.T

def _rotate_r(r0: mp.matrix) -> mp.matrix:
    angle = -mp.atan(r0[2]/r0[1]) if r0[1] else mp.pi/2
    rotation = _rotation_matrix(angle, mp.matrix([1., 0, 0]))
    r1 = rotation * r0
    angle2 = -mp.atan(r1[1]/r1[0]) if r1[0] else mp.pi/2
    rotation2 = _rotation_matrix(angle2, mp.matrix([0., 0, 1]))
    r2 = rotation2 * r1
    factor = 1.
    if r2[0] < 0:
        factor *= _rotation_matrix(mp.pi, mp.matrix([0., 0, 1]))

    return factor * rotation2 * rotation

def _rotate(r0: mp.matrix, v0: mp.matrix) -> mp.matrix:
    rotation1 = _rotate_r(r0)
    r1, v1 = rotation1 * r0, rotation1 * v0
    angle = -mp.atan(v1[2]/v1[1]) if v1[1] else mp.pi/2
    rotation2 = _rotation_matrix(angle, mp.matrix([1., 0, 0]))
    v2 = rotation2 * v1
    angle2 = -mp.atan(v2[1]/v2[0]) if v2[0] else mp.pi/2
    rotation3 = _rotation_matrix(angle2, mp.matrix([0, 0, 1]))
    r2 = rotation3 * r1
    factor = 1.

    if r2[1] < 0:
        factor *= _rotation_matrix(mp.pi, mp.matrix([1., 0, 0]))

    return (factor * rotation3 * rotation2 * rotation1) ** (-1)

def get_3dtrajectory(r0: mp.matrix, v0: mp.matrix, k: float) -> tuple:
    d0, b, v02d = _twodim_params(r0, v0)
    (thetamin, _, thetamax), r, rdot, thetadot, t = get_2dtrajectory(d0, b, v02d, k)

    rotation = _rotate(r0, v0)

    r3d = lambda theta: rotation * mp.matrix([r(theta) * mp.cos(theta), r(theta) * mp.sin(theta), 0.])
    v3d = lambda theta: rotation * mp.matrix([rdot(theta)*mp.cos(theta) - r(theta)*thetadot(theta)*mp.sin(theta),
                                            rdot(theta)*mp.sin(theta) + r(theta)*thetadot(theta)*mp.cos(theta),
                                            0.])

    return (thetamin, thetamax), r3d, v3d, t

def plot_3dtraj(r0: mp.matrix, v0: mp.matrix, k: float, n_points: int = 10_000, filepath: str = ''):
    (thetamin, thetamax), r, *_ = get_3dtrajectory(r0, v0, k)
    thetas = mp.linspace(thetamin, thetamax, n_points)[1:-1]

    rs = np.array([r(theta) for theta in thetas], dtype=float)

    fig = px.line_3d(x=rs[:, 0], y=rs[:, 1], z=rs[:, 2])
    if filepath:
        fig.write_html(filepath)
    fig.show()
