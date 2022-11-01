from typing import Callable

import mpmath as mp
import numpy as np
from matplotlib import pyplot as plt

mp.mp.dps = 50

def get_orbit_params(r02d: mp.matrix, v02d: mp.matrix, k: float) -> tuple:
    Jbar = np.cross(r02d, v02d)
    Ebar = mp.norm(v02d)**2/2 - k/mp.norm(r02d)
    semilatus = (Jbar) ** 2 / k
    eccentricity = mp.sqrt(2*Ebar*semilatus/k  + 1)

    return Jbar, Ebar, semilatus, eccentricity

def get_thetas(r02d: mp.matrix, v02d: mp.matrix, k: float) -> float:
    Jbar, _, semilatus, eccentricity = get_orbit_params(r02d, v02d, k)
    theta0 = mp.atan(r02d[1] / r02d[0]) + (1-mp.sign(r02d[0]))/2*mp.pi if r02d[0] else mp.sign(r02d[1])*mp.pi/2

    in_or_out = mp.sign(np.dot(r02d, v02d))
    is_theta0_gtr = np.sign(Jbar) * in_or_out

    if is_theta0_gtr == 1:
        thetaclose = theta0 - mp.acos(1/eccentricity * (semilatus / mp.norm(r02d) - 1))
    elif is_theta0_gtr == -1:
        thetaclose = theta0 + mp.acos(1/eccentricity * (semilatus / mp.norm(r02d) - 1))

    thetamin, thetamax = thetaclose - mp.acos(-1/eccentricity), thetaclose + mp.acos(-1/eccentricity)

    return thetamin, thetamax

def get_rv2d(r02d: mp.matrix, v02d: mp.matrix, k: float) -> tuple:
    Jbar, _, semilatus, eccentricity = get_orbit_params(r02d, v02d, k)
    thetamin, thetamax = get_thetas(r02d, v02d, k)
    thetaclose = (thetamin + thetamax) / 2.

    r = lambda theta: semilatus / (1+eccentricity*mp.cos(theta-thetaclose))
    rdot = lambda theta: eccentricity * Jbar / semilatus * mp.sin(theta-thetaclose)
    thetadot = lambda theta: Jbar / r(theta)**2

    return r, rdot, thetadot

def get_time(r02d: mp.matrix, v02d: mp.matrix, k: float, t0: float = 0.) -> Callable:
    Jbar, _, semilatus, eccentricity = get_orbit_params(r02d, v02d, k)
    theta0 = mp.atan(r02d[1] / r02d[0]) + (1-mp.sign(r02d[0]))/2*mp.pi if r02d[0] else mp.sign(r02d[1])*mp.pi/2
    thetamin, thetamax = get_thetas(r02d, v02d, k)
    thetaclose = (thetamin + thetamax) / 2.

    in_or_out = mp.sign(np.dot(r02d, v02d))
    is_theta0_gtr = np.sign(Jbar) * in_or_out

    hyperbolic_anomaly = lambda theta: mp.acosh((mp.cos(theta-thetaclose)+eccentricity) / (1+eccentricity*mp.cos(theta-thetaclose)))
    mean_anom = lambda theta: eccentricity*mp.sinh(hyperbolic_anomaly(theta)) - hyperbolic_anomaly(theta)

    t_same_half = lambda theta1, theta2: semilatus**2 / Jbar / (eccentricity**2-1)**1.5 * (mean_anom(theta1)-mean_anom(theta2))
    t = lambda theta: -is_theta0_gtr * t_same_half(theta0, theta) + t0 if is_theta0_gtr*theta >= is_theta0_gtr*thetaclose \
        else -is_theta0_gtr * (t_same_half(theta0, theta) + 2*t_same_half(theta, thetaclose)) + t0

    return t

def get_2dtrajectory(r02d: mp.matrix, v02d: mp.matrix, k: float, t0: float = 0.) -> tuple:
    theta0 = mp.atan(r02d[1] / r02d[0]) + (1-mp.sign(r02d[0]))/2*mp.pi if r02d[0] else mp.sign(r02d[1])*mp.pi/2
    thetamin, thetamax = get_thetas(r02d, v02d, k)
    r, rdot, thetadot = get_rv2d(r02d, v02d, k)
    t = get_time(r02d, v02d, k, t0)
    thetas = [thetamin, theta0, thetamax]
    if t((2*thetamin+thetamax)/3) > t((thetamin+2*thetamax)/3):
        thetas.reverse()

    return tuple(thetas), r, rdot, thetadot, t

def plot_2dtraj(r02d: mp.matrix, v02d: mp.matrix, k: float, n_points: int = 10_000, theta_range: tuple = None):
    (thetamin, _, thetamax), r, *_ = get_2dtrajectory(r02d, v02d, k)
    if theta_range is None:
        thetas = mp.linspace(thetamin, thetamax, n_points)[1:-1]
    else:
        thetas = mp.linspace(theta_range[0], theta_range[1], n_points)

    xs = [r(theta)*mp.cos(theta) for theta in thetas]
    ys = [r(theta)*mp.sin(theta) for theta in thetas]

    plt.plot(xs, ys)

def get_min_approach(r02d: mp.matrix, v02d: mp.matrix, k: float) -> float:
    *_, semilatus, eccentricity = get_orbit_params(r02d, v02d, k)

    return semilatus / (1+eccentricity)
