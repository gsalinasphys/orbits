from math import acos, atan, cos, sqrt

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad


def get_2dtrajectory(d0: float, b: float, v0: float, k: float) -> tuple:
    semilatus = (v0 * b) ** 2 / k
    eccentricity = np.sqrt(v0**2 * semilatus / k  + 1)
    thetai = atan(b / d0)
    theta0 = thetai + np.sign(b)*acos(1/eccentricity * (semilatus / sqrt(d0**2 + b**2) - 1))

    def dt(theta):
        return semilatus**2 / (v0*b) / (1+eccentricity*cos(theta-theta0))**2

    return (thetai, 2*theta0-thetai), \
        lambda thetas: semilatus / (1 + eccentricity*np.cos(thetas - theta0)), \
        np.vectorize(lambda thetaf: quad(dt, thetai, thetaf))


def plot_2dtraj(d0: float, b: float, v0: float, k: float, n_points=100) -> None:
    (thetai, thetaf), rs, _ = get_2dtrajectory(d0, b, v0, k)
    thetas = np.linspace(thetai, thetaf, n_points)
    plt.plot(rs(thetas)*np.cos(thetas), rs(thetas)*np.sin(thetas))
