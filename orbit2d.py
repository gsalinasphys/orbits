from math import acos, atan, cos, log, sin, sqrt

import numpy as np
from matplotlib import pyplot as plt
from mpmath import quad


def get_2dtrajectory(d0: float, b: float, v0: float, k: float, theta_cut: float = 1e-4) -> tuple:
    semilatus = (v0 * b) ** 2 / k
    eccentricity = np.sqrt(v0**2 * semilatus / k  + 1)
    thetai = atan(b / d0)
    theta0 = thetai + np.sign(b)*acos(1/eccentricity * (semilatus / sqrt(d0**2 + b**2) - 1))

    def dt(theta):
        return semilatus**2 / (v0*b) / (1+eccentricity*cos(theta-theta0))**2
    
    def indefinite_integral(x, a=eccentricity*sin(theta0)):
        return -2*(a+x)/a**2/x/(2*a+x) - log(x)/a**3 + log(2*a+x)/a**3

    def definite_integral(theta1, theta2):
        return semilatus**2/(v0*b) * (indefinite_integral(theta2) - indefinite_integral(theta1))

    integral_to_cut = definite_integral(thetai, theta_cut)

    t = lambda thetaf: definite_integral(thetai, thetaf) \
        if thetaf < theta_cut else quad(dt, [theta_cut, thetaf])

    return (thetai, 2*theta0-thetai), \
        lambda thetas: semilatus / (1 + eccentricity*np.cos(thetas - theta0)), \
        np.vectorize(t)


def plot_2dtraj(d0: float, b: float, v0: float, k: float, n_points=100) -> None:
    (thetai, thetaf), rs, _ = get_2dtrajectory(d0, b, v0, k)
    thetas = np.linspace(thetai, thetaf, n_points)
    plt.plot(rs(thetas)*np.cos(thetas), rs(thetas)*np.sin(thetas))
