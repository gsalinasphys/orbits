from math import acos, atan, sqrt

import mpmath as mp
import numpy as np
from matplotlib import pyplot as plt

mp.mp.dps = 50

def get_2dtrajectory(d0: float, b: float, v0: float, k: float) -> tuple:
    d0, b, v0, k = mp.mpf(d0), mp.mpf(b), mp.mpf(v0), mp.mpf(k)
    Jbar = v0 * b
    Ebar = v0**2/2 - k/d0
    semilatus = (Jbar) ** 2 / k
    eccentricity = mp.sqrt(2*Ebar*semilatus/k  + 1)
    thetai = mp.atan(b / d0)
    theta0 = thetai + mp.sign(b)*mp.acos(1/eccentricity * (semilatus / mp.sqrt(d0**2 + b**2) - 1))

    hyperbolic_anomaly = lambda theta: mp.acosh((mp.cos(theta-theta0)+eccentricity) / (1+eccentricity*mp.cos(theta-theta0)))
    mean_anom = lambda theta: eccentricity*mp.sinh(hyperbolic_anomaly(theta)) - hyperbolic_anomaly(theta)
    t = lambda theta1, theta2: -mp.sign(b)*semilatus**2 / Jbar  / (eccentricity**2-1)**1.5 * (mean_anom(theta2) - mean_anom(theta1))


    return (thetai, 2*theta0-thetai), \
        np.vectorize(lambda theta: semilatus / (1 + eccentricity*mp.cos(theta - theta0))), \
        np.vectorize(lambda thetaf: t(thetai, thetaf) if mp.sign(b)*thetaf <= mp.sign(b)*theta0 \
            else 2*t(thetai, theta0) - t(2*theta0-thetai, thetaf))


def plot_2dtraj(d0: float, b: float, v0: float, k: float, n_points=100) -> None:
    (thetai, thetaf), rs, _ = get_2dtrajectory(d0, b, v0, k)
    thetas = mp.linspace(thetai, thetaf, n_points)

    x = np.vectorize(lambda theta: rs(theta) * mp.cos(theta))
    y = np.vectorize(lambda theta: rs(theta) * mp.sin(theta))

    plt.plot(x(thetas), y(thetas))
