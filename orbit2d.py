import mpmath as mp
import numpy as np
from matplotlib import pyplot as plt

mp.mp.dps = 50

def get_2dtrajectory(d0: float, b: float, v0: float, k: float) -> tuple:
    d0, b, v0, k = mp.mpf(d0), mp.mpf(b), mp.mpf(v0), mp.mpf(k)
    Jbar = -v0 * b
    Ebar = v0**2/2 - k/mp.sqrt(d0**2 + b**2)
    semilatus = (Jbar) ** 2 / k
    eccentricity = mp.sqrt(2*Ebar*semilatus/k  + 1)
    thetai = mp.atan(b / d0) if d0 else mp.sign(b)*mp.pi/2
    theta0 = thetai + mp.acos(1/eccentricity * (semilatus / mp.sqrt(d0**2 + b**2) - 1))
    thetamin, thetamax = theta0 - mp.acos(-1/eccentricity), theta0 + mp.acos(-1/eccentricity)

    rs = np.vectorize(lambda theta: semilatus / (1 + eccentricity*mp.cos(theta - theta0)))
    rdots = np.vectorize(lambda theta: eccentricity * Jbar / semilatus * mp.sin(theta-theta0))
    thetadots = lambda thetas: Jbar / rs(thetas)**2

    hyperbolic_anomaly = lambda theta: mp.acosh((mp.cos(theta-theta0)+eccentricity) / (1+eccentricity*mp.cos(theta-theta0)))
    mean_anom = lambda theta: eccentricity*mp.sinh(hyperbolic_anomaly(theta)) - hyperbolic_anomaly(theta)
    t = lambda theta1, theta2: -semilatus**2 / Jbar  / (eccentricity**2-1)**1.5 * (mean_anom(theta2) - mean_anom(theta1))
    ts = np.vectorize(lambda thetaf: t(thetai, thetaf) if thetaf <= theta0 \
            else 2*t(thetai, theta0) - t(2*theta0-thetai, thetaf))

    return (thetamin, thetamax), rs, rdots, thetadots, ts

def plot_2dtraj(d0: float, b: float, v0: float, k: float, n_points: int = 10_000) -> None:
    (thetamin, thetamax), rs, *_ = get_2dtrajectory(d0, b, v0, k)
    thetas = mp.linspace(thetamin, thetamax, n_points)[1:-1]

    x = np.vectorize(lambda theta: rs(theta) * mp.cos(theta))
    y = np.vectorize(lambda theta: rs(theta) * mp.sin(theta))

    plt.plot(x(thetas), y(thetas))
