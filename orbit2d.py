import mpmath as mp
import numpy as np
from matplotlib import pyplot as plt

mp.mp.dps = 50

def get_2dtrajectory(d0: float, b: float, v0: float, k: float, t0: float = 0.) -> tuple:
    d0, b, v0, k = mp.mpf(d0), mp.mpf(b), mp.mpf(v0), mp.mpf(k)
    Jbar = -v0 * b
    Ebar = v0**2/2 - k/mp.sqrt(d0**2 + b**2)
    semilatus = (Jbar) ** 2 / k
    eccentricity = mp.sqrt(2*Ebar*semilatus/k  + 1)
    thetai = mp.atan(b / d0) if d0 else mp.sign(b)*mp.pi/2
    theta0 = thetai + mp.acos(1/eccentricity * (semilatus / mp.sqrt(d0**2 + b**2) - 1))
    thetamin, thetamax = theta0 - mp.acos(-1/eccentricity), theta0 + mp.acos(-1/eccentricity)

    r = lambda theta: semilatus / (1 + eccentricity*mp.cos(theta - theta0))
    rdot = lambda theta: eccentricity * Jbar / semilatus * mp.sin(theta-theta0)
    thetadot = lambda theta: Jbar / r(theta)**2

    hyperbolic_anomaly = lambda theta: mp.acosh((mp.cos(theta-theta0)+eccentricity) / (1+eccentricity*mp.cos(theta-theta0)))
    mean_anom = lambda theta: eccentricity*mp.sinh(hyperbolic_anomaly(theta)) - hyperbolic_anomaly(theta)
    t_1st_half = lambda theta1, theta2: -semilatus**2 / Jbar  / (eccentricity**2-1)**1.5 * (mean_anom(theta2) - mean_anom(theta1))
    t = lambda thetaf: t_1st_half(thetai, thetaf) + t0 if thetaf <= theta0 \
        else 2*t_1st_half(thetai, theta0) - t_1st_half(2*theta0-thetai, thetaf) + t0

    return (thetamin, thetai, thetamax), r, rdot, thetadot, t

def get_2dtrajectory2(r02d: mp.matrix, v02d: mp.matrix, k: float):
    Jbar = np.cross(r02d, v02d)
    Ebar = mp.norm(v02d)**2/2 - k/mp.norm(r02d)
    semilatus = (Jbar) ** 2 / k
    eccentricity = mp.sqrt(2*Ebar*semilatus/k  + 1)
    theta0 = mp.atan(r02d[1] / r02d[0]) + (1-mp.sign(r02d[0]))/2*mp.pi if r02d[0] else mp.sign(r02d[1])*mp.pi/2

    in_or_out = mp.sign(np.dot(r02d, v02d))
    is_theta0_gtr = np.sign(Jbar) * in_or_out
    if is_theta0_gtr == 1:
        thetaclose = theta0 - mp.acos(1/eccentricity * (semilatus / mp.norm(r02d) - 1))
    elif is_theta0_gtr == -1:
        thetaclose = theta0 + mp.acos(1/eccentricity * (semilatus / mp.norm(r02d) - 1))

    thetamin, thetamax = thetaclose - mp.acos(-1/eccentricity), thetaclose + mp.acos(-1/eccentricity)

    r = lambda theta: semilatus / (1 + eccentricity*mp.cos(theta - thetaclose))
    rdot = lambda theta: eccentricity * Jbar / semilatus * mp.sin(theta-thetaclose)
    thetadot = lambda theta: Jbar / r(theta)**2

    hyperbolic_anomaly = lambda theta: mp.acosh((mp.cos(theta-thetaclose)+eccentricity) / (1+eccentricity*mp.cos(theta-thetaclose)))
    mean_anom = lambda theta: eccentricity*mp.sinh(hyperbolic_anomaly(theta)) - hyperbolic_anomaly(theta)

    t_same_half = lambda theta1, theta2: semilatus**2 / Jbar / (eccentricity**2-1)**1.5 * (mean_anom(theta1)-mean_anom(theta2))
    t = lambda theta: -is_theta0_gtr * t_same_half(theta0, theta) if is_theta0_gtr*theta >= is_theta0_gtr*thetaclose \
        else -is_theta0_gtr * (t_same_half(theta0, theta) + 2*t_same_half(theta, thetaclose)) 
    # if is_theta0_gtr == 1:
    #     t = lambda theta: -t_same_half(theta0, theta) if theta >= thetaclose \
    #         else -t_same_half(theta0, theta) - 2*t_same_half(theta, thetaclose)
    # elif is_theta0_gtr == -1:
    #     t = lambda theta: t_same_half(theta0, theta) if theta <= thetaclose \
    #         else t_same_half(theta0, theta) + 2*t_same_half(theta, thetaclose)

    return (thetamin, theta0, thetamax), r, rdot, thetadot, t

def plot_2dtraj(d0: float, b: float, v0: float, k: float, n_points: int = 10_000, theta_range: tuple = None) -> None:
    (thetamin, _, thetamax), r, *_ = get_2dtrajectory(d0, b, v0, k)
    if theta_range is None:
        thetas = mp.linspace(thetamin, thetamax, n_points)[1:-1]
    else:
        thetas = mp.linspace(theta_range[0], theta_range[1], n_points)

    xs = [r(theta)*mp.cos(theta) for theta in thetas]
    ys = [r(theta)*mp.sin(theta) for theta in thetas]

    plt.plot(xs, ys)

def plot_2dtraj2(r02d: mp.matrix, v02d: mp.matrix, k: float, n_points: int = 10_000, theta_range: tuple = None):
    (thetamin, _, thetamax), r, *_ = get_2dtrajectory2(r02d, v02d, k)
    if theta_range is None:
        thetas = mp.linspace(thetamin, thetamax, n_points)[1:-1]
    else:
        thetas = mp.linspace(theta_range[0], theta_range[1], n_points)

    xs = [r(theta)*mp.cos(theta) for theta in thetas]
    ys = [r(theta)*mp.sin(theta) for theta in thetas]

    plt.plot(xs, ys)
