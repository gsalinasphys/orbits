import mpmath as mp
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import root_scalar

from orbit2d import get_2dtrajectory, get_min_approach, plot_2dtraj

mp.mp.dps = 50

def enter_ns(r02d: mp.matrix, v02d: mp.matrix, k: float, radius: float, t0: float = 0.) -> tuple:
    (thetamin, _, thetamax), r, rdot, thetadot, t = get_2dtrajectory(r02d, v02d, k, t0)
    thetaclose = (thetamin+thetamax) / 2.

    try:
        root_found = root_scalar(lambda theta: mp.norm(r(theta))-radius, bracket=(thetamin, thetaclose), xtol=1e-15, rtol=1e-15)
        if root_found.converged:
            return mp.mpf(root_found.root), r(root_found.root), rdot(root_found.root), thetadot(root_found.root), t(root_found.root)
        else:
            return None
    except ValueError as e:
        print(e)
        return None

def get_2dtraj_in_ns(rin: mp.matrix, vin: mp.matrix, tin: float, k: float, radius: float) -> mp.matrix:
    omega = mp.sqrt(k / radius**3)
    rns = lambda t: mp.matrix([rin[0]*mp.cos(omega*(t-tin)) + vin[0]/omega*mp.sin(omega*(t-tin)),
                            rin[1]*mp.cos(omega*(t-tin)) + vin[1]/omega*mp.sin(omega*(t-tin))])

    vns = lambda t: mp.matrix([vin[0]*mp.cos(omega*(t-tin)) - rin[0]*omega*mp.sin(omega*(t-tin)),
                            vin[1]*mp.cos(omega*(t-tin)) - rin[1]*omega*mp.sin(omega*(t-tin))])

    return rns, vns

def plot_2dtraj_in_ns(rin: mp.matrix, vin: mp.matrix, tin: float, k: float, radius: float, n_points: int = 10_000) -> mp.matrix:
    omega = mp.sqrt(k / radius**3)
    rns, _ = get_2dtraj_in_ns(rin, vin, tin, k, radius)

    tsns = mp.linspace(tin, tin + 2*mp.pi/omega, n_points)
    rsns = np.array([rns(tns) for tns in tsns])

    plt.plot(rsns.T[0], rsns.T[1])

def find_ns_exit(rin: mp.matrix, vin: mp.matrix, tin: float, k: float, radius: float, n_points: int = 10_000) -> tuple:
    omega = mp.sqrt(k / radius**3)
    r2d, v2d = get_2dtraj_in_ns(rin, vin, tin, k, radius)
    to_root = lambda angle: mp.norm(r2d(angle/omega+tin)) - radius

    angles = mp.linspace(0, 2*mp.pi, n_points, endpoint=False)
    to_root_vals = np.array([to_root(angle) for angle in angles])
    indices = np.where(to_root_vals[:-1]*to_root_vals[1:] < 0)[0]
    index = indices[0] if indices[0] else indices[1]

    angle_exit = mp.mpf(root_scalar(to_root, bracket=(angles[index], angles[index+1]), xtol=1e-15, rtol=1e-15).root)

    return r2d(angle_exit/omega+tin), v2d(angle_exit/omega+tin), angle_exit/omega+tin

def get_2dtraj_through_ns(r02d: mp.matrix, v02d: mp.matrix, k: float, radius: float, t0: float = 0.) -> tuple:
    infall = get_2dtrajectory(r02d, v02d, k)
    thetain, r2din, rdotin, thetadotin, tin = enter_ns(r02d, v02d, k, radius)
    
    rin = mp.matrix([r2din*mp.cos(thetain), r2din*mp.sin(thetain)])
    vin = mp.matrix([rdotin*mp.cos(thetain) - r2din*thetadotin*mp.sin(thetain),
                    rdotin*mp.sin(thetain) + r2din*thetadotin*mp.cos(thetain)])

    rout, vout, tout = find_ns_exit(rin, vin, tin, k, radius)

    tinfall = lambda theta: infall[-1](theta) - tout

    outfall = get_2dtrajectory(rout, vout, k)

    return (infall[0][0], thetain, outfall[0][1], outfall[0][2]), infall[1:-1] + (tinfall, ), outfall[1:]

def plot_2dtraj_through_ns(r02d: mp.matrix, v02d: mp.matrix, k: float, radius: float, n_points: int = 10_000) -> None:
    thetas, *_ = get_2dtraj_through_ns(r02d, v02d, k, radius)
    thetas_infall = mp.linspace(thetas[0], thetas[1], n_points)[1:]
    plot_2dtraj(r02d, v02d, k, n_points, (thetas_infall[0], thetas_infall[-1]))

    thetain, r2din, rdotin, thetadotin, tin = enter_ns(r02d, v02d, k, radius)

    rin = mp.matrix([r2din*mp.cos(thetain), r2din*mp.sin(thetain)])
    vin = mp.matrix([rdotin*mp.cos(thetain) - r2din*thetadotin*mp.sin(thetain),
                    rdotin*mp.sin(thetain) + r2din*thetadotin*mp.cos(thetain)])

    plot_2dtraj_in_ns(rin, vin, tin, k, radius)

    thetas_outfall = mp.linspace(thetas[-2], thetas[-1], n_points)[:-1]
    rout, vout, _ = find_ns_exit(rin, vin, tin, k, radius)

    plot_2dtraj(rout, vout, k, n_points, (thetas_outfall[0], thetas_outfall[-1]))

def get_2dtraj_ns(r02d: mp.matrix, v02d: mp.matrix, k: float, radius: float, t0: float = 0.) -> tuple:
    min_approach = get_min_approach(r02d, v02d, k)
    if min_approach > radius:
        return get_2dtrajectory(r02d, v02d, k, t0)
    else:
        return get_2dtraj_through_ns(r02d, v02d, k, radius, t0)

def plot_2dtraj_ns(r02d: mp.matrix, v02d: mp.matrix, k: float, radius: float, n_points: int = 10_000) -> None:
    min_approach = get_min_approach(r02d, v02d, k)
    if min_approach > radius:
        plot_2dtraj(r02d, v02d, k, n_points)
    else:
        plot_2dtraj_through_ns(r02d, v02d, k, radius, n_points)
