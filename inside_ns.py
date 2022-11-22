from typing import Callable

import mpmath as mp
import numpy as np
from matplotlib import pyplot as plt
from plotly import express as px
from scipy.optimize import root_scalar

from orbit2d import get_2dtrajectory, get_min_approach, plot_2dtraj
from orbit3d import _rotate, get_3dtrajectory, get_min_approach3d, plot_3dtraj

mp.mp.dps = 50

def newton(f: Callable, fprime: Callable, x0: mp.mpf, tol: float = 1e-12, maxiters: int = 100):
    i = 0
    while abs(f(x0)) > tol:
        i += 1
        if i > maxiters:
            return None
        x0 -= f(x0)/fprime(x0)  # Newton-Raphson
        
    return x0

def enter_ns(r02d: mp.matrix, v02d: mp.matrix, k: float, radius: float, t0: float = 0.) -> tuple:
    (thetamin, _, thetamax), r, rdot, thetadot, t = get_2dtrajectory(r02d, v02d, k, t0)
    thetaclose = (thetamin+thetamax) / 2.

    to_root = lambda theta: mp.norm(r(theta)) - radius
    theta = newton(to_root, lambda theta: mp.diff(to_root, theta), (thetamin+thetaclose)/2.)
    if theta:
        return theta, r(theta), rdot(theta), thetadot(theta), t(theta)

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

def find_ns_exit(rin: mp.matrix, vin: mp.matrix, tin: float, k: float, radius: float, epsilon: float = 1e-12) -> tuple:
    omega = mp.sqrt(k / radius**3)
    r2d, v2d = get_2dtraj_in_ns(rin, vin, tin, k, radius)

    to_root = lambda angle: mp.norm(r2d(angle/omega+tin)) - radius
    angle_exit = mp.mpf('0.')
    while abs(angle_exit) < epsilon or abs(angle_exit-mp.pi) < epsilon:
        angle_exit = newton(to_root, lambda theta: mp.diff(to_root, theta), mp.pi*mp.rand()) % mp.pi
    
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

    thetamin, thetaout, thetamax = infall[0][0], outfall[0][1], outfall[0][2]
    return (thetamin, thetain, thetaout, thetamax), infall[1:-1] + (tinfall, ), outfall[1:]

def plot_2dtraj_through_ns(r02d: mp.matrix, v02d: mp.matrix, k: float, radius: float, n_points: int = 10_000) -> None:
    thetas, *_ = get_2dtraj_through_ns(r02d, v02d, k, radius)
    thetas_infall = mp.linspace(thetas[0], thetas[1], n_points)[1:]
    plotin = plot_2dtraj(r02d, v02d, k, n_points, (thetas_infall[0], thetas_infall[-1]))

    thetain, r2din, rdotin, thetadotin, tin = enter_ns(r02d, v02d, k, radius)

    rin = mp.matrix([r2din*mp.cos(thetain), r2din*mp.sin(thetain)])
    vin = mp.matrix([rdotin*mp.cos(thetain) - r2din*thetadotin*mp.sin(thetain),
                    rdotin*mp.sin(thetain) + r2din*thetadotin*mp.cos(thetain)])

    plotns = plot_2dtraj_in_ns(rin, vin, tin, k, radius)

    thetas_outfall = mp.linspace(thetas[-2], thetas[-1], n_points)[:-1]
    rout, vout, _ = find_ns_exit(rin, vin, tin, k, radius)

    plotout = plot_2dtraj(rout, vout, k, n_points, (thetas_outfall[0], thetas_outfall[-1]))

    return plotin, plotns, plotout

def get_2dtraj_ns(r02d: mp.matrix, v02d: mp.matrix, k: float, radius: float, t0: float = 0.) -> tuple:
    min_approach = get_min_approach(r02d, v02d, k)
    if min_approach > radius:
        return get_2dtrajectory(r02d, v02d, k, t0)
    else:
        return get_2dtraj_through_ns(r02d, v02d, k, radius, t0)

def plot_2dtraj_ns(r02d: mp.matrix, v02d: mp.matrix, k: float, radius: float, n_points: int = 10_000) -> None:
    min_approach = get_min_approach(r02d, v02d, k)
    if min_approach > radius:
        return plot_2dtraj(r02d, v02d, k, n_points)
    else:
        return plot_2dtraj_through_ns(r02d, v02d, k, radius, n_points)

def get_3dtraj_through_ns(r0: mp.matrix, v0: mp.matrix, k: float, radius: float) -> tuple:
    rotation = _rotate(r0, v0)
    r02d, v02d = rotation * r0, rotation * v0

    thetas, infall, outfall = get_2dtraj_through_ns(r02d[:2], v02d[:2], k, radius)

    rotation_inv = rotation**(-1)
    rinfall, rdotinfall, thetadotinfall, tinfall = infall
    r3dinfall = lambda theta: rotation_inv * mp.matrix([rinfall(theta) * mp.cos(theta), rinfall(theta) * mp.sin(theta), 0.])
    v3dinfall = lambda theta: rotation_inv * mp.matrix([rdotinfall(theta)*mp.cos(theta) - rinfall(theta)*thetadotinfall(theta)*mp.sin(theta),
                                                rdotinfall(theta)*mp.sin(theta) + rinfall(theta)*thetadotinfall(theta)*mp.cos(theta),
                                                0.])
    infall3d = r3dinfall, v3dinfall, tinfall

    routfall, rdotoutfall, thetadotoutfall, toutfall = outfall
    r3doutfall = lambda theta: rotation_inv * mp.matrix([routfall(theta) * mp.cos(theta), routfall(theta) * mp.sin(theta), 0.])
    v3doutfall = lambda theta: rotation_inv * mp.matrix([rdotoutfall(theta)*mp.cos(theta) - routfall(theta)*thetadotoutfall(theta)*mp.sin(theta),
                                                rdotoutfall(theta)*mp.sin(theta) + routfall(theta)*thetadotoutfall(theta)*mp.cos(theta),
                                                0.])
    outfall3d = r3doutfall, v3doutfall, toutfall

    return thetas, infall3d, outfall3d

def plot_3dtraj_through_ns(r0: mp.matrix, v0: mp.matrix, k: float, radius: float, n_points: int = 10_000):
    (thetamin, thetain, thetaout, thetamax), infall2d, outfall3d = get_3dtraj_through_ns(r0, v0, k, radius)
    thetasin = mp.linspace(thetamin, thetain, n_points)[1:-1]
    thetasout = mp.linspace(thetaout, thetamax, n_points)[1:-1]

    rsin = np.array([infall2d[0](theta) for theta in thetasin], dtype=float)
    rsout = np.array([outfall3d[0](theta) for theta in thetasout], dtype=float)
    rs = np.concatenate((rsin, rsout))

    return px.line_3d(x=rs[:, 0], y=rs[:, 1], z=rs[:, 2])

def get_3dtraj_ns(r0: mp.matrix, v0: mp.matrix, k: float, radius: float, t0: float = 0.) -> tuple:
    min_approach = get_min_approach3d(r0, v0, k)
    if min_approach > radius:
        return get_3dtrajectory(r0, v0, k, t0)
    else:
        return get_3dtraj_through_ns(r0, v0, k, radius, t0)

def plot_3dtraj_ns(r0: mp.matrix, v0: mp.matrix, k: float, radius: float, n_points: int = 10_000) -> None:
    min_approach = get_min_approach3d(r0, v0, k)
    if min_approach > radius:
        return plot_3dtraj(r0, v0, k, n_points)
    else:
        return plot_3dtraj_through_ns(r0, v0, k, radius, n_points)