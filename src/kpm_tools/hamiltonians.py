"""Quick reference of topological systems.

Defined both with open boundary and periodic boundary conditions.
"""
from itertools import product

import kwant
import numpy as np
from kwant import HoppingKind


# neighbors for honeycomb lattice
honeycomb = kwant.lattice.honeycomb(1, norbs=1, name=["a", "b"])
honeycomb_a, honeycomb_b = honeycomb.sublattices

first_neighbors = [
    HoppingKind((0, 1), honeycomb_a, honeycomb_b),
    HoppingKind((0, 0), honeycomb_a, honeycomb_b),
    HoppingKind((-1, 1), honeycomb_a, honeycomb_b),
]

second_neighbors_a = [
    HoppingKind((0, 1), honeycomb_a, honeycomb_a),
    HoppingKind((1, -1), honeycomb_a, honeycomb_a),
    HoppingKind((-1, 0), honeycomb_a, honeycomb_a),
]

second_neighbors_b = [
    HoppingKind((1, 0), honeycomb_b, honeycomb_b),
    HoppingKind((0, -1), honeycomb_b, honeycomb_b),
    HoppingKind((-1, 1), honeycomb_b, honeycomb_b),
]


def haldane_pbc(trans_symm=((1, 0), (0, 1)), e_0=0, t=1, t2=0.5, return_builder=False):
    """Build a Haldane system with periodic boundary conditions."""
    periods = np.dot(trans_symm, honeycomb.prim_vecs)
    syst = kwant.builder.Builder(kwant.lattice.TranslationalSymmetry(*periods))

    syst[honeycomb.shape(lambda s: True, (0, 0))] = e_0
    syst[first_neighbors] = t
    syst[second_neighbors_a] = 1j * t2
    syst[second_neighbors_b] = 1j * t2

    if return_builder:
        return honeycomb, syst
    return honeycomb, kwant.wraparound.wraparound(syst).finalized()


def haldane_obc(trans_symm=((1, 0), (0, 1)), e_0=0, t=1, t2=0.5):
    """Build a Haldane system with open boundary conditions."""
    trans_symm = np.array(trans_symm, dtype=int)
    indices_left = trans_symm[1] - trans_symm[0]
    sign = np.sign(indices_left)

    syst = kwant.builder.Builder()

    for i, j in product(
        range(0, indices_left[0], sign[0]), range(0, indices_left[1], sign[1])
    ):
        tag = trans_symm[0] + np.array([i, j], dtype=int)
        site_a = honeycomb_a(*tag)
        site_b = honeycomb_b(*tag)
        syst[site_a] = e_0
        syst[site_b] = e_0

    syst[first_neighbors] = t
    syst[second_neighbors_a] = 1j * t2
    syst[second_neighbors_b] = 1j * t2

    return honeycomb, syst.finalized()


params_qhe = dict(mu=-4, t=-1, B=np.pi / 7)


def qhe_obc(length=50, width=10, init_pos=(0, 0)):
    """Build a quantum Hall system with open boundary conditions."""

    def bar(pos):
        x = pos[0] - init_pos[0]
        y = pos[1] - init_pos[1]
        return (x >= 0 and x < length) and (y >= 0 and y < width)

    # Onsite and hoppings
    def onsite(site, t, mu):
        return 4 * t - mu

    def hopping_ax(site1, site2, t, B):  # noqa: N803
        xt, yt = site1.pos
        xs, ys = site2.pos
        return -t * np.exp(-0.5j * B * (xt + xs) * (yt - ys))

    # Building system
    lat = kwant.lattice.square(norbs=1)
    syst = kwant.builder.Builder()

    syst[lat.shape(bar, init_pos)] = onsite
    syst[lat.neighbors()] = hopping_ax

    return syst.finalized()


def qhe_pbc(trans_symm=((1, 0), (0, 1)), init_pos=(0, 0), return_builder=False):
    """Build a quantum Hall system with periodic boundary conditions."""
    # Building system
    lat = kwant.lattice.square(norbs=1)
    periods = np.dot(trans_symm, lat.prim_vecs)
    syst = kwant.builder.Builder(kwant.lattice.TranslationalSymmetry(*periods))

    # Onsite and hoppings
    def onsite(site, t, mu):
        return 4 * t - mu

    def hopping_ax(site1, site2, t, B):  # noqa: N803
        xt, yt = site1.pos
        xs, ys = site2.pos
        return -t * np.exp(-0.5j * B * (xt + xs) * (yt - ys))

    syst[lat.shape(lambda pos: True, init_pos)] = onsite
    syst[lat.neighbors()] = hopping_ax

    if return_builder:
        return syst
    return kwant.wraparound.wraparound(syst).finalized()
