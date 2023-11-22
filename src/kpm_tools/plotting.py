"""Plotting functions and others."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from .common import trace_unit_cell


def plot_kwant(
    sites,
    site_color=None,
    linewidths=0.2,
    site_size=10,
    edgecolor="k",
    cmap="Blues",
    depth=False,
    elev=None,
    azim=None,
    ax=None,
    show=False,
    unit_cell=False,
    pos_transform=None,
    colorbar=True,
    **kwargs
):
    """Plot a list of kwant system sites with site colors and colorbar."""
    if site_color is None:
        site_color = np.ones(len(sites))
    if unit_cell:
        family_name = sites[0].family.name
        sites = [s for s in sites if s.family.name == family_name]
        site_color = trace_unit_cell(sites, site_color)
        # traces norbs and sublattices
    else:
        sites = sites
        site_color = site_color
    if len(sites) != site_color.shape[0]:
        raise ValueError(
            "The number of sites (orbitals) to plot must match " "the 'array' length"
        )

    dim = sites[0].family.dim

    fig = plt.gcf()

    if pos_transform is None:

        def pos_transform(x):
            return x

    positions = np.array([pos_transform(s.pos) for s in sites]).T

    if site_color is None:
        site_color = np.ones(positions.shape[1])

    return_artists = True
    if ax is None:
        return_artists = False
        if dim == 3:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

    # TODO: check *positions warning
    if dim == 3:
        art = ax.scatter3D(
            *positions,
            c=site_color,
            cmap=cmap,
            linewidths=linewidths,
            s=site_size,
            edgecolor=edgecolor,
            depthshade=depth,
            **kwargs
        )

        ax.view_init(elev=elev, azim=azim)
    else:
        art = ax.scatter(
            *positions,
            c=site_color,
            cmap=cmap,
            linewidths=linewidths,
            s=site_size,
            edgecolor=edgecolor,
            **kwargs
        )

    if colorbar:
        fig.colorbar(art, ax=ax)

    if show:
        fig.show()

    if return_artists:
        return art
    return fig


def plot_neighbors(
    syst,
    lat=None,
    neighs=None,
    show=False,
    more_sites=False,
    init_tags=((0, 0)),
    ax=None,
    scatter_kwargs=None,
    arrow_kwargs=None,
):
    """Plot arrows between the neighbors."""
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if arrow_kwargs is None:
        arrow_kwargs = {}
    plot_kwargs = dict(
        color="k", arrowstyle="simple", mutation_scale=15, lw=0, alpha=0.5
    )
    plot_kwargs.update(arrow_kwargs)
    dim = lat.sublattices[0].dim
    if neighs is None:
        if lat is None:
            try:
                lat = syst.sites[0].family
            except Exception:
                lat = list(syst.sites())[0].family
        neighs = []
        for sublat in lat.sublattices:
            neighs.extend(sublat.neighbors())

    if ax is None:
        ax = plt.gca()

    for init_tag in init_tags:
        for nei in neighs:
            site_a = nei.family_a(*init_tag)
            second_tag = np.array(init_tag) - np.array(nei.delta)
            site_b = nei.family_b(*(second_tag))
            delta_pos = site_b.pos - site_a.pos

            if dim == 2:
                ax.arrow(
                    x=site_a.pos[0],
                    y=site_a.pos[1],
                    dx=delta_pos[0],
                    dy=delta_pos[1],
                    head_width=0.05,
                    head_length=0.1,
                    fc="k",
                    ec="k",
                )
            elif dim == 3:
                rs = np.array([site_a.pos, site_a.pos + delta_pos]).T
                arrow = Arrow3D(*rs, **plot_kwargs)
                ax.add_artist(arrow)

            ax.scatter(*site_a.pos, **scatter_kwargs)
            ax.scatter(*site_b.pos, **scatter_kwargs)

            if more_sites:
                site_a = nei.family_a(*second_tag)
                second_tag = second_tag - np.array(nei.delta)
                site_b = nei.family_b(*(second_tag))
                delta_pos = site_b.pos - site_a.pos

                if dim == 2:
                    ax.arrow(
                        x=site_a.pos[0],
                        y=site_a.pos[1],
                        dx=delta_pos[0],
                        dy=delta_pos[1],
                        head_width=0.05,
                        head_length=0.1,
                        fc="k",
                        ec="k",
                    )
                elif dim == 3:
                    rs = np.array([site_a.pos, site_a.pos + delta_pos]).T
                    arrow = Arrow3D(*rs, **plot_kwargs)
                    ax.add_artist(arrow)
                ax.scatter(*site_a.pos, **scatter_kwargs)
                ax.scatter(*site_b.pos, **scatter_kwargs)
    if show:
        plt.show()
    return ax


def plot_vectors(
    vectors,
    ax=None,
    hl=0.1,  # head_length
    head_width=0.05,
    color="k",  # can be overridden,
    show=False,
    fig_size=None,
    resize_ax=False,
    margin=0.2,
    **kwargs
):
    """Plot vectors.

    Args:
        vectors (_type_): _description_
        ax (_type_, optional): _description_. Defaults to None.
        hl (float, optional): _description_. Defaults to 0.1.
        color (str, optional): _description_. Defaults to "k".
        show (bool, optional): _description_. Defaults to False.
        fig_size (_type_, optional): _description_. Defaults to None.
        resize_ax (bool, optional): _description_. Defaults to False.
        margin (float, optional): _description_. Defaults to 0.2.

    Returns:
        _type_: _description_
    """
    fig = plt.gcf()
    if fig_size is not None:
        fig.set_size_inches(fig_size)
    if ax is None:
        ax = plt.gca()

    directions = []
    for vector in vectors:
        _ec = vector.get("color", color)
        _fc = vector.get("color", color)
        _hl = vector.get("hl", hl)
        _hw = vector.get("head_width", head_width)
        origin = vector.get("origin", (0, 0))
        direction = vector.get("direction", (1, 1))
        directions.append(direction)

        ax.arrow(
            x=origin[0],
            y=origin[1],
            dx=direction[0],
            dy=direction[1],
            head_width=_hw,
            head_length=_hl,
            fc=_fc,
            ec=_ec,
            **kwargs
        )

    directions = np.array(directions)

    if resize_ax:
        ptp = np.ptp(directions, axis=0) * margin  # % margin
        lower_bound = np.min(directions, axis=0)
        upper_bound = np.max(directions, axis=0)

        ax.set_xlim(lower_bound[0] - ptp[0], upper_bound[0] + ptp[0])
        ax.set_ylim(lower_bound[1] - ptp[1], upper_bound[1] + ptp[1])
    if show:
        plt.show()
    return fig


class Arrow3D(FancyArrowPatch):
    """Patch fancy arrow.

    Args:
        FancyArrowPatch (_type_): _description_
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        """Initialize.

        Args:
            xs (_type_): _description_
            ys (_type_): _description_
            zs (_type_): _description_
        """
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """Draw.

        Args:
            renderer (_type_): _description_
        """
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plot_arrows3d(start=None, arrow=None, ax=None, show=False, **kwargs):
    """Plot 3d arrows.

    Args:
        start (_type_, optional): _description_. Defaults to None.
        arrow (_type_, optional): _description_. Defaults to None.
        ax (_type_, optional): _description_. Defaults to None.
        show (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    fig = plt.gcf()
    if ax is None:
        ax = fig.add_subplot(111, projection="3d")

    plot_kwargs = dict(mutation_scale=5, lw=3, arrowstyle="-|>", color="b", alpha=0.5)
    plot_kwargs.update(kwargs)

    for i in range(len(start)):
        a = Arrow3D(
            [start[i, 0], start[i, 0] + arrow[i, 0]],
            [start[i, 1], start[i, 1] + arrow[i, 1]],
            [start[i, 2], start[i, 2] + arrow[i, 2]],
            **plot_kwargs
        )
        ax.add_artist(a)

    if show:
        plt.show()
    return fig
