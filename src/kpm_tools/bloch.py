"""Handle systems with periodic boundary conditions.

'wraparound_by_parts' is reworked from `~kwant.wraparound`, specially MR !363.
"""
import cmath
import collections
import inspect
import warnings

import numpy as np
import tinyarray as ta
from kwant._common import get_parameters
from kwant._common import memoize
from kwant.builder import Builder
from kwant.builder import herm_conj
from kwant.lattice import TranslationalSymmetry
from kwant.system import SiteArray
from kwant.wraparound import WrappedBuilder
from kwant.wraparound import _callable_herm_conj
from kwant.wraparound import _set_signature
from kwant.wraparound import wraparound


def wraparound_by_parts(
    builder, keep=None, *, coordinate_names="xyz", separate_sites=False
):
    """Replace translational symmetries by momentum parameters.

    A new Builder instance is returned.  By default, each symmetry is replaced
    by one scalar momentum parameter that is appended to the already existing
    arguments of the system.  Optionally, one symmetry may be kept by using the
    `keep` argument. The momentum parameters will have names like 'k_n' where
    the 'n' are specified by 'coordinate_names'.

    Parameters
    ----------
    builder : `~kwant.builder.Builder`
        Kwant builder with translational symmetries.
    keep : int, optional
        Which (if any) translational symmetry to keep.
    coordinate_names : sequence of strings, default: 'xyz'
        The names of the coordinates along the symmetry
        directions of 'builder'.
    separate_sites : bool, default to 'False'
        Wheather to construct a new system without the hoppings that cross
        the fundamental domain of the translational symmetry. If 'True'
        return the system contained inside the fundamental domain, a
        dictionary of wrapped sites and a dictionary of wrapped hoppings.

    Notes
    -----
    Wraparound is stop-gap functionality until Kwant 2.x which will include
    support for higher-dimension translational symmetry in the low-level system
    format. It will be deprecated in the 2.0 release of Kwant.

    """

    @memoize
    def bind_site(val):
        def f(*args):
            a, *args = args
            return val(a, *args[:mnp])

        if not callable(val):
            raise ValueError("Value function should be a Callable.")
        _set_signature(f, get_parameters(val) + momenta)
        return f

    @memoize
    def bind_hopping_as_site(elem, val):
        def f(*args):
            a, *args = args
            phase = cmath.exp(1j * ta.dot(elem, args[mnp:]))
            v = val(a, sym.act(elem, a), *args[:mnp]) if callable(val) else val
            pv = phase * v
            return pv + herm_conj(pv)

        params = ("_site0",)
        if callable(val):
            params += get_parameters(val)[2:]  # cut off both site parameters
        _set_signature(f, params + momenta)
        return f

    @memoize
    def bind_hopping(elem, val):
        def f(*args):
            a, b, *args = args
            phase = cmath.exp(1j * ta.dot(elem, args[mnp:]))
            v = val(a, sym.act(elem, b), *args[:mnp]) if callable(val) else val
            return phase * v

        params = ("_site0", "_site1")
        if callable(val):
            params += get_parameters(val)[2:]  # cut off site parameters
        _set_signature(f, params + momenta)
        return f

    @memoize
    def bind_sum(num_sites, *vals):
        """Construct joint signature for all 'vals'."""

        def f(*in_args):
            acc = 0
            for val, selection in val_selection_pairs:
                if selection:  # Otherwise: reuse previous out_args.
                    out_args = tuple(in_args[i] for i in selection)
                if callable(val):
                    acc = acc + val(*out_args)
                else:
                    acc = acc + val
            return acc

        params = collections.OrderedDict()

        # Add the leading one or two 'site' parameters.
        site_params = [f"_site{i}" for i in range(num_sites)]
        for name in site_params:
            params[name] = inspect.Parameter(name, inspect.Parameter.POSITIONAL_ONLY)

        # Add all the other parameters (except for the momenta).  Setup the
        # 'selections'.
        selections = []
        for val in vals:
            if not callable(val):
                selections.append(())
                continue
            val_params = get_parameters(val)[num_sites:]
            if val_params[mnp:] != momenta:
                raise ValueError("Shapes don't match.")
            val_params = val_params[:mnp]
            selections.append((*site_params, *val_params))
            for p in val_params:
                # Skip parameters that exist in previously added functions.
                if p in params:
                    continue
                params[p] = inspect.Parameter(p, inspect.Parameter.POSITIONAL_ONLY)

        # Sort values such that ones with the same arguments are bunched.
        # Prepare 'val_selection_pairs' that is used in the function 'f' above.
        params_keys = list(params.keys())
        val_selection_pairs = []
        prev_selection = None
        argsort = sorted(range(len(selections)), key=selections.__getitem__)
        momenta_sel = tuple(range(mnp, 0, 1))
        for i in argsort:
            selection = selections[i]
            if selection and selection != prev_selection:
                prev_selection = selection = (
                    tuple(params_keys.index(s) for s in selection) + momenta_sel
                )
            else:
                selection = ()
            val_selection_pairs.append((vals[i], selection))

        # Finally, add the momenta.
        for k in momenta:
            params[k] = inspect.Parameter(k, inspect.Parameter.POSITIONAL_ONLY)

        f.__signature__ = inspect.Signature(params.values())
        return f

    try:
        momenta = [
            f"k_{coordinate_names[i]}" for i in range(len(builder.symmetry.periods))
        ]
    except IndexError as err:
        raise ValueError(
            "All symmetry directions must have a name specified " "in coordinate_names"
        ) from err

    if keep is None:
        ret = WrappedBuilder()
        sym = builder.symmetry
    else:
        periods = list(builder.symmetry.periods)
        ret = WrappedBuilder(TranslationalSymmetry(periods.pop(keep)))
        sym = TranslationalSymmetry(*periods)
        momenta.pop(keep)
    momenta = tuple(momenta)
    mnp = -len(momenta)  # Used by the bound functions above.

    # Store the names of the momentum parameters and the symmetry of the
    # old Builder (this will be needed for band structure plotting)
    ret._momentum_names = momenta
    ret._wrapped_symmetry = builder.symmetry

    # Wrapped around system retains conservation law and chiral symmetry.
    # We use 'bind_site' to add the momenta arguments if required.
    cons = builder.conservation_law
    ret.conservation_law = bind_site(cons) if callable(cons) else cons
    chiral = builder.chiral
    ret.chiral = bind_site(chiral) if callable(chiral) else chiral

    if builder.particle_hole is not None or builder.time_reversal is not None:
        warnings.warn(
            "`particle_hole` and `time_reversal` symmetries are set "
            "on the input builder. However they are ignored for the "
            "wrapped system, since Kwant lacks a way to express the "
            "existence (or not) of a symmetry at k != 0.",
            RuntimeWarning,
            stacklevel=2,
        )
    ret.particle_hole = None
    ret.time_reversal = None

    ret.vectorize = builder.vectorize

    sites = {}
    hops = collections.defaultdict(list)

    # Store lists of values, so that multiple values can be assigned to the
    # same site or hopping.
    for site, val in builder.site_value_pairs():
        # Every 'site' is in the FD of the original symmetry.
        # Move the sites to the FD of the remaining symmetry, this guarantees that
        # every site in the new system is an image of an original FD site translated
        # purely by the remaining symmetry.
        sites[ret.symmetry.to_fd(site)] = [val]  # a list to append wrapped hoppings

    for hop, val in builder.hopping_value_pairs():
        a, b = hop
        # 'a' is in the FD of original symmetry.
        # Get translation from FD of original symmetry to 'b',
        # this is different from 'b_dom = sym.which(b)'.
        b_dom = builder.symmetry.which(b)
        # Throw away part that is in the remaining translation direction, so we get
        # an element of 'sym' which is being wrapped
        b_dom = ta.array([t for i, t in enumerate(b_dom) if i != keep])
        # Pull back using the remainder, which is purely in the wrapped directions.
        # This guarantees that 'b_wa' is an image of an original FD site translated
        # purely by the remaining symmetry.
        b_wa = sym.act(-b_dom, b)
        # Move the hopping to the FD of the remaining symmetry
        a, b_wa = ret.symmetry.to_fd(a, b_wa)

        if a == b_wa:
            # The hopping gets wrapped-around into an onsite Hamiltonian.
            # Since site `a` already exists in the system, we can simply append.
            sites[a].append(bind_hopping_as_site(b_dom, val))
        else:
            # The hopping remains a hopping.
            if any(b_dom):
                # The hopping got wrapped-around.
                val = bind_hopping(b_dom, val)

            # Make sure that there is only one entry for each hopping
            # pointing in one direction, modulo the remaining translations.
            b_wa_r, a_r = ret.symmetry.to_fd(b_wa, a)
            if (b_wa_r, a_r) in hops:
                if (a, b_wa) in hops:
                    raise ValueError("More than one entry per hopping.")

                if callable(val):
                    val = _callable_herm_conj(val)
                else:
                    val = herm_conj(val)

                hops[b_wa_r, a_r].append((val, b_dom))
            else:
                hops[a, b_wa].append((val, b_dom))

    # Copy stuff into result builder, converting lists of more than one element
    # into summing functions.

    # Optionally, store the wrapped sites and hoppings, and return them
    # together with the builder stripped of these sites.

    wrapped_sites = {}
    wrapped_hoppings = {}
    for site, vals in sites.items():
        if len(vals) == 1:
            # no need to bind onsites without extra wrapped hoppings
            ret[site] = vals[0]
        else:
            if separate_sites:
                ret[site] = vals[0]  # add the site without the wrapped part
                # conjugate is included by 'bind_sum'
                wrapped_sites[site] = bind_sum(1, *vals[1:])
            else:
                val = vals[0]
                vals[0] = bind_site(val) if callable(val) else val
                ret[site] = bind_sum(1, *vals)

    for hop, vals_doms in hops.items():
        if len(vals_doms) == 1:
            # no need to bind hoppings that are not already bound
            val, b_dom = vals_doms[0]

            # check if this hoppings is purely from wrapped translational symm.
            if any(b_dom) and separate_sites:
                wrapped_hoppings[hop] = val
            else:
                ret[hop] = val
        else:
            new_vals = [
                bind_hopping(b_dom, val)
                if callable(val) and not any(b_dom)  # skip hoppings already bound
                else val
                for val, b_dom in vals_doms
            ]
            if separate_sites:
                # add the hoppings that are not wrapped
                vals_in_fd = [val for val, b_dom in vals_doms if not any(b_dom)]
                if len(vals_in_fd) != 1:
                    raise ValueError("There should be only one hopping.")
                ret[hop] = vals_in_fd[0]

                vals_out_fd = [
                    val for val, b_dom in vals_doms if any(b_dom)
                ]  # vals are already bound

                wrapped_hoppings[hop] = bind_sum(2, *vals_out_fd)
            else:
                ret[hop] = bind_sum(2, *new_vals)

    if separate_sites:
        return ret, wrapped_sites, wrapped_hoppings
    return ret


def separate_bloch_components(builder):
    """Construct two systems: the supercell onsite and supercell hopping."""
    # get wrapped sites and hoppings separately
    syst_superonsite, w_sites, w_hoppings = wraparound_by_parts(
        builder, separate_sites=True
    )
    fsyst_bloch_superonsite = syst_superonsite.finalized()

    bloch_builder = Builder(symmetry=builder.symmetry)

    # the sites that are in the builder but not in the wrapped sites
    # should be added with zero value
    for s in builder.sites():
        norbs = s.family.norbs
        bloch_builder[s] = w_sites.get(s, np.zeros((norbs, norbs)))  # val or 0
    for hop, val in w_hoppings.items():
        bloch_builder[hop] = val
    fsyst_bloch_superhopping = wraparound(bloch_builder).finalized()

    return fsyst_bloch_superonsite, fsyst_bloch_superhopping


def _hopping_distance(site1, site2, direction):
    norbs = site1.family.norbs
    if norbs != site2.family.norbs:
        raise NotImplementedError(
            "Only hopppings between sites of equal number of orbitals is implemented."
        )

    if isinstance(site1, SiteArray):
        pos1 = site1.positions().transpose()
        pos2 = site2.positions().transpose()
        d = np.dot(direction, (pos1 - pos2))[:, np.newaxis, np.newaxis]
    else:
        pos1 = site1.pos
        pos2 = site2.pos
        d = np.dot(direction, (pos1 - pos2))

    # return an imaginary number so that the matrix is antisymmetric
    # but hermitian
    return 1j * d * np.identity(norbs)


def wrap_velocity(builder):
    """Construct velocity operators with translational symmetries.

    The system has an extra parameter 'direction' that should be a unit
    lenght vector in real space, of the dimensions of the site positions.

    The method 'hamiltonian_submatrix' produces the velocity operator matrix
    in the 'direction' specified.

    Note that this matrix depends on the value that takes the
    Hamiltonian of the original system, the 'builder'.
    """
    direction = ("direction",)
    dnp = -len(direction)

    @memoize
    def bind_velocity_hopping(val):
        def f(*args):
            a, b, *args = args  # first two args are sites
            d = _hopping_distance(a, b, *args[dnp:])
            # d is a diagonal of size norbs times a constant
            v = val(a, b, *args[:dnp]) if callable(val) else val
            v = np.atleast_1d(v)  # shape scalars
            return v @ d.swapaxes(-1, -2)  # transpose last two axes

        params = ("_site0", "_site1")
        if callable(val):
            params += get_parameters(val)[2:]  # cut off site parameters
        _set_signature(f, params + direction)
        return f

    velocity_builder = Builder(builder.symmetry, vectorize=builder.vectorize)

    for s in builder.sites():
        norbs = s.family.norbs
        velocity_builder[s] = np.zeros((norbs, norbs))

    for hop, val in builder.hopping_value_pairs():
        velocity_builder[hop] = bind_velocity_hopping(val)

    fsyst_velocity = wraparound(velocity_builder).finalized()

    return fsyst_velocity


def wrap_distance(builder):
    """Construct distance operators with translational symmetries.

    The system has a single parameter 'direction' that should be a unit
    lenght vector in real space, of the dimensions of the site positions.

    The method 'hamiltonian_submatrix' produces the distance operator matrix
    in the 'direction' specified.

    Note that this matrix is independent of the value that takes the
    Hamiltonian of the original system, and does not depends on the
    paramters of the original system, the 'builder'.
    """
    distance_builder = Builder(builder.symmetry, vectorize=builder.vectorize)

    for s in builder.sites():
        norbs = s.family.norbs
        distance_builder[s] = np.zeros((norbs, norbs))

    for hop, _ in builder.hopping_value_pairs():
        distance_builder[hop] = _hopping_distance

    fsyst_distance = wraparound(distance_builder).finalized()

    return fsyst_distance
