"""Pre-built common functions using the KPM generators."""
import numpy as np

from .common import identity_operator
from .common import position_operator
from .kpm_generator import concatenator
from .kpm_generator import general_function


def greens_function(syst, params, **kwargs):
    """Build a Green's function operator using KPM.

    Returns a function that takes an energy or a list of energies, and returns
    the Green's function with that energy acting on `vectors`.

    Parameters
    ----------
    syst : kwant.System or ndarray
        Finalized kwant system or dense or sparse ndarray of the
        Hamiltonian with shape `(N, N)`.
    params : dict, optional
        Parameters for the kwant system.
    kpm_params : dict, optional
        Dictionary containing the parameters to pass to the `~kwant.kpm`
        module. 'num_vectors' will be overwritten to match the number
        of vectors, and 'operator' key will be deleted.
    precalculate_moments: bool, default False
        Whether to precalculate and store all the KPM moments of `vectors`.
        This is useful if the Green's function is evaluated at a large
        number of energies, but uses a large amount of memory.
        If False, the KPM expansion is performed every time the Green's
        function is called, which minimizes memory use.

    Returns
    -------
    green_expansion : callable
        Takes an energy or array of energies and returns the Greens function
        acting on the vectors, for those energies.

    """
    return general_function(syst, params, coef_function=coef_greens_function, **kwargs)


def delta_function(syst, params, **kwargs):
    """Build a projector over the occupied energies.

    Returns a function that takes a Fermi energy, and returns the
    projection of the `vectors` over the occupied energies of the
    Hamiltonian.

    Parameters
    ----------
    syst : kwant.System or ndarray
        Finalized kwant system or dense or sparse ndarray of the
        Hamiltonian with shape `(N, N)`.
    params : dict, optional
        Parameters for the kwant system.
    kpm_params : dict, optional
        Dictionary containing the parameters to pass to the `~kwant.kpm`
        module. 'num_vectors' will be overwritten to match the number
        of vectors, and 'operator' key will be deleted.
    precalculate_moments: bool, default False
        Whether to precalculate and store all the KPM moments of `vectors`.
        This is useful if the Green's function is evaluated at a large
        number of energies, but uses a large amount of memory.
        If False, the KPM expansion is performed every time the Green's
        function is called, which minimizes memory use.


    Returns
    -------
    projector : callable
        Takes an energy or array of energies and returns the projection
        onto the occupied states of the vectors, for those energies.

    """
    return general_function(syst, params, coef_function=coef_delta_function, **kwargs)


def projector_operator(syst, params, **kwargs):
    """Build a projector over the occupied energies.

    Returns a function that takes a Fermi energy, and returns the
    projection of the `vectors` over the occupied energies of the
    Hamiltonian.

    Parameters
    ----------
    syst : kwant.System or ndarray
        Finalized kwant system or dense or sparse ndarray of the
        Hamiltonian with shape `(N, N)`.
    params : dict, optional
        Parameters for the kwant system.
    kpm_params : dict, optional
        Dictionary containing the parameters to pass to the `~kwant.kpm`
        module. 'num_vectors' will be overwritten to match the number
        of vectors, and 'operator' key will be deleted.
    precalculate_moments: bool, default False
        Whether to precalculate and store all the KPM moments of `vectors`.
        This is useful if the Green's function is evaluated at a large
        number of energies, but uses a large amount of memory.
        If False, the KPM expansion is performed every time the Green's
        function is called, which minimizes memory use.


    Returns
    -------
    projector : callable
        Takes an energy or array of energies and returns the projection
        onto the occupied states of the vectors, for those energies.

    """
    return general_function(
        syst, params, coef_function=coef_projector_function, **kwargs
    )


def coef_greens_function(m, e, sign=-1, ab=(1, 0)):
    """Generate Green's functions coefficients."""
    a, b = ab
    e_rescaled = (np.atleast_1d(e) - b) / a
    prefactor = -2j * sign / (np.sqrt((1 - e_rescaled) * (1 + e_rescaled)) * a)
    phi_e = np.arccos(e_rescaled)

    m_array = np.arange(m)

    coef = np.exp(1j * sign * np.outer(m_array, phi_e))

    coef[0] = coef[0] / 2

    return prefactor * coef


def coef_greens_function_r(m, e, ab=(1, 0)):
    """Generate the retarded Green's functions coefficients."""
    return coef_greens_function(m, e, ab=ab, sign=-1)


def coef_greens_function_a(m, e, ab=(1, 0)):
    """Generate the advanced Green's functions coefficients."""
    return coef_greens_function(m, e, ab=ab, sign=1)


def coef_greens_function_prime(m, e, sign=-1, ab=(1, 0)):
    """Generate the prime Green's functions coefficients."""
    a, b = ab
    e_rescaled = (np.atleast_1d(e) - b) / a
    one_e2 = (1 - e_rescaled) * (1 + e_rescaled)
    sqrt_e = np.sqrt(one_e2)
    prefactor = -2j * sign / (one_e2 * a**2)
    phi_e = np.arccos(e_rescaled)

    m_array = np.arange(m)

    coef = -1j * sign * m_array[:, np.newaxis]
    coef = coef + (e_rescaled / sqrt_e)[np.newaxis, :]
    coef = coef * np.exp(1j * sign * np.outer(m_array, phi_e))

    coef[0] = coef[0] / 2

    return prefactor * coef


def coef_greens_function_prime_r(m, e, ab=(1, 0)):
    """Generate the prime retarded Green's functions coefficients."""
    return coef_greens_function_prime(m, e, ab=ab, sign=-1)


def coef_greens_function_prime_a(m, e, ab=(1, 0)):
    """Generate the prime advanced Green's functions coefficients."""
    return coef_greens_function_prime(m, e, ab=ab, sign=1)


def coef_projector_function(m, e, ab=(1, 0)):
    """Projector over occupied states below the Fermi energy."""
    a, b = ab
    e_rescaled = (np.atleast_1d(e) - b) / a
    m_array = np.arange(m)
    m_array[0] = 1
    phi_e = np.arccos(e_rescaled)
    coef = -2 / np.pi * np.sin(np.outer(phi_e, m_array)) / m_array
    coef[:, 0] = 1 - phi_e / np.pi
    return coef.T


def coef_projector_function_unoccupied(m, e, ab=(1, 0)):
    """Projector over unoccupied states below the Fermi energy."""
    a, b = ab
    e_rescaled = (np.atleast_1d(e) - b) / a
    m_array = np.arange(m)
    m_array[0] = 1
    phi_e = np.arccos(e_rescaled)
    coef = 2 / np.pi * np.sin(np.outer(phi_e, m_array)) / m_array
    coef[:, 0] = phi_e / np.pi
    return coef.T


def coef_delta_function(m, e, ab=(1, 0)):
    """Generate the delta function coefficients."""
    coef = 2 * chebysev_polynomial_series(m, e, ab=ab)
    a, b = ab
    e_rescaled = (np.atleast_1d(e) - b) / a
    g_e = (np.pi * np.sqrt(1 - e_rescaled) * np.sqrt(1 + e_rescaled)) * a

    return coef / g_e


def chebysev_polynomial_series(m, e, ab=(1, 0)):
    """Generate the Chebyshev polynomial series."""
    # the first coefficient is halved
    a, b = ab
    e_rescaled = (np.atleast_1d(e) - b) / a
    e2 = 2 * e_rescaled
    coef = np.ndarray((m, e_rescaled.shape[0]))

    coef[0] = np.ones_like(e_rescaled)
    if m > 1:
        coef[1] = e_rescaled
        for i in range(2, m):
            coef[i] = coef[i - 1] * e2 - coef[i - 2]

    coef[0] = coef[0] / 2
    return coef


def chern_number(syst, x=None, y=None, **kwargs):
    """Compute the Chern number in real space.

    Parameters
    ----------
    syst : kwant system or Hamiltonian matrix
        The system to compute the Chern number.
    x : callable, optional
        First (second) position operator. If provided, it
        must be the '.dot' method of an ndarray. If not
        provided, the 'x' and 'y' canonical coordinates
        extraxted from 'syst' will be used. In this case,
        'syst' must be a kwant system.
    y : callable, optional
        Same as `x`.
    kwargs : dict
        The remaining keyword arguments are passed to
        `~kpm_generator.concatenator`.

    """
    # make x and y callables
    if x is None and y is None:
        x_op, y_op = position_operator(syst)
        x = x_op.dot
        y = y_op.dot
    elif x is None:
        x_op, y_op = position_operator(syst)
        x = x_op.dot
    elif y is None:
        x_op, y_op = position_operator(syst)
        y = y_op.dot

    chern = concatenator(
        syst,
        operator_array=[None, x, y, None],
        coef_array=[coef_projector_function] * 3,
        first_transpose=False,
        ouroboros=True,
        **kwargs
    )

    return lambda e: 4 * np.pi * chern(e).imag


def chern_number_pbc(syst, **kwargs):
    """Generate the Chern number for periodic boundary conditions."""
    x_op, y_op = position_operator(syst)

    chern_1 = concatenator(
        syst,
        operator_array=[None, x_op.dot, y_op.dot, None],
        coef_array=[
            coef_projector_function,
            coef_projector_function_unoccupied,
            coef_projector_function,
        ],
        first_transpose=False,
        ouroboros=True,
        **kwargs
    )

    chern_2 = concatenator(
        syst,
        operator_array=[None, y_op.dot, x_op.dot, None],
        coef_array=[
            coef_projector_function,
            coef_projector_function_unoccupied,
            coef_projector_function,
        ],
        first_transpose=False,
        ouroboros=True,
        **kwargs
    )

    return lambda e: 2j * np.pi * (chern_1(e) - chern_2(e))


def conductivity(syst, op_a="x", op_b="y", **kwargs):
    """Compute the Kubo-Bastin conductivity kernel.

    The result must be normalized with the area per site,
    and integrated up to the Fermi energy to obtain the Kubo-Bastin
    conductivity.
    """
    from kwant.kpm import _velocity

    params = kwargs.get("params", None)
    positions = kwargs.get("positions", None)

    v_a = _velocity(syst, params, op_a, positions=positions)
    v_b = _velocity(syst, params, op_b, positions=positions)

    kb_1 = concatenator(
        syst,
        operator_array=[v_a.dot, v_b.dot, identity_operator],
        coef_array=[coef_delta_function, coef_greens_function_prime_a],
        first_transpose=True,
        ouroboros=False,
        **kwargs
    )

    kb_2 = concatenator(
        syst,
        operator_array=[v_a.dot, v_b.dot, identity_operator],
        coef_array=[coef_greens_function_prime_r, coef_delta_function],
        first_transpose=True,
        ouroboros=False,
        **kwargs
    )

    return lambda e: 2 * np.pi * np.real(1j * (kb_1(e) - kb_2(e)))


def longitudinal_conductivity(syst, direction="x", **kwargs):
    """Compute the Kubo-Greenwood conductivity.

    The result must be normalized with the area per site.
    The Kubo-Greenwood KPM expansion uses of the velocity operator.
    This operator can be either provided or computed inside this function,
    check the 'direction' parameter.

    Parameters
    ----------
    syst: ndarray or a Kwant system
        System for which the velocity operator is calculated.
    params : dict
        Parametres of the system
    direction: str, matrix or operator
        If 'direction' is a `str` in {'x', 'y', 'z'}, the velocity operator
        is calculated using the 'syst' and 'positions', else
        if 'direction' is an operator or a matrix, then that is the velocity
        operator.
    positions : ndarray of shape '(N, dim)', optional
        Positions of each orbital. This parameter is not used if
        'syst' is a Kwant system.
    """
    from kwant.kpm import _velocity

    params = kwargs.get("params", None)
    positions = kwargs.get("positions", None)

    v_a = _velocity(syst, params, direction, positions=positions)

    kg = concatenator(
        syst,
        operator_array=[v_a.dot, v_a.dot, identity_operator],
        coef_array=[coef_delta_function, coef_delta_function],
        first_transpose=True,
        ouroboros=False,
        **kwargs
    )

    return kg
