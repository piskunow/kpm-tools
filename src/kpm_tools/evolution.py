"""Evolution in time with a KPM expansion. (c) Pablo Piskunow 2021.

Evolve an initial vector to arbitrary chosen points in time
(not necessarily small intervals), and compute expectation values of
observables.


"""

from functools import partial

import numpy as np
from scipy.special import jv

from .kpm_generator import GeneralVectorExpansion
from .kpm_generator import SpectralDensityIterator
from .kpm_generator import _BaseKPM


def coef_evo(m, e, ab, delta_time):
    """Return the coefficients of the time evolution operator.

    The 'delta_time' parameter must be the real time, and will be rescaled
    with respect to the bounds of the spectrum.

    The parameter 'e' is to conform with the signatures, but it is a
    dummy parameter, as the time evolution of a vector only depends on
    the Hamiltonian and the vector.
    """
    a, b = ab
    m_array = np.arange(m)
    num_e = len(np.atleast_1d(e).flatten())
    coef = jv(m_array, a * delta_time)
    coef = 2 * coef * ((-1j) ** m_array)
    coef[0] /= 2
    coef *= np.exp(-1j * b * delta_time)
    return np.array([coef for _ in range(num_e)]).T


def _max_num_moments(dt, accuracy):
    if dt == 0:
        return 1

    def func(dt, a, b, c, d, accu):
        return int(a * np.log(1 / accu) * (dt) ** b + c * dt + d * np.log(1 / accu) + 1)

    if accuracy < 1e-16:
        accu = 1e-32  # actual accuracy goal
        p = [0.33944873, 0.3304045, 1.0063549, 0.07014769, accu]
    elif accuracy < 1e-8:
        accu = 1e-16  # actual accuracy goal
        p = [0.27909421, 0.32633684, 1.02264908, 0.08207106, accu]
    elif accuracy < 1e-4:
        accu = 1e-8  # actual accuracy goal
        p = [0.23109766, 0.31671058, 1.0528007, 0.08506937, accu]
    else:
        raise ValueError(
            "This 'accuracy' is not very accurate, " "must be 'accuracy < 1e-4'."
        )

    return func(dt, *p)


def evolve_vectors(syst, vectors, dt=0, accuracy=1e-16, **kwargs):
    """Evolve vectors in time, according to a Hamiltonian.

    Parameters
    ----------
    syst : `~kwant.system` or Hamiltonian matrix
        Hamiltonian of the system.
    vectors : iterable of arrays
        Vectors to be passed to `~SpectralDensityIterator`.
    dt : float
        Time in units of E/hbar, where E is the units
        of energy of the Hamiltonian.
    accuracy : float, default to '1e-16'
        Accuracy goal for the Chebyshev expansion.
    kwargs : dict
        Extra keyword arguments to pass to `~SpectralDensityIterator`.
    """

    def identity(x):
        return x

    kwargs["kernel"] = identity
    kwargs["vector_factory"] = vectors
    num_vectors = len(vectors)
    kwargs["num_vectors"] = num_vectors

    bounds = kwargs.get("bounds", None)
    a = np.ptp(bounds) / 2
    # bound for the minimum number of moments
    rescaled_time = a * dt
    num_moments = _max_num_moments(rescaled_time, accuracy)
    kwargs["num_moments"] = num_moments

    gvec = GeneralVectorExpansion(
        syst, coef_function=partial(coef_evo, delta_time=dt), **kwargs
    )

    return gvec(0).squeeze(0)


def evolve_vectors_time_steps(
    syst, time_steps, vectors, save_vectors=True, accuracy=1e-16, **kwargs
):
    """Evolve vectors in time using time steps.

    Parameters
    ----------
    syst : `~kwant.system` or Hamiltonian matrix
        Hamiltonian of the system, where the vectors will evolve.
    time_steps : sequence of floats
        Discrete time steps that will be used to evolve the vectors.
    vectors : ndarray
        Initial sequence of vectors that belong to the system.
    save_vectors : bool, default to 'True'
        Whether to save the vectors at intermediate steps, and return them.
    accuracy : float, default to '1e-16'
        Precission of each time-evolution step.
    kwargs : dict
        Key-word arguments to be passed to 'evolve_vectors'
    """
    params = kwargs.get("params", dict())
    bounds = kwargs.get("bounds", None)
    if bounds is None:
        base = _BaseKPM(syst, params=params, num_moments=2, num_vectors=1)
        bounds = base.bounds

    dts = np.ediff1d(time_steps)
    v_t = vectors

    final_v = [v_t]
    for dt in dts:
        v_t = evolve_vectors(
            syst, vectors=v_t, dt=dt, accuracy=accuracy, bounds=bounds, params=params
        )
        if save_vectors:
            final_v.append(v_t)
    if not save_vectors:
        final_v = v_t
    return np.array(final_v)


def evolve_observable(
    syst,
    time_steps,
    operator=None,
    vectors=None,
    accuracy=1e-16,
    save_intermediate=True,
    **kwargs
):
    """Evolve an observable in time using time steps.

    Parameters
    ----------
    syst : `~kwant.system` or Hamiltonian matrix
        Hamiltonian of the system, where the vectors will evolve.
    time_steps : sequence of floats
        Discrete time steps that will be used to evolve the vectors.
    vectors : ndarray
        Initial sequence of vectors that belong to the system.
    operator : callable, ndarray, or sparse matrix, optional
        Any operator that can be accepted by `SpectralDensityIterator`.
    accuracy : float, default to '1e-16'
        Precission of each time-evolution step.
    save_intermediate: bool, default to True
        Save intermediate vectors.
    kwargs : dict
        Key-word arguments to be passed to 'evolve_vectors'
    """
    kwargs["operator"] = operator
    kwargs.pop("vector_factory", None)

    # parameters to pass to the time evolution KPM expansion
    tkwargs = dict(bounds=kwargs.get("bounds"), params=kwargs.get("params"))

    dts = np.ediff1d(time_steps)
    v_t = vectors
    densities_t = []

    if save_intermediate:
        spectrum = SpectralDensityIterator(syst, vector_factory=v_t, **kwargs)
        densities_t.append(spectrum.densities)

    for dt in dts:
        v_t = evolve_vectors(syst, v_t, dt=dt, accuracy=accuracy, **tkwargs)
        if save_intermediate:
            spectrum = SpectralDensityIterator(syst, vector_factory=v_t, **kwargs)
            densities_t.append(spectrum.densities)

    if not save_intermediate:
        spectrum = SpectralDensityIterator(syst, vector_factory=v_t, **kwargs)
        densities_t.append(spectrum.densities)

    return densities_t
