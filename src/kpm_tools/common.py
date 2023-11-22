"""Utils."""

from collections import OrderedDict

import kwant
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.pylab import cm
from scipy.sparse import diags


def identity_operator(x):
    """Return argument."""
    return x


def position_operator(syst, pos_transform=None):
    """Return a list of position operators as 'csr' matrices."""
    operators = []
    norbs = syst.sites[0].family.norbs
    if pos_transform is None:
        pos = np.array([s.pos for s in syst.sites])
    else:
        pos = np.array([pos_transform(s.pos) for s in syst.sites])

    for i in range(pos.shape[1]):
        operators.append(diags(np.repeat(pos[:, i], norbs), format="csr"))
    return operators


def trace_orbs(array, norbs=4, axis=-1):
    """Return the trace per unit-cell.

    'array' has a length 'num_sites * norbs' over the 'axis',
    and the return array has a length 'num_sites' over the
    same axis.
    """
    array = np.asarray(array)
    if array.shape[axis] % norbs != 0:
        raise ValueError(f"The shape along axis {axis} must have shape {norbs}.")

    slices = [[slice(None) for i in range(array.ndim)] for i in range(norbs)]
    for n in range(norbs):
        slices[n][axis] = slice(n, None, norbs)
        slices[n] = tuple(slices[n])
    return sum([array[slices[n]] for n in range(norbs)])


def trace_unit_cell(sites, array):
    """Trace the array assuming it has the values per orbital per site.

    If
    value_per_orbital_per_site
    """
    norbs = sites[0].family.norbs
    value_per_site = trace_orbs(array, norbs, axis=0)
    norbs = sites[0].family.norbs

    trace = OrderedDict()
    for s, value in zip(sites, value_per_site, strict=True):
        _previous_value = trace.get(s.tag, 0)
        trace[s.tag] = _previous_value + value
    value_per_uc = np.array(list(trace.values()))

    return value_per_uc


def transparent_cmap(cmap=None, alpha="linear"):
    """Add transparency to a matplotlib ListedColormap.

    Parameters
    ----------
    cmap : instance of `matplotlib.pylab.cm`, optional
        The colormap to use.
    alpha : 'linear', float or array of floats
        If 'alpha' is a `float` or array of floats, this will be
        the transparency value. If it is 'linear' the transparency
        will decrease from '1' to '0'.

    """
    # Choose colormap
    if cmap is None:
        cmap = cm.viridis

    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))

    # Set alpha
    if alpha == "linear":
        my_cmap[:, -1] = np.arange(cmap.N) / cmap.N
    elif isinstance(alpha, float):
        my_cmap[:, -1] = np.ones(cmap.N) * alpha
    else:
        my_cmap[:, -1] = alpha

    # Create new colormap
    return ListedColormap(my_cmap)


# Disorder distributions
def standard_gaussian(label, salt, s1, s2=None):
    """Return a gaussian distribution."""
    return kwant.digest.gauss(str(hash((s1, s2))), salt=salt + label)


def uniform(label, salt, s1, s2=None):
    """Return a uniform distribution in the interval '[-0.5, 0.5)'."""
    return kwant.digest.uniform(str(hash((s1, s2))), salt=salt + label) - 0.5


def expectation_value(vectors, operator):
    """Braket 'operator' with 'vectors'.

    'vectors' must have t1he last axis as the dimension to be multiplied
    by 'operator'
    """
    dim = vectors.shape[-1]
    new_shape = vectors.shape[:-1]
    vecs = vectors.reshape(-1, dim)
    output = np.empty(new_shape, dtype=complex).flatten()
    for i, vec in enumerate(vecs):
        output[i] = np.dot(vec.conj(), operator.dot(vec))

    output = output.reshape(new_shape)
    return output


def local_expectation_value(vector, operator, norbs=2):
    """Trace over sites."""
    output = np.multiply(vector.conj(), operator.dot(vector.ravel()))
    return np.real_if_close(trace_orbs(output, norbs=norbs))
