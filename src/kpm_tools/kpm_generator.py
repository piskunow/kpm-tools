"""Main module containing iterables and generators of KPM vectors."""

from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from multiprocessing import Pool

import kwant
import numpy as np
from kwant.operator import _LocalOperator
from scipy.sparse import csr_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import eigsh

from .common import identity_operator


DEFAULT_MOMENTS = 100


class IteratorKPM:
    r"""Iterator of KPM vectors.

    This iterator yields vectors as

    `T_n(H) | v >`

    for vectors `| v >` in `vectors`,
    for 'n' in '[0, max_moments]'.

    Notes
    -----
    Returns a sequence of expanded vectors of shape `(M, N)`.
    If the input is a vector then `M=1`.
    """

    def __init__(self, ham, vectors, max_moments=None, operator=None, num_threads=None):
        """Initialize the iterator.

        Parameters
        ----------
        ham : 2D array
            Hamiltonian, dense or sparse array with shape '(N, N)'.
        vectors : 1D or 2D array
            Vector of length 'N' or array of vectors with shape '(M, N)'.
        max_moments : int, optional
            Number of moments to stop the iteration. If not defined, the
            iterator has no end.
        operator : callable, optioinal
            Operator to act on the expanded vectors. The iterator will yield
            'operator(v_n)', where `v_n = T_n(H) | v >`. If omited, the
            identity operator is assumed.
        num_threads : int, optional
            Number of threads to use in matrix vector multiplications.
            If not provided, the maximum number of threads are used.

        """
        self.num_threads = num_threads
        self.alpha0 = np.array(np.atleast_2d(vectors).T, dtype=complex)
        self.len = max_moments if max_moments is not None else np.inf
        self.ham = ham
        self.n = 0
        self._operator = operator
        self.alpha_prev = None
        self.alpha = None

    def operator(self, vector):
        """Normalize operator."""
        if self._operator is None:
            return vector
        return self._operator(vector)

    def __len__(
        self,
    ):
        """Maximum number of moments and vectors to yield."""
        return self.len

    def add_moments(self, num_moments=0):
        """Increase the maximum number of moments returned.

        The iterator can now return `num_moments` new expanded vectors.
        Note that it will continue after the last vector, and not start from
        the beggining.
        """
        if num_moments >= 0 and num_moments == int(num_moments):
            self.len = self.len + num_moments
        else:
            raise ValueError("'num_moments' must be a positive integer.")

    def __iter__(
        self,
    ):
        """Return instance to iterate."""
        return self

    def __next__(
        self,
    ):
        """Yield the next vector."""
        if self.n >= self.len:
            raise StopIteration("Exhausted vectors.")
        if self.n == 0:
            self.alpha_prev = np.copy(self.alpha0)
            self.n = 1
            self.alpha = self.ham @ self.alpha_prev
            return self.operator(self.alpha_prev).T
        if self.n == 1:
            self.n = 2
            return self.operator(self.alpha).T
        if self.n >= 2:
            self.n = self.n + 1
            temp = self.alpha.copy()
            self.alpha = 2 * self.ham @ self.alpha - self.alpha_prev
            self.alpha_prev = temp

            return self.operator(self.alpha).T

    # TODO : add __getitem__ method to simplify calls in other modules,
    # specially `Correlator`

    def save(self):
        """Save internal state, except the 'ham'."""
        keys = ["n", "len", "alpha0", "alpha_prev", "alpha", "_operator"]
        return {k: self.__dict__[k] for k in keys}

    def load(self, d):
        """Load internal state."""
        self.__dict__.update(d)


def thread_update(spectrum, num_moments, executor=None, shutdown=True):
    """Update thread."""
    if executor is None:
        executor = ThreadPoolExecutor()

    all_moments = list(
        executor.map(
            spectrum._update_one_vector,
            np.arange(spectrum.num_vectors),
            [spectrum.num_moments + num_moments] * spectrum.num_vectors,
        )
    )
    executor.shutdown()

    for r in range(spectrum.num_vectors):
        spectrum._moments_list[r] = all_moments[r][0]
        spectrum._iterators[r].load(all_moments[r][1])

    spectrum.num_moments += num_moments

    spectrum._update_densities()


def process_update(spectrum, num_moments):
    """Update process."""
    with ProcessPoolExecutor() as executor:
        all_moments = list(
            executor.map(
                spectrum._update_one_vector,
                np.arange(spectrum.num_vectors),
                [spectrum.num_moments + num_moments] * spectrum.num_vectors,
            )
        )

    for r in range(spectrum.num_vectors):
        spectrum._moments_list[r] = all_moments[r][0]
        spectrum._iterators[r].load(all_moments[r][1])
    spectrum.num_moments = spectrum.num_moments + num_moments

    spectrum._update_densities()


def multiprocess_update(spectrum, num_moments):
    """Update multiprocess."""
    with Pool() as executor:
        all_moments = list(
            executor.starmap(
                spectrum._update_one_vector,
                [
                    (r, spectrum.num_moments + num_moments)
                    for r in range(spectrum.num_vectors)
                ],
            )
        )

    for r in range(spectrum.num_vectors):
        spectrum._moments_list[r] = all_moments[r][0]
        spectrum._iterators[r].load(all_moments[r][1])
    spectrum.num_moments += num_moments

    spectrum._update_densities()


def _normalize_num_moments(num_moments=None, energy_resolution=None, a=1):
    if (num_moments is not None) and (energy_resolution is not None):
        raise TypeError(
            "either 'num_moments' or 'energy_resolution' " "must be provided."
        )
    if energy_resolution is not None:
        if energy_resolution <= 0:
            raise ValueError("'energy resolution' must be positive")
        num_moments = ceil((1.6 * a) / energy_resolution)
    elif num_moments is None:
        num_moments = DEFAULT_MOMENTS

    if num_moments <= 0 or num_moments != int(num_moments):
        raise ValueError("'num_moments' must be a positive integer")

    return num_moments


class _BaseKPM:
    def __init__(
        self,
        hamiltonian,
        params=None,
        operator=None,
        num_vectors=10,
        num_moments=None,
        energy_resolution=None,
        vector_factory=None,
        bounds=None,
        eps=0.05,
        rng=None,
        kernel=None,
        mean=True,
        accumulate_vectors=True,
        num_threads=1,
    ):
        self.num_threads = num_threads

        # Normalize the format of 'ham'
        if isinstance(hamiltonian, kwant.system.System):
            hamiltonian = hamiltonian.hamiltonian_submatrix(params=params, sparse=True)
        try:
            hamiltonian = csr_matrix(hamiltonian)
        except Exception as err:
            raise ValueError(
                "'hamiltonian' is neither a matrix " "nor a Kwant system."
            ) from err

        # Normalize 'operator' to a common format.
        self.operator = _normalize_operator(operator, params)

        self.mean = mean
        rng0 = kwant._common.ensure_rng(rng)

        _v0 = None
        if bounds is None:
            # create this vector for reproducibility
            _v0 = np.exp(2j * np.pi * rng0.random_sample(hamiltonian.shape[0]))

        if eps <= 0:
            raise ValueError("'eps' must be positive")

        # Hamiltonian rescaled as in Eq. (24)
        self.hamiltonian, (self._a, self._b) = _rescale(
            hamiltonian, eps=eps, v0=_v0, bounds=bounds
        )
        self.bounds = (self._b - self._a, self._b + self._a)

        if vector_factory is None:
            self._vector_factory = kwant.kpm._VectorFactory(
                kwant.kpm.RandomVectors(hamiltonian, rng=rng),
                num_vectors=num_vectors,
                accumulate=accumulate_vectors,
            )
        else:
            if not isinstance(vector_factory, Iterable):
                raise TypeError("vector_factory must be iterable")
            try:
                len(vector_factory)
            except TypeError as err:
                if num_vectors is None:
                    raise ValueError(
                        "num_vectors must be provided if"
                        "vector_factory has no length."
                    ) from err
            self._vector_factory = kwant.kpm._VectorFactory(
                vector_factory, num_vectors=num_vectors, accumulate=accumulate_vectors
            )

        self._moments_list = []
        # sets self.num_vectors = 0
        self._iterators = []

        # set kernel before calling moments
        self.kernel = kernel if kernel is not None else kwant.kpm.jackson_kernel

        self.num_moments = _normalize_num_moments(
            num_moments, energy_resolution, self._a
        )

    @property
    def num_vectors(self):
        return len(self._iterators)

    def add_vectors(self, num_vectors=None):
        """Increase the number of vectors.

        If called with no arguments, or with 'num_vectors=None', then the
        vector factory is initialized and iterated to produce the vectors.

        Parameters
        ----------
        num_vectors: positive int, optional
            The number of vectors to add.

        """
        if num_vectors is None:
            num_vectors = self._vector_factory.num_vectors
        current_vectors = self.num_vectors
        total_vectors = current_vectors + num_vectors
        new_vectors_factory = total_vectors - self._vector_factory.num_vectors
        if new_vectors_factory > 0:
            self._vector_factory.add_vectors(new_vectors_factory)
        new_vectors = self._vector_factory.num_vectors - current_vectors

        if new_vectors <= 0 or new_vectors != int(new_vectors):
            raise ValueError("'num_vectors' must be a positive integer")

        self._iterators.extend([None] * new_vectors)

        for r in range(current_vectors, total_vectors):
            self._iterators[r] = IteratorKPM(
                self.hamiltonian, self._vector_factory[r], num_threads=self.num_threads
            )

    def save(self):
        """Save state."""
        exclude = ["hamiltonian", "_iterators"]
        d = {key: self.__dict__[key] for key in self.__dict__ if key not in exclude}

        iterators_data = [ite.save() for ite in self._iterators]
        d["iterators_data"] = iterators_data
        return d

    def load(self, d):
        """Load previous state."""
        if "iterators_data" in d:
            iterators = []
            iterators_data = d.pop("iterators_data")
            for data in iterators_data:
                ite = IteratorKPM(self.hamiltonian, data["alpha0"])
                ite.load(data)
                iterators.append(ite)
            d["_iterators"] = iterators

        self.__dict__.update(d)


class SpectralDensityIterator(_BaseKPM, kwant.kpm.SpectralDensity):
    """Inherit from ~`_BaseKPM` and ~`kwant.kpm.SpectralDensity`."""

    def __init__(self, *args, **kwargs):
        """Initialize Spectral Density Iterator.

        See ~`kwant.kpm.SpectralDensity`.
        """
        super().__init__(*args, **kwargs)

        self.add_vectors()
        self._update_moments_list(self.num_moments)
        self._update_densities()

    def _update_densities(self):
        moments = self._moments()
        self.densities, self._gammas = kwant.kpm._calc_fft_moments(moments)

    def add_moments(self, num_moments=None, *, energy_resolution=None):
        """Increase the number of Chebyshev moments.

        Parameters
        ----------
        num_moments: positive int
            The number of Chebyshev moments to add. Mutually
            exclusive with `energy_resolution`.
        energy_resolution: positive float, optional
            Features wider than this resolution are visible
            in the spectral density. Mutually exclusive with
            `num_moments`.

        """
        new_moments = _normalize_num_moments(num_moments, energy_resolution, self._a)

        self._update_moments_list(self.num_moments + new_moments)
        self.num_moments += new_moments

        # recalculate quantities derived from the moments
        self._update_densities()

    def add_vectors(self, num_vectors=None):
        """Increase the number of vectors.

        Parameters
        ----------
        num_vectors: positive int, optional
            The number of vectors to add.

        """
        current_vectors = self.num_vectors
        super().add_vectors(num_vectors=num_vectors)
        new_vectors = self.num_vectors - current_vectors

        self._moments_list.extend([[]] * new_vectors)

        for r in range(current_vectors, self.num_vectors):
            one_moment, _ = self._update_one_vector(r, self.num_moments)
            self._moments_list[r] = one_moment[:]

        self._update_densities()

    def _update_moments_list(self, n_moments):
        for r in range(self.num_vectors):
            one_moment, _ = self._update_one_vector(r, n_moments)
            self._moments_list[r] = one_moment[:]

    def _update_one_vector(self, r, n_moments):
        this_iterator = self._iterators[r]
        one_moment = [None] * n_moments
        one_moment[0 : this_iterator.n] = self._moments_list[r]

        for n in range(this_iterator.n, n_moments):
            alpha_next = next(this_iterator)
            one_moment[n] = np.dot(
                this_iterator.alpha0.conj().T, self.operator(alpha_next.T)
            ).squeeze()
        return one_moment, this_iterator.save()


class GeneralVectorExpansion(_BaseKPM):
    """Inherit from ~`_BaseKPM`."""

    def __init__(self, *args, coef_function=None, **kwargs):
        """Initialize general vector expansion."""
        super().__init__(*args, **kwargs)
        self._ab = (self._a, self._b)
        self.gs = self.kernel(np.ones(self.num_moments))

        if coef_function is None:
            raise NotImplementedError
        else:
            self.coef_function = coef_function

        self.add_vectors()

    def _initialize_iterators(
        self,
    ):
        for ite in self._iterators:
            ite.n = 0

    def __call__(self, e):
        """Return the general function spectrum for energy `e`."""
        iterator = self.vector_iterator(e)
        # vecs.shape = (num_vecs, dim)
        return sum(c[:, None, None] * vecs[None, :, :] for c, vecs in iterator)

    def vector_iterator(self, e):
        """Return vector iterator."""
        self._initialize_iterators()

        e = np.atleast_1d(e).flatten()

        # coef_array is finite and makes the zip iterator to stop
        coef_array = self.coef_function(self.num_moments, e, ab=self._ab)
        coef_array = self.gs[:, None] * coef_array

        for c in coef_array:
            vectors = []
            for r in range(self.num_vectors):
                vec = next(self._iterators[r])
                # vec.shape = (num_vecs_ite, dim)
                # where num_vecs_ite == 1
                vectors.append(vec.squeeze(0))
            yield c, np.array(vectors)

    def add_moments(self, num_moments=None, *, energy_resolution=None):
        """Increase the number of Chebyshev moments.

        Parameters
        ----------
        num_moments: positive int
            The number of Chebyshev moments to add. Mutually
            exclusive with `energy_resolution`.
        energy_resolution: positive float, optional
            Features wider than this resolution are visible
            in the spectral density. Mutually exclusive with
            `num_moments`.

        """
        new_moments = _normalize_num_moments(num_moments, energy_resolution, self._a)
        self.num_moments += new_moments
        for ite in self._iterators:
            ite.len += new_moments


def general_function(*args, **kwargs):
    """Build a general function operator using KPM.

    Returns a function that takes an energy or a list of energies, and returns
    the general function with that energy acting on `vectors`.

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
    coef_function : callable, optional
        A function that takes a number of moments sequence of rescaled
        energies (to the interval '(-1, 1)'), and returns the coeficients
        of the expansion of the function.

    Returns
    -------
    f : callable
        Takes an energy or array of energies and returns the general function
        acting on the vectors, for those energies.

    """
    kpm_expansion = GeneralVectorExpansion(*args, **kwargs)
    init_vectors = np.array(kpm_expansion._vector_factory.saved_vectors)

    def f(e):
        """Return the function evaluated at 'e' applied to the vectors.

        The ndarray returned has initial dimensions '(N_E, M, N)',
        where 'N_E' is the number of energies passed, 'M' is the
        number of vectors, and 'N' is the dimension of the vectors.
        """
        vecs_in_energy = kpm_expansion(e)
        densities = np.einsum(
            "ijk,jk->ji",  # shape = (num_vectors, num_e)
            vecs_in_energy,
            init_vectors.conj(),
        )

        if kpm_expansion.mean:
            return np.mean(densities, axis=0)
        return densities

    return f


def concatenator(
    hamiltonian,
    params=None,
    operator_array=None,
    coef_array=None,
    return_expansion=False,
    first_transpose=False,
    ouroboros=False,
    logging=False,
    **kwargs
):
    """Expand a concatenation of functions and matrix operators.

    Take 'n' functions [f_i] and 'n+1' operators [O_j] and
    expand `<O_0 f_0 O_1 f_1... f_n 0_{n+1}>`.

    'O_j' is an operator acting on vectors, and [f_i] are functions
    defined by their coefficients on a KPM expansion.

    The instance created is a function. When called
    it returns the expectation value for a single energy.
    (maybe an array of energies too)

    Parameters
    ----------
    hamiltonian: ndarray or `kwant.system`
        Hamiltonian or system defining a Hamiltonian matrix.
    params: dict
        Parameters of the Hamiltonian or `kwant.system`.
    operator_array : iterable of operators
        The operators are normalized to an equivalent of a 'dot'
        function. If one of the elements in the iterable is 'None', then
        the identity operator is assumed. This identity is compatible
        with 'first_transpose'.
    coef_array: iterable of floats
        Array of coefficients for the expansion.
    return_expansion: bool, default to 'False'
        Return the expansion.
    first_transpose : bool, default to 'False'
        If there are more than one set of coefficients, then the first operator
        and set of coefficients are used to expand the vectors to the left.
        This means that the coefficients will be cojugated, and the that
        operator will be conjugate-transposed. This is achieved by different
        means, depending on the operator type. If the operator is a matrix, the
        conjugate-transpose of the matrix is taken. If it is a
        `~kwant.operator`, then this must be Hermitian. If it is a callable,
        then it must be a method from a numpy ndarray, or a scipy sparse
        matrix. In this case, the `__self__` attribute is attempted to
        retrieve, and the conjugate-transpose is taken.
        Mutually exclusive with 'ourorboros', in which case the saved expanded
        vectors to the right are the conjugate of the first expanded vectors to
        the left.
    ouroboros : bool, default to 'False'
        The first and last operators are equal, and the first and last
        functions defined by their KPM coefficients are also equal.
        Mutually exclusive with 'first_transpose'. Note that the operators and
        coefficients must nonetheless be passed, or at least place-holders.
    logging: bool, default to 'False'
        Use logging.
    Note
    ----
    The first operator in the operator array must be a matrix that can be
    transposed.

    """
    if first_transpose and ouroboros:
        raise ValueError("Only one of 'first_transpose' or 'ouroboros' can be True.")
    if len(coef_array) + 1 != len(operator_array):
        raise ValueError(
            "The length of the coefficients array must be the length of "
            "the operators minus one."
        )
    if (len(coef_array) == 1) and first_transpose:
        raise ValueError(
            "'first_transpose' cannot be true if only one operator " "is defined."
        )
    # Normalize 'operator_array' to functions that take and return
    # a vector.
    if logging:
        debug = ["First transpose is " + str(first_transpose)]
    operator_list = [
        _normalize_operator(operator_array[0], params, dagger=first_transpose)
    ]
    operator_list.extend(
        [_normalize_operator(op, params, dagger=False) for op in operator_array[1:]]
    )
    kwargs.pop("operator", None)  # no operators needed in kwargs

    # check that we accumulate the vectors from the factory
    if not kwargs.get("accumulate_vectors", True):
        raise ValueError("'accumulate_vectors' must be 'True'")
    kwargs["accumulate_vectors"] = True

    _temp = _BaseKPM(hamiltonian, params=params, **kwargs)
    _temp.add_vectors()
    # remove vector factory, since later is replaced with iterables of vectors
    kwargs.pop("vector_factory", None)

    vectors_left = np.array(_temp._vector_factory.saved_vectors)
    vectors_right = [operator_list[-1](vec) for vec in vectors_left]
    if logging:
        debug.append("vecs_right_0 = Op[-1](vecs)")

    if first_transpose:
        # first operator in the list is already dagger
        vectors_left = [operator_list[0](vec) for vec in vectors_left]
        if logging:
            debug.append("vecs_left_0 = vecs_right_0")

    # catch mean, and set it to False for the expansions
    mean = kwargs.get("mean", True)
    kwargs["mean"] = False

    if kwargs.get("bounds", None) is None:
        kwargs["bounds"] = _temp.bounds

    _psi = GeneralVectorExpansion(
        hamiltonian,
        params=params,
        coef_function=coef_array[-1],
        vector_factory=vectors_right,
        **kwargs
    )
    if logging:
        debug.append("vecs_right_1 = expand(coef[-1], vecs_right_0)")

    if len(coef_array) == 1:

        def f(e):
            e = np.atleast_1d(e)
            vecs_in_energy = _psi(e)
            # shape = (num_vectors, num_e)
            densities = np.einsum("ijk,jk->ji", vecs_in_energy, vectors_left.conj())

            if mean:
                densities = np.mean(densities, axis=0)

            if return_expansion:
                return densities, vectors_left, vecs_in_energy
            return densities

    elif len(coef_array) > 1:
        if first_transpose:
            _omega = GeneralVectorExpansion(
                hamiltonian,
                params=params,
                coef_function=_dagger(coef_array[0]),
                vector_factory=vectors_left,
                **kwargs
            )
            if logging:
                debug.append("vecs_left_1 = expand(coef[0].conj(), " "vecs_left_0)")
            remaining_coefs = coef_array[1:-1]
            remaining_operators = operator_list[1:-2]
        elif ouroboros:
            remaining_coefs = coef_array[1:-1]
            remaining_operators = operator_list[1:-2]
            if logging:
                debug.append("vecs_left_1 = vecs_right_1")
        else:
            remaining_coefs = coef_array[:-1]
            remaining_operators = operator_list[:-2]
            _vectors_left = vectors_left[np.newaxis].copy()

        if logging:
            # place-holders for next operations
            debug.extend([None, None, None])

        def f(e):
            e = np.atleast_1d(e)

            psi = _psi(e)

            if first_transpose:
                vectors_left = _omega(e)
            elif ouroboros:
                vectors_left = psi.copy()
            else:
                vectors_left = _vectors_left

            densities = []
            vecs_e_right = []
            for this_e, this_psi in zip(e, psi, strict=True):
                this_vecs_e_right = [operator_list[-2](vec) for vec in this_psi]
                if logging:
                    debug[-3] = "vecs_right_2 = Op[-2](vecs_right_1)"
                # flipping the order also flips which idx are included/excluded
                for coefs, op in zip(
                    remaining_coefs[::-1], remaining_operators[::-1], strict=True
                ):
                    _temp_expansion = GeneralVectorExpansion(
                        hamiltonian,
                        params=params,
                        coef_function=coefs,
                        vector_factory=this_vecs_e_right,
                        **kwargs
                    )
                    # shape=(num_e, num_vecs,dim)
                    _temp_vectors = _temp_expansion(this_e)[0]
                    if logging:
                        debug[-2] = "vecs_right_3 = expand(coef[-2], " "vecs_right_2)"
                    this_vecs_e_right = [op(vec) for vec in _temp_vectors]

                    if logging:
                        debug[-1] = "vecs_right_4 = Op[-3](vecs_right_3)"

                vecs_e_right.append(this_vecs_e_right)

            vecs_e_right = np.array(vecs_e_right)

            # shape=(num_vectors, num_e)
            densities = np.einsum("ijk,ijk->ji", vecs_e_right, vectors_left.conj())
            if mean:
                densities = np.mean(densities, axis=0)

            if logging:
                return densities, debug

            if return_expansion:
                return densities, vectors_left, vecs_e_right
            return densities

    return f


def _dagger(f):
    def dag(*args, **kwargs):
        return f(*args, **kwargs).conj()

    return dag


def _rescale(hamiltonian, eps, v0, bounds):
    """Rescale a Hamiltonian and return a LinearOperator.

    Parameters
    ----------
    hamiltonian : 2D array
        Hamiltonian of the system.
    eps : scalar
        Ensures that the bounds are strict.
    v0 : random vector, or None
        Used as the initial residual vector for the algorithm that
        finds the lowest and highest eigenvalues.
    bounds : tuple, or None
        Boundaries of the spectrum. If not provided the maximum and
        minimum eigenvalues are calculated.

    """
    # Relative tolerance to which to calculate eigenvalues.  Because after
    # rescaling we will add eps / 2 to the spectral bounds, we don't need
    # to know the bounds more accurately than eps / 2.
    tol = eps / 2

    if bounds:
        lmin, lmax = bounds
    else:
        lmax = float(
            eigsh(
                hamiltonian, k=1, which="LA", return_eigenvectors=False, tol=tol, v0=v0
            )
        )
        lmin = float(
            eigsh(
                hamiltonian, k=1, which="SA", return_eigenvectors=False, tol=tol, v0=v0
            )
        )

    a = np.abs(lmax - lmin) / (2.0 - eps)
    b = (lmax + lmin) / 2.0

    if lmax - lmin <= abs(lmax + lmin) * tol / 2:
        raise ValueError(
            "The Hamiltonian has a single eigenvalue, it is not possible to "
            "obtain a spectral density."
        )

    id = identity(hamiltonian.shape[0], format="csr")
    rescaled_ham = (hamiltonian - b * id) / a
    return rescaled_ham, (a, b)


def _normalize_operator(op, params, dagger=False):
    """Normalize 'op' to a function that takes and returns a vector."""
    if op is None:
        r_op = identity_operator
    elif isinstance(op, _LocalOperator):
        if dagger:
            if not op.check_hermiticity:
                raise ValueError(
                    "Cannot ensure that this operator is "
                    "Hermitian. Create this operator with "
                    "'check_hermiticity=True'"
                )
        op = op.bind(params=params)
        r_op = op.act
    elif callable(op):
        if dagger:
            try:
                r_op = op.__self__.T.conj().dot
            except Exception as err:
                raise ValueError(
                    "Cannot ensure that this operator is " "Hermitian."
                ) from err
        r_op = op
    elif hasattr(op, "dot"):
        op = csr_matrix(op)
        if dagger:
            r_op = op.T.conj().dot
        else:
            r_op = op.dot
    else:
        raise TypeError(
            "The operators must have a '.dot' " "attribute or must be callable."
        )
    return r_op
