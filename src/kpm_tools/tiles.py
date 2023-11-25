"""Generate tile vectors to compute the effective Bloch spectrum."""

from collections import defaultdict

import kwant
import numpy as np
from kwant._common import ensure_rng


def momentum_to_ts(k, ts):
    """Transform 'k' to the reciprocal lattice basis."""
    # columns of B are lattice vectors
    _b = np.array(ts.periods).T
    # columns of A are reciprocal lattice vectors
    _a = np.linalg.pinv(_b).T
    k, residuals = np.linalg.lstsq(_a, k, rcond=None)[:2]
    if np.any(abs(residuals) > 1e-7):
        raise RuntimeError(
            "Requested momentum doesn't correspond to "
            "any lattice momentum or TranslationalSymmetry."
        )
    return k


class _TileVectors:
    def __init__(self, sites, tile, ts, *, check=True):
        """Construct an iterator over the sites in 'tile'.

        The iterator returns one vector for every orbital for every site
        in the fundamental domain (FD) of 'tile'.
        Each vector contains the orbital and its copies on the other
        tiles.

        Parameters
        ----------
        sites : iterable of `kwant.builder.Site` instances
            The sites of the system. Should belong to a lattice that
            is commensurate with both 'tile' and 'ts'.
        tile :  instance of `~kwant.TranslationalSymmetry`
            Translational symmetry of the tiles within the supercell.
            The tiles should cover the supercell 'ts'.
        ts : instance of `~kwant.TranslationalSymmetry` or
            Translational symmetry of the supercell.
        check : bool, default to 'True'
            Check that 'tile' covers the 'ts'. This is done by checking
            that the translational symmetry 'tile' has the subgroup 'ts'.
            If 'False', do not check anything.

        """
        if check:
            # This does not yet checks
            # for lattice commensuration with tile nor TS.
            # The check is done when calling `tile.to_fd()`.
            if not tile.has_subgroup(ts):
                raise ValueError(f"The tile {tile} has no subgroup {ts}.")

        self.tile = tile
        self.ts = ts

        # assume all norbs are the same
        self.norbs = sites[0].family.norbs
        self.tot_norbs = self.norbs * len(sites)

        self.site_replicas = defaultdict(list)
        for idx, site in enumerate(sites):
            base = tile.to_fd(site)
            self.site_replicas[base].append(idx)

        self.keys = list(self.site_replicas.keys())

        # these counters can be reset to default values
        self.idx = 0
        self.orb_idx = 0

    def _reset_count(self):
        self.idx = 0
        self.orb_idx = 0

    def __len__(self):
        return self.norbs * len(self.keys)

    def __iter__(self):
        return self

    @property
    def _norm(self):
        return 1 / len(self)

    def __next__(self):
        try:
            keys = self.keys[self.idx]
        except IndexError as err:
            raise StopIteration(
                "Too many vectors requested " "from this generator."
            ) from err

        site_indices = np.array(self.site_replicas[keys], dtype=int)
        orbs = site_indices * self.norbs + self.orb_idx

        self.orb_idx += 1
        # bump indices if needed
        if self.orb_idx >= self.norbs:
            self.idx += 1
            self.orb_idx = 0

        return site_indices, orbs


class TileKVectors(_TileVectors):
    """Construct k-vectors over the sites in 'tile'.

    The iterator returns one vector for every orbital for every site
    in the fundamental domain (FD) of 'tile'.
    Each vector contains the orbital and its copies on the other
    tiles, multiplied with the phase given by the position and the
    'k' vector.

    Parameters
    ----------
    k : pair or triplet floats
        The vector in k-space. Should have the same dimensions as the
        positions of the 'syst'.
    tile :  instance of `~kwant.TranslationalSymmetry`
        Translational symmetry of the tiles within the supercell.
        The tiles should cover the supercell defined by the wrapped
        symmetry of 'syst'.
    syst : instance of `~kwant.system.FiniteSystem`
        System with wrapped symmetry.

    Note
    ----
    'kwargs' are passed to `_TileVectors`.

    """

    def __init__(self, k, tile, syst, **kwargs):
        """Initialize tiles."""
        self.k = k
        self.positions = np.array([s.pos for s in syst.sites])

        if isinstance(syst.symmetry, kwant.builder.NoSymmetry):
            ts = syst._wrapped_symmetry
        else:
            ts = syst.symmetry

        super().__init__(syst.sites, tile, ts, **kwargs)

    def _get_orbs_and_phase(self):
        site_indices, orbs = super().__next__()
        pos = self.positions[site_indices]
        phase = np.exp(1j * pos.dot(self.k))
        return orbs, phase

    def __next__(
        self,
    ):
        """Get the next vector."""
        orbs, phase = self._get_orbs_and_phase()

        vector = np.zeros(self.tot_norbs, dtype=complex)
        vector[orbs] = phase
        return vector


def tile_random_kvectors(k, tile, syst, *, rng=0, **kwargs):
    """Construct an iterator of random phase vectors inside 'tile'.

    The other copies of 'tile' will have a phase relative to the base
    tile. That phase is given by 'k' and the positions of the sites in
    the 'syst'. Check 'TileKVectors' for details.

    Parameters
    ----------
    k : pair or triplet floats
        The vector in k-space. Should have the same dimensions as the
        positions of the 'syst'.
    tile :  instance of `~kwant.TranslationalSymmetry`
        Translational symmetry of the tiles within the supercell.
        The tiles should cover the supercell defined by the wrapped
        symmetry of 'syst'.
    syst : instance of `~kwant.system.FiniteSystem`
        System with wrapped symmetry.
    rng : int, float, or string, default to 'int(0)'
        Initial seed for the random vectors to ensure reproducibility.

    Note
    ----
    'kwargs' are passed to `_TileVectors`.
    """
    rng = ensure_rng(rng)
    tile_kvecs = TileKVectors(k, tile, syst, **kwargs)

    while True:
        tile_kvecs._reset_count()
        vector = np.zeros(tile_kvecs.tot_norbs, dtype=complex)
        for _ in range(len(tile_kvecs)):
            orbs, phase = tile_kvecs._get_orbs_and_phase()
            vector[orbs] = phase * np.exp(2j * np.pi * rng.random_sample())
        yield vector
