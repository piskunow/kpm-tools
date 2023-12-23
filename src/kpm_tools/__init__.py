"""KPM Tools."""
# flake8: noqa
try:
    import kwant
except ImportError:
    raise ImportError(
        "Missing dependency 'kwant'. Please install it manually. "
        "Visit https://kwant-project.org/ for installation instructions."
    )
from .bloch import _hopping_distance
from .bloch import separate_bloch_components
from .bloch import wrap_distance
from .bloch import wrap_velocity
from .bloch import wraparound_by_parts
from .common import expectation_value
from .common import identity_operator
from .common import local_expectation_value
from .common import position_operator
from .common import standard_gaussian
from .common import trace_orbs
from .common import trace_unit_cell
from .common import transparent_cmap
from .common import uniform
from .evolution import _max_num_moments
from .evolution import coef_evo
from .evolution import evolve_observable
from .evolution import evolve_vectors
from .evolution import evolve_vectors_time_steps
from .hamiltonians import haldane_obc
from .hamiltonians import haldane_pbc
from .hamiltonians import qhe_obc
from .hamiltonians import qhe_pbc
from .kpm_funcs import chebysev_polynomial_series
from .kpm_funcs import chern_number
from .kpm_funcs import chern_number_pbc
from .kpm_funcs import coef_delta_function
from .kpm_funcs import coef_greens_function
from .kpm_funcs import coef_greens_function_a
from .kpm_funcs import coef_greens_function_prime
from .kpm_funcs import coef_greens_function_prime_a
from .kpm_funcs import coef_greens_function_prime_r
from .kpm_funcs import coef_greens_function_r
from .kpm_funcs import coef_projector_function
from .kpm_funcs import coef_projector_function_unoccupied
from .kpm_funcs import conductivity
from .kpm_funcs import delta_function
from .kpm_funcs import greens_function
from .kpm_funcs import longitudinal_conductivity
from .kpm_funcs import projector_operator
from .kpm_generator import GeneralVectorExpansion
from .kpm_generator import IteratorKPM
from .kpm_generator import SpectralDensityIterator
from .kpm_generator import _BaseKPM
from .kpm_generator import _dagger
from .kpm_generator import _normalize_num_moments
from .kpm_generator import _normalize_operator
from .kpm_generator import _rescale
from .kpm_generator import concatenator
from .kpm_generator import general_function
from .kpm_generator import multiprocess_update
from .kpm_generator import process_update
from .kpm_generator import thread_update
from .plotting import Arrow3D
from .plotting import plot_arrows3d
from .plotting import plot_kwant
from .plotting import plot_neighbors
from .plotting import plot_vectors
from .tiles import TileKVectors
from .tiles import _TileVectors
from .tiles import momentum_to_ts
from .tiles import tile_random_kvectors
