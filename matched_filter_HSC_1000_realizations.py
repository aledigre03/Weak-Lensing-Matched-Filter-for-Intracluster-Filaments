########################################################################
#Imports and user configuration
########################################################################

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
#One numerical thread per process prevents CPU oversubscription.
for thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[thread_variable] = "1"

from pathlib import Path
import multiprocessing as mp
import re
import time
import warnings

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import astropy.units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM

import illustris_python as il
import h5py
import pyccl as ccl
import emcee
import corner

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.spatial import cKDTree


########################################################################
#Input/output locations
########################################################################

SCRIPT_DIR = Path(__file__).resolve().parent

HSC_FILE = SCRIPT_DIR / "HSC" / "999883.csv.gz"

KTNG_FOLDER = Path(
    "/vera/ptmp/gc/aldigre/kTNG/dmonly/maps"
)

KTNG_PATTERN = "LP*_run*_maps.hdf5"
KTNG_EXPECTED_LP_COUNT = 100
KTNG_EXPECTED_RUN_COUNT = 10

_KTNG_FILENAME_RE = re.compile(
    r"LP(?P<lp>\d+)_run(?P<run>\d+)_maps\.hdf5"
)


def ktng_universe_id(path):
    '''Read the LP and run numbers from a kappaTNG map filename.'''
    match = _KTNG_FILENAME_RE.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"Unexpected kappaTNG filename: {Path(path).name}")
    return int(match.group("lp")), int(match.group("run"))


def ktng_run_major_key(path):
    '''Sort maps by run number, then LP number, so the first 100 use run001.'''
    lp_number, run_number = ktng_universe_id(path)
    return run_number, lp_number


all_kTNG_files = sorted(
    KTNG_FOLDER.rglob(KTNG_PATTERN),
    key=ktng_run_major_key,
)
all_kTNG_universe_ids = [
    ktng_universe_id(path) for path in all_kTNG_files
]
expected_all_kTNG_universe_ids = [
    (lp_number, run_number)
    for run_number in range(1, KTNG_EXPECTED_RUN_COUNT + 1)
    for lp_number in range(1, KTNG_EXPECTED_LP_COUNT + 1)
]

if all_kTNG_universe_ids != expected_all_kTNG_universe_ids:
    missing_ids = sorted(
        set(expected_all_kTNG_universe_ids) - set(all_kTNG_universe_ids)
    )
    duplicate_count = (
        len(all_kTNG_universe_ids) - len(set(all_kTNG_universe_ids))
    )
    unexpected_ids = sorted(
        set(all_kTNG_universe_ids) - set(expected_all_kTNG_universe_ids)
    )
    raise RuntimeError(
        "The kappaTNG bank is not the expected 100 LP x 10 run grid. "
        f"Missing={missing_ids}; unexpected={unexpected_ids}; "
        f"duplicate_count={duplicate_count}."
    )

RESULTS_DIR = SCRIPT_DIR / "matched_filter_hsc_analytic_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


########################################################################
#Analysis configuration
########################################################################

#Use N_MC distinct kappaTNG map files and N_MC independent shape-noise draws.
#Reuse the finite set of disjoint HSC sampling windows as needed.
N_MC = 1000  #Change to 100 for the smaller ensemble.
RNG_SEED = 20260716
WINDOW_SELECTION_SEED = RNG_SEED + 1
WINDOW_SCHEDULE_SEED = RNG_SEED + 2

#Require quality-selected galaxies in every cell of the coarse coverage grid.
#This rejects windows near catalogue edges or large holes without changing their area.
WINDOW_COVERAGE_BINS = 12

#Shift the sky grid several times and keep the placement with the most complete windows.
#Require each square window to fit inside its own disjoint grid cell.
WINDOW_GRID_PHASES_PER_AXIS = 8
WINDOW_TILE_GAP_ARCMIN = 0.0

#Keep LP100_run001 as the displayed LSS realization for both ensemble sizes.
#Sorting by run first gives this map index 99 for N_MC=100 and N_MC=1000.
DATA_LP_NUMBER = 100
DATA_RUN_NUMBER = 1
DATA_SHAPE_SPAWN_INDEX = 100
DATA_BASE_WINDOW_INDEX = 0

if not 1 <= N_MC <= len(all_kTNG_files):
    raise ValueError(
        f"N_MC must lie in [1, {len(all_kTNG_files)}], found {N_MC}."
    )

kTNG_files = all_kTNG_files[:N_MC]
kTNG_universe_ids = all_kTNG_universe_ids[:N_MC]
headline_kTNG_id = (DATA_LP_NUMBER, DATA_RUN_NUMBER)

if headline_kTNG_id not in kTNG_universe_ids:
    raise RuntimeError(
        f"Headline universe {headline_kTNG_id} is absent from N_MC={N_MC}."
    )

DATA_REALIZATION_INDEX = kTNG_universe_ids.index(headline_kTNG_id)

print(f"Found {len(all_kTNG_files)} unique kappaTNG files.")
print(f"Selected {len(kTNG_files)} files for N_MC={N_MC}.")
print("First selected file:", kTNG_files[0])
print("Last selected file: ", kTNG_files[-1])
print(
    "Fixed headline file:",
    kTNG_files[DATA_REALIZATION_INDEX],
    "at realization index",
    DATA_REALIZATION_INDEX,
)

#Use a fixed intrinsic ellipticity RMS per component and the HSC response <e>=2*R*g.
SIGMA_E_INTRINSIC_COMPONENT = 0.4
SHEAR_RESPONSIVITY = 1.0 - SIGMA_E_INTRINSIC_COMPONENT**2
KTNG_MAP_SIZE_DEG = 5.0

R_CUT_MPC_H = 4.5
HC_MPC_H = 0.5

#Choose the source redshift plane for the snapshot shear plot.
#This index selects a background source plane, not the lens redshift.
PLOT_SOURCE_PLANE_INDEX = 23

APPLY_GAUSSIAN_WINDOW = False

#Supply both coordinates to include a chosen HSC centre as the displayed window.
#Otherwise select the window centres reproducibly from the catalogue.
HSC_CENTER_RA_DEG = None
HSC_CENTER_DEC_DEG = None

PRINT_EVERY = 10
SAVE_CHECKPOINT_EVERY = 25
MC_RESUME = True
MC_CHECKPOINT_PATH = (
    RESULTS_DIR / "noise_monte_carlo_checkpoint_parallel.npz"
)

#Use at most 16 worker processes within the available CPU allocation.
#A smaller worker limit can help if reading the shared kappaTNG files limits performance.
cpu_limits = [os.cpu_count() or 1]
if hasattr(os, "sched_getaffinity"):
    cpu_limits.append(len(os.sched_getaffinity(0)))
for slurm_variable in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
    slurm_value = os.environ.get(slurm_variable, "")
    slurm_match = re.match(r"\d+", slurm_value)
    if slurm_match is not None:
        cpu_limits.append(int(slurm_match.group()))
ALLOCATED_CPU_COUNT = max(1, min(cpu_limits))
MC_N_WORKERS = min(16, ALLOCATED_CPU_COUNT)
#Smooth the spatial diagnostic maps over ten snapshot pixels for display only.
#The matched-filter measurements use the unsmoothed galaxy values.
DISPLAY_SMOOTHING_PIXELS = 10.0

#Configure the MCMC settings following Shinde & Dell'Antonio (2025).
MCMC_N_WALKERS = 50
MCMC_N_STEPS = 2000
MCMC_BURN_IN = 200
MCMC_KAPPA0_BOUNDS = (0.001, 0.1)
MCMC_HC_BOUNDS_MPC = (0.01, 0.75)
MCMC_HREF_MPC = 1.5
MCMC_RANDOM_SEED = RNG_SEED + 3

########################################################################
#Helper functions
########################################################################

def rotate_coords(x, y, theta):
    '''Rotate the coordinate axes by theta in radians to get parallel and transverse offsets.'''
    ct, st = np.cos(theta), np.sin(theta)
    xp = ct * x + st * y
    yp = -st * x + ct * y
    return xp, yp


def kappa_of_h(h_coord, hc):
    '''Return the unit-amplitude filament profile at transverse distance h_coord, with width hc.'''
    return 1.0 / (1.0 + (h_coord / hc) ** 2)


def tau_template(
    theta_deg,
    x_coord_arcmin,
    y_coord_arcmin,
    hc_arcmin,
    Lfil_arcmin,
    apply_gaussian_window=False,
):
    '''
    Build the filament template at theta_deg, keeping only the forward branch.
    Optionally apply a Gaussian taper along its length.
    '''
    theta = np.deg2rad(theta_deg)
    xp, yp = rotate_coords(x_coord_arcmin, y_coord_arcmin, theta)

    tau = kappa_of_h(yp, hc_arcmin)
    tau = np.where(xp > 0.0, tau, 0.0)

    if apply_gaussian_window:
        sigma_x = Lfil_arcmin / 4.0
        tau *= np.exp(-0.5 * (xp / sigma_x) ** 2)

    return tau


def mirror_extend_2d(field):
    '''
    Mirror the field along both axes to make a grid twice as wide and tall.
    This reduces jumps at the periodic boundaries used by the Fourier transform.
    '''
    upper = np.concatenate([field, np.fliplr(field)], axis=1)
    return np.concatenate([upper, np.flipud(upper)], axis=0)


def matched_filter_from_tau_mirrored(tau, Pn_extended):
    '''
    Divide the mirrored template Fourier transform by the noise power.
    Return the real-space filter cropped to the original field and its full Fourier transform.
    '''
    ny, nx = tau.shape
    tau_extended = mirror_extend_2d(tau)

    if Pn_extended.shape != tau_extended.shape:
        raise ValueError(
            f"Pn has shape {Pn_extended.shape}, but the mirrored "
            f"template has shape {tau_extended.shape}."
        )

    psi_hat_extended = fft2(tau_extended) / Pn_extended
    psi_extended = np.real(ifft2(psi_hat_extended))
    return psi_extended[:ny, :nx], psi_hat_extended


def make_safe_power(power, floor_fraction=1e-12):
    '''Replace invalid or very small noise power values with a positive floor.'''
    safe = np.array(power, dtype=np.float64, copy=True)
    good = np.isfinite(safe) & (safe > 0.0)

    if not np.any(good):
        raise ValueError("Noise power has no finite positive values.")

    floor = floor_fraction * np.nanmedian(safe[good])
    safe[~good] = floor
    safe[safe < floor] = floor
    return safe


def sample_grid_at_galaxies(
    grid2d,
    x_axis_arcmin,
    y_axis_arcmin,
    xgal_arcmin,
    ygal_arcmin,
):
    '''
    Interpolate a grid at the galaxy positions, including the outer half-pixels.
    Return NaN for positions outside the full grid footprint.
    '''
    grid2d = np.asarray(grid2d)
    x_axis_arcmin = np.asarray(x_axis_arcmin, dtype=np.float64)
    y_axis_arcmin = np.asarray(y_axis_arcmin, dtype=np.float64)
    xgal_arcmin = np.asarray(xgal_arcmin, dtype=np.float64)
    ygal_arcmin = np.asarray(ygal_arcmin, dtype=np.float64)

    if grid2d.shape != (len(y_axis_arcmin), len(x_axis_arcmin)):
        raise ValueError("Grid and coordinate axes are inconsistent.")
    if len(x_axis_arcmin) < 2 or len(y_axis_arcmin) < 2:
        raise ValueError("At least two grid cells per axis are required.")

    dx_axis = float(x_axis_arcmin[1] - x_axis_arcmin[0])
    dy_axis = float(y_axis_arcmin[1] - y_axis_arcmin[0])
    if dx_axis <= 0.0 or dy_axis <= 0.0:
        raise ValueError("Grid coordinate axes must be increasing.")

    #Include the full pixel footprint when sampling the catalogue positions.
    #Positions in an outer half-pixel use the nearest pixel-centre value.
    valid = (
        np.isfinite(xgal_arcmin)
        & np.isfinite(ygal_arcmin)
        & (xgal_arcmin >= x_axis_arcmin[0] - 0.5 * dx_axis)
        & (xgal_arcmin <= x_axis_arcmin[-1] + 0.5 * dx_axis)
        & (ygal_arcmin >= y_axis_arcmin[0] - 0.5 * dy_axis)
        & (ygal_arcmin <= y_axis_arcmin[-1] + 0.5 * dy_axis)
    )

    sampled = np.full(xgal_arcmin.shape, np.nan, dtype=np.float64)
    if not np.any(valid):
        return sampled

    interpolator = RegularGridInterpolator(
        (y_axis_arcmin, x_axis_arcmin),
        grid2d,
        bounds_error=True,
    )
    points = np.column_stack([
        np.clip(
            ygal_arcmin[valid],
            y_axis_arcmin[0],
            y_axis_arcmin[-1],
        ),
        np.clip(
            xgal_arcmin[valid],
            x_axis_arcmin[0],
            x_axis_arcmin[-1],
        ),
    ])
    sampled[valid] = interpolator(points)
    return sampled


def centred_patch_slices(map_shape, patch_ny, patch_nx):
    '''Return the row and column slices for a central patch of a kappaTNG map.'''
    full_ny, full_nx = map_shape

    if patch_ny > full_ny or patch_nx > full_nx:
        raise ValueError(
            f"Requested patch {(patch_ny, patch_nx)} exceeds {map_shape}."
        )

    i0 = (full_ny - patch_ny) // 2
    j0 = (full_nx - patch_nx) // 2
    return slice(i0, i0 + patch_ny), slice(j0, j0 + patch_nx)


def sample_centred_patch(
    patch,
    pixel_scale_arcmin,
    xgal_arcmin,
    ygal_arcmin,
):
    '''Interpolate a map patch at galaxy positions, with the patch centre at x=y=0.'''
    ny, nx = patch.shape
    col = xgal_arcmin / pixel_scale_arcmin + (nx - 1) / 2.0
    row = ygal_arcmin / pixel_scale_arcmin + (ny - 1) / 2.0

    return map_coordinates(
        patch,
        [row, col],
        order=1,   #Use bilinear interpolation.
        mode="constant",   #Return NaN for positions outside the patch.
        cval=np.nan,
        prefilter=False,
    )


def draw_hsc_shear_noise(
    rng,
    sigma_e_measurement,
    intrinsic_component_rms,
    shear_responsivity,
):
    '''
    Draw intrinsic and measurement noise for both ellipticity components.
    Divide by twice the shear responsivity to express the noise in shear units.
    '''
    sigma_e_measurement = np.asarray(
        sigma_e_measurement,
        dtype=np.float64,
    )
    if np.any(~np.isfinite(sigma_e_measurement)):
        raise ValueError("Non-finite HSC sigma_e in shape-noise draw.")
    if np.any(sigma_e_measurement < 0.0):
        raise ValueError("HSC sigma_e cannot be negative.")
    if intrinsic_component_rms <= 0.0:
        raise ValueError("Intrinsic component RMS must be positive.")
    if not 0.0 < shear_responsivity <= 1.0:
        raise ValueError("Shear responsivity must lie in (0, 1].")

    size = sigma_e_measurement.shape
    intrinsic_1 = rng.normal(0.0, intrinsic_component_rms, size=size)
    intrinsic_2 = rng.normal(0.0, intrinsic_component_rms, size=size)
    measurement_1 = rng.normal(0.0, sigma_e_measurement, size=size)
    measurement_2 = rng.normal(0.0, sigma_e_measurement, size=size)

    inverse_response = 1.0 / (2.0 * shear_responsivity)
    return (
        (intrinsic_1 + measurement_1) * inverse_response,
        (intrinsic_2 + measurement_2) * inverse_response,
    )


def apply_filter_bank(
    e1,
    e2,
    weighted_filter,
    thetas_deg,
    valid_mask=None,
):
    '''
    Project the two input components along each filter angle and apply the weighted filters.
    Return tangential and cross responses, each normalized by sum_i w_i Psi_i, and the denominators.
    '''
    e1 = np.asarray(e1, dtype=np.float64)
    e2 = np.asarray(e2, dtype=np.float64)

    if valid_mask is None:
        filter_use = weighted_filter
        e1_use = e1
        e2_use = e2
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        e1_use = np.where(valid_mask, e1, 0.0)
        e2_use = np.where(valid_mask, e2, 0.0)

        if np.all(valid_mask):
            filter_use = weighted_filter
        else:
            filter_use = weighted_filter * valid_mask[None, :]

    denominator = np.sum(filter_use, axis=1, dtype=np.float64)
    projection_1 = filter_use @ e1_use
    projection_2 = filter_use @ e2_use

    angle_2 = 2.0 * np.deg2rad(thetas_deg)
    numerator_t = (
        -np.cos(angle_2) * projection_1
        -np.sin(angle_2) * projection_2
    )
    numerator_x = (
        np.sin(angle_2) * projection_1
        -np.cos(angle_2) * projection_2
    )

    gamma_t = np.full(len(thetas_deg), np.nan, dtype=np.float64)
    gamma_x = np.full(len(thetas_deg), np.nan, dtype=np.float64)

    safe = np.isfinite(denominator) & (np.abs(denominator) > 1e-30)
    gamma_t[safe] = numerator_t[safe] / denominator[safe]
    gamma_x[safe] = numerator_x[safe] / denominator[safe]
    return gamma_t, gamma_x, denominator


def smooth_weighted_catalogue_field(
    values,
    xgal_arcmin,
    ygal_arcmin,
    weights,
    x_edges_arcmin,
    y_edges_arcmin,
    smoothing_pixels,
):
    '''
    Bin weighted galaxy values and weights separately, then smooth both maps.
    Divide them to obtain a weighted mean field for display, masking unsupported pixels.
    '''
    numerator, _, _ = np.histogram2d(
        ygal_arcmin,
        xgal_arcmin,
        bins=[y_edges_arcmin, x_edges_arcmin],
        weights=weights * values,
    )
    denominator, _, _ = np.histogram2d(
        ygal_arcmin,
        xgal_arcmin,
        bins=[y_edges_arcmin, x_edges_arcmin],
        weights=weights,
    )

    numerator_smooth = gaussian_filter(
        numerator,
        sigma=smoothing_pixels,
        mode="constant",
    )
    denominator_smooth = gaussian_filter(
        denominator,
        sigma=smoothing_pixels,
        mode="constant",
    )

    result = np.full_like(numerator_smooth, np.nan, dtype=np.float64)
    supported = denominator_smooth > 1e-12
    result[supported] = (
        numerator_smooth[supported] / denominator_smooth[supported]
    )
    return result, denominator_smooth


########################################################################
#Reusable HSC sampling-pattern helpers
########################################################################

def radec_to_unit_vectors(ra_deg, dec_deg):
    '''Convert right ascension and declination in degrees into Cartesian unit vectors.'''
    ra_rad = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec_rad = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    cos_dec = np.cos(dec_rad)
    return np.column_stack([
        cos_dec * np.cos(ra_rad),
        cos_dec * np.sin(ra_rad),
        np.sin(dec_rad),
    ])


def tangent_offsets_arcmin(
    centre_ra_deg,
    centre_dec_deg,
    point_unit_vectors,
):
    '''Calculate spherical east and north offsets from the chosen sky centre, in arcminutes.'''
    ra0 = np.deg2rad(float(centre_ra_deg))
    dec0 = np.deg2rad(float(centre_dec_deg))

    centre = np.array([
        np.cos(dec0) * np.cos(ra0),
        np.cos(dec0) * np.sin(ra0),
        np.sin(dec0),
    ])
    east = np.array([-np.sin(ra0), np.cos(ra0), 0.0])
    north = np.array([
        -np.sin(dec0) * np.cos(ra0),
        -np.sin(dec0) * np.sin(ra0),
        np.cos(dec0),
    ])

    east_component = point_unit_vectors @ east
    north_component = point_unit_vectors @ north
    centre_component = point_unit_vectors @ centre

    delta_lon = np.arctan2(east_component, centre_component)
    delta_lat = np.arcsin(np.clip(north_component, -1.0, 1.0))

    radians_to_arcmin = u.rad.to(u.arcmin)
    return (
        delta_lon * radians_to_arcmin,
        delta_lat * radians_to_arcmin,
    )


def inverse_tangent_offsets_unit_vectors(
    centre_ra_deg,
    centre_dec_deg,
    x_offset_arcmin,
    y_offset_arcmin,
):
    '''Convert local spherical east and north offsets back into Cartesian unit vectors.'''
    ra0 = np.deg2rad(float(centre_ra_deg))
    dec0 = np.deg2rad(float(centre_dec_deg))

    centre = np.array([
        np.cos(dec0) * np.cos(ra0),
        np.cos(dec0) * np.sin(ra0),
        np.sin(dec0),
    ])
    east = np.array([-np.sin(ra0), np.cos(ra0), 0.0])
    north = np.array([
        -np.sin(dec0) * np.cos(ra0),
        -np.sin(dec0) * np.sin(ra0),
        np.cos(dec0),
    ])

    arcmin_to_rad = u.arcmin.to(u.rad)
    lon = np.asarray(x_offset_arcmin, dtype=np.float64) * arcmin_to_rad
    lat = np.asarray(y_offset_arcmin, dtype=np.float64) * arcmin_to_rad

    return (
        (np.cos(lat) * np.cos(lon))[:, None] * centre
        + (np.cos(lat) * np.sin(lon))[:, None] * east
        + np.sin(lat)[:, None] * north
    )


def spherical_window_boundary_unit_vectors(
    centre_ra_deg,
    centre_dec_deg,
    x_half_arcmin,
    y_half_arcmin,
    samples_per_edge=33,
):
    '''Sample the four edges of a square sky window and return their unit vectors.'''
    edge_parameter = np.linspace(-1.0, 1.0, samples_per_edge)

    x_boundary = np.concatenate([
        x_half_arcmin * edge_parameter,
        np.full(samples_per_edge, x_half_arcmin),
        x_half_arcmin * edge_parameter[::-1],
        np.full(samples_per_edge, -x_half_arcmin),
    ])
    y_boundary = np.concatenate([
        np.full(samples_per_edge, -y_half_arcmin),
        y_half_arcmin * edge_parameter,
        np.full(samples_per_edge, y_half_arcmin),
        y_half_arcmin * edge_parameter[::-1],
    ])

    return inverse_tangent_offsets_unit_vectors(
        centre_ra_deg,
        centre_dec_deg,
        x_boundary,
        y_boundary,
    )


def exact_window_fits_disjoint_grid_cell(
    centre_ra_deg,
    centre_dec_deg,
    x_half_arcmin,
    y_half_arcmin,
    cell_ra_width_deg,
    cell_dec_height_deg,
):
    '''
    Check that the sampled square-window boundary lies inside its assigned sky-grid cell.
    The grid cells are disjoint, so this check keeps the selected windows separated.
    '''
    boundary_vectors = spherical_window_boundary_unit_vectors(
        centre_ra_deg,
        centre_dec_deg,
        x_half_arcmin,
        y_half_arcmin,
    )

    boundary_ra_deg = (
        np.rad2deg(np.arctan2(
            boundary_vectors[:, 1],
            boundary_vectors[:, 0],
        ))
        % 360.0
    )
    boundary_dec_deg = np.rad2deg(
        np.arcsin(np.clip(boundary_vectors[:, 2], -1.0, 1.0))
    )

    delta_ra_deg = (
        (boundary_ra_deg - centre_ra_deg + 180.0) % 360.0
    ) - 180.0
    delta_dec_deg = boundary_dec_deg - centre_dec_deg

    tolerance_deg = 1e-10
    return (
        np.max(np.abs(delta_ra_deg))
        <= 0.5 * cell_ra_width_deg + tolerance_deg
        and np.max(np.abs(delta_dec_deg))
        <= 0.5 * cell_dec_height_deg + tolerance_deg
    )


HSC_SQUARE_SYMMETRY_NAMES = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "reflect_x",
    "reflect_y",
    "reflect_diagonal",
    "reflect_antidiagonal",
)


def transform_square_offsets(x, y, symmetry_index):
    '''Apply one of the eight square rotations or reflections to the galaxy positions.'''
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    symmetry_index = int(symmetry_index)

    if symmetry_index == 0:
        transformed = (x, y)
    elif symmetry_index == 1:
        transformed = (-y, x)
    elif symmetry_index == 2:
        transformed = (-x, -y)
    elif symmetry_index == 3:
        transformed = (y, -x)
    elif symmetry_index == 4:
        transformed = (-x, y)
    elif symmetry_index == 5:
        transformed = (x, -y)
    elif symmetry_index == 6:
        transformed = (y, x)
    elif symmetry_index == 7:
        transformed = (-y, -x)
    else:
        raise ValueError(
            f"Unknown square symmetry index: {symmetry_index}"
        )

    return transformed


def build_balanced_hsc_schedule(
    n_realizations,
    n_base_windows,
    rng_seed,
    data_realization_index,
    forced_data_base_window=None,
):
    '''
    Assign a base HSC window and a square symmetry to each noise realization.
    Balance window reuse and cycle through the eight symmetries before repeating them.
    Keep the displayed realization in its original orientation.
    '''
    if n_realizations < 1:
        raise ValueError("n_realizations must be positive.")
    if not 1 <= n_base_windows <= n_realizations:
        raise ValueError(
            "n_base_windows must lie in [1, n_realizations]."
        )
    if not 0 <= data_realization_index < n_realizations:
        raise ValueError("data_realization_index is outside the ensemble.")

    rng = np.random.default_rng(rng_seed)
    n_complete_blocks, remainder = divmod(
        n_realizations,
        n_base_windows,
    )

    schedule_blocks = [
        rng.permutation(n_base_windows)
        for _ in range(n_complete_blocks)
    ]
    if remainder:
        schedule_blocks.append(
            rng.permutation(n_base_windows)[:remainder]
        )

    base_window_index = np.concatenate(schedule_blocks).astype(
        np.int64,
        copy=False,
    )

    if forced_data_base_window is not None:
        forced_data_base_window = int(forced_data_base_window)
        if not 0 <= forced_data_base_window < n_base_windows:
            raise ValueError("forced_data_base_window is invalid.")
        forced_positions = np.flatnonzero(
            base_window_index == forced_data_base_window
        )
        swap_position = int(forced_positions[0])
        base_window_index[data_realization_index], base_window_index[
            swap_position
        ] = (
            base_window_index[swap_position],
            base_window_index[data_realization_index],
        )

    symmetry_index = np.empty(n_realizations, dtype=np.int8)
    n_symmetries = len(HSC_SQUARE_SYMMETRY_NAMES)

    for base_index in range(n_base_windows):
        positions = np.flatnonzero(base_window_index == base_index)
        symmetry_blocks = []
        remaining = len(positions)
        while remaining > 0:
            permutation = rng.permutation(n_symmetries)
            take = min(remaining, n_symmetries)
            symmetry_blocks.append(permutation[:take])
            remaining -= take
        symmetry_index[positions] = np.concatenate(symmetry_blocks)

    #Keep the displayed catalogue in its original orientation.
    data_base_index = base_window_index[data_realization_index]
    identity_positions = np.flatnonzero(
        (base_window_index == data_base_index)
        & (symmetry_index == 0)
    )
    if identity_positions.size:
        swap_position = int(identity_positions[0])
        symmetry_index[data_realization_index], symmetry_index[
            swap_position
        ] = (
            symmetry_index[swap_position],
            symmetry_index[data_realization_index],
        )
    else:
        symmetry_index[data_realization_index] = 0

    return base_window_index, symmetry_index


def select_non_overlapping_hsc_windows(
    catalogue_columns,
    unit_vectors,
    n_windows,
    x_half_arcmin,
    y_half_arcmin,
    z_lens,
    theta_core_arcmin,
    rng_seed,
    coverage_bins,
    grid_phases_per_axis,
    tile_gap_arcmin,
    forced_centre=None,
):
    '''
    Test shifted sky grids and keep the one with the most complete, disjoint HSC windows.
    Require coverage in every coarse cell and background sources outside the central cut.
    Select up to n_windows with a fixed seed, returning a smaller set for reuse if needed.
    '''
    if n_windows < 1:
        raise ValueError("At least one HSC base window is required.")
    if coverage_bins < 1:
        raise ValueError("coverage_bins must be positive.")
    if grid_phases_per_axis < 1:
        raise ValueError("grid_phases_per_axis must be positive.")
    if tile_gap_arcmin < 0.0:
        raise ValueError("tile_gap_arcmin cannot be negative.")

    tree = cKDTree(unit_vectors)

    width_arcmin = 2.0 * x_half_arcmin
    height_arcmin = 2.0 * y_half_arcmin
    width_deg = width_arcmin / 60.0
    height_deg = height_arcmin / 60.0
    gap_deg = tile_gap_arcmin / 60.0

    cap_radius_arcmin = np.hypot(x_half_arcmin, y_half_arcmin)
    cap_radius_rad = np.deg2rad(cap_radius_arcmin / 60.0)
    query_chord = 2.0 * np.sin(0.5 * cap_radius_rad) * (1.0 + 1e-12)

    x_coverage_edges = np.linspace(
        -x_half_arcmin,
        x_half_arcmin,
        coverage_bins + 1,
    )
    y_coverage_edges = np.linspace(
        -y_half_arcmin,
        y_half_arcmin,
        coverage_bins + 1,
    )

    z_all = catalogue_columns["z_source"]
    dec_all = catalogue_columns["dec"]
    dec_min = float(np.min(dec_all))
    dec_max = float(np.max(dec_all))

    def build_candidate(
        centre_ra_deg,
        centre_dec_deg,
        phase_y,
        phase_x,
        grid_row,
        grid_column,
        cell_ra_width_deg,
        cell_dec_height_deg,
        forced_centre_vector=None,
    ):
        '''
        Collect galaxies in one candidate window and check coverage and background sources.
        Reject empty or incomplete windows and any overlap with a manually selected window.
        '''
        centre_vector = radec_to_unit_vectors(
            [centre_ra_deg],
            [centre_dec_deg],
        )[0]

        #Use a conservative angular-separation check around a manually selected window.
        #This window is not tied to the grid, so its surrounding tiles need a separate overlap check.
        if forced_centre_vector is not None:
            separation = np.arccos(np.clip(
                centre_vector @ forced_centre_vector,
                -1.0,
                1.0,
            ))
            if separation < 2.0 * cap_radius_rad - 1e-12:
                return None

        nearby_rows = np.asarray(
            tree.query_ball_point(centre_vector, query_chord),
            dtype=np.int64,
        )
        if nearby_rows.size == 0:
            return None

        x_offset, y_offset = tangent_offsets_arcmin(
            centre_ra_deg,
            centre_dec_deg,
            unit_vectors[nearby_rows],
        )
        inside = (
            (x_offset >= -x_half_arcmin)
            & (x_offset < x_half_arcmin)
            & (y_offset >= -y_half_arcmin)
            & (y_offset < y_half_arcmin)
        )

        window_rows = nearby_rows[inside]
        x_window = x_offset[inside]
        y_window = y_offset[inside]
        if window_rows.size == 0:
            return None

        occupancy, _, _ = np.histogram2d(
            y_window,
            x_window,
            bins=[y_coverage_edges, x_coverage_edges],
        )
        coverage_fraction = float(np.mean(occupancy > 0))
        if coverage_fraction < 1.0:
            return None

        filter_sources = (
            (z_all[window_rows] > z_lens)
            & (
                np.hypot(x_window, y_window)
                >= theta_core_arcmin
            )
        )
        if not np.any(filter_sources):
            return None

        return {
            "centre_ra_deg": float(centre_ra_deg),
            "centre_dec_deg": float(centre_dec_deg),
            "centre_unit_vector": centre_vector,
            "row_indices": np.sort(window_rows),
            "coverage_fraction": coverage_fraction,
            "grid_phase_y": phase_y,
            "grid_phase_x": phase_x,
            "grid_row": grid_row,
            "grid_column": grid_column,
            "cell_ra_width_deg": cell_ra_width_deg,
            "cell_dec_height_deg": cell_dec_height_deg,
            "contained_in_disjoint_grid_cell": True,
        }

    forced_spec = None
    forced_centre_vector = None
    if forced_centre is not None:
        forced_ra = float(forced_centre[0])
        forced_dec = float(forced_centre[1])
        forced_centre_vector = radec_to_unit_vectors(
            [forced_ra],
            [forced_dec],
        )[0]

        nearby_rows = np.asarray(
            tree.query_ball_point(forced_centre_vector, query_chord),
            dtype=np.int64,
        )
        if nearby_rows.size == 0:
            raise RuntimeError(
                "The requested manual HSC centre contains no catalogue rows."
            )

        x_offset, y_offset = tangent_offsets_arcmin(
            forced_ra,
            forced_dec,
            unit_vectors[nearby_rows],
        )
        inside = (
            (x_offset >= -x_half_arcmin)
            & (x_offset < x_half_arcmin)
            & (y_offset >= -y_half_arcmin)
            & (y_offset < y_half_arcmin)
        )
        window_rows = nearby_rows[inside]
        x_window = x_offset[inside]
        y_window = y_offset[inside]

        occupancy, _, _ = np.histogram2d(
            y_window,
            x_window,
            bins=[y_coverage_edges, x_coverage_edges],
        )
        if window_rows.size == 0 or np.any(occupancy == 0):
            raise RuntimeError(
                "The requested manual HSC centre does not define a complete "
                "window."
            )

        filter_sources = (
            (z_all[window_rows] > z_lens)
            & (
                np.hypot(x_window, y_window)
                >= theta_core_arcmin
            )
        )
        if not np.any(filter_sources):
            raise RuntimeError(
                "The requested manual HSC window has no filter sources."
            )

        forced_spec = {
            "centre_ra_deg": forced_ra,
            "centre_dec_deg": forced_dec,
            "centre_unit_vector": forced_centre_vector,
            "row_indices": np.sort(window_rows),
            "coverage_fraction": float(np.mean(occupancy > 0)),
            "grid_phase_y": None,
            "grid_phase_x": None,
            "grid_row": None,
            "grid_column": None,
            "cell_ra_width_deg": None,
            "cell_dec_height_deg": None,
            "contained_in_disjoint_grid_cell": False,
        }

    dec_cell_height_deg = height_deg + gap_deg
    best_specs = []
    best_phase = None
    phases_tested = 0
    grid_centres_tested = 0

    for phase_y in range(grid_phases_per_axis):
        dec_shift_deg = (
            phase_y / grid_phases_per_axis
        ) * dec_cell_height_deg

        first_row = int(np.floor(
            (dec_min - dec_shift_deg) / dec_cell_height_deg
        )) - 1
        last_row = int(np.ceil(
            (dec_max - dec_shift_deg) / dec_cell_height_deg
        )) + 1

        for phase_x in range(grid_phases_per_axis):
            phases_tested += 1
            phase_specs = []
            phase_used_rows = np.zeros(len(unit_vectors), dtype=bool)

            if forced_spec is not None:
                phase_used_rows[forced_spec["row_indices"]] = True

            for grid_row in range(first_row, last_row + 1):
                centre_dec_deg = (
                    dec_shift_deg + grid_row * dec_cell_height_deg
                )
                if not -89.0 < centre_dec_deg < 89.0:
                    continue

                cos_dec = np.cos(np.deg2rad(centre_dec_deg))
                minimum_ra_cell_width_deg = (
                    (width_deg + gap_deg) / cos_dec
                )
                n_ra_cells = max(
                    1,
                    int(np.floor(
                        360.0 / minimum_ra_cell_width_deg
                    )),
                )
                ra_cell_width_deg = 360.0 / n_ra_cells

                #Check window containment once per declination row, where the grid geometry is shared.
                if not exact_window_fits_disjoint_grid_cell(
                    180.0,
                    centre_dec_deg,
                    x_half_arcmin,
                    y_half_arcmin,
                    ra_cell_width_deg,
                    dec_cell_height_deg,
                ):
                    continue

                ra_shift_deg = (
                    phase_x / grid_phases_per_axis
                ) * ra_cell_width_deg

                for grid_column in range(n_ra_cells):
                    centre_ra_deg = (
                        ra_shift_deg
                        + grid_column * ra_cell_width_deg
                    ) % 360.0
                    grid_centres_tested += 1

                    spec = build_candidate(
                        centre_ra_deg,
                        centre_dec_deg,
                        phase_y,
                        phase_x,
                        grid_row,
                        grid_column,
                        ra_cell_width_deg,
                        dec_cell_height_deg,
                        forced_centre_vector=forced_centre_vector,
                    )
                    if spec is None:
                        continue

                    #Reject shared catalogue rows as an additional check on the disjoint sky windows.
                    if np.any(phase_used_rows[spec["row_indices"]]):
                        continue

                    phase_specs.append(spec)
                    phase_used_rows[spec["row_indices"]] = True

            if len(phase_specs) > len(best_specs):
                best_specs = phase_specs
                best_phase = (phase_y, phase_x)

    available_windows = len(best_specs) + int(forced_spec is not None)
    if available_windows == 0:
        square_area_deg2 = width_deg * height_deg
        raise RuntimeError(
            "No complete HSC base window was found. At "
            f"z_lens={z_lens:.4f}, each window is "
            f"{width_arcmin:.2f} x {height_arcmin:.2f} arcmin "
            f"({square_area_deg2:.4f} deg^2)."
        )

    rng = np.random.default_rng(rng_seed)
    n_windows_selected = min(n_windows, available_windows)
    n_grid_windows_needed = (
        n_windows_selected - int(forced_spec is not None)
    )
    selected_indices = rng.permutation(len(best_specs))[
        :n_grid_windows_needed
    ]
    selected = [best_specs[index] for index in selected_indices]

    if forced_spec is not None:
        selected.insert(0, forced_spec)

    #Explicitly verify that no catalogue object enters two windows.
    selected_used_rows = np.zeros(len(unit_vectors), dtype=bool)
    rows_are_disjoint = True
    for spec in selected:
        rows = spec["row_indices"]
        if np.any(selected_used_rows[rows]):
            rows_are_disjoint = False
            break
        selected_used_rows[rows] = True

    if len(selected) > 1:
        centre_matrix = np.vstack([
            spec["centre_unit_vector"] for spec in selected
        ])
        dot_matrix = np.clip(
            centre_matrix @ centre_matrix.T,
            -1.0,
            1.0,
        )
        separation_matrix = np.arccos(dot_matrix)
        separation_matrix[
            np.diag_indices_from(separation_matrix)
        ] = np.inf
        minimum_selected_separation_arcmin = (
            np.rad2deg(np.min(separation_matrix)) * 60.0
        )
    else:
        minimum_selected_separation_arcmin = np.inf

    grid_windows_contained = all(
        spec["contained_in_disjoint_grid_cell"]
        for spec in selected
        if spec is not forced_spec
    )

    selection_info = {
        "selection_method": "dense_disjoint_spherical_grid",
        "grid_phases_tested": phases_tested,
        "grid_centres_tested": grid_centres_tested,
        "best_phase_y": None if best_phase is None else best_phase[0],
        "best_phase_x": None if best_phase is None else best_phase[1],
        "requested_realizations": n_windows,
        "selected_base_windows": len(selected),
        "valid_windows_available": available_windows,
        "window_reuse_required": len(selected) < n_windows,
        "tile_gap_arcmin": tile_gap_arcmin,
        "minimum_selected_separation_arcmin": (
            minimum_selected_separation_arcmin
        ),
        "rows_are_disjoint": rows_are_disjoint,
        "strictly_non_overlapping": (
            grid_windows_contained and rows_are_disjoint
        ),
    }
    return selected, selection_info

def materialize_hsc_window(
    catalogue_columns,
    unit_vectors,
    window_spec,
    background_mean_weight,
    z_lens,
    theta_core_arcmin,
    area_arcmin2,
    filter_area_arcmin2,
    symmetry_index=0,
):
    '''
    Build one HSC sampling pattern with its selected rotation or reflection.
    Return galaxy positions, weights, noise variances, source masks, and density diagnostics.
    '''
    rows = window_spec["row_indices"]
    x_window, y_window = tangent_offsets_arcmin(
        window_spec["centre_ra_deg"],
        window_spec["centre_dec_deg"],
        unit_vectors[rows],
    )
    x_window, y_window = transform_square_offsets(
        x_window,
        y_window,
        symmetry_index,
    )

    z_window = catalogue_columns["z_source"][rows].astype(
        np.float64,
        copy=False,
    )
    catalogue_weight_raw_window = catalogue_columns["weight"][rows].astype(
        np.float64,
        copy=False,
    )
    weight_raw_window = catalogue_columns["analysis_weight"][rows].astype(
        np.float64,
        copy=False,
    )
    weights_window = weight_raw_window / background_mean_weight
    shape_variance_distortion_window = catalogue_columns[
        "analysis_shape_variance_distortion"
    ][rows].astype(np.float64, copy=False)
    shape_variance_shear_window = catalogue_columns[
        "analysis_shape_variance_shear"
    ][rows].astype(np.float64, copy=False)

    mask_back_window = z_window > z_lens
    mask_front_window = ~mask_back_window
    mask_core_window = (
        np.hypot(x_window, y_window) >= theta_core_arcmin
    )
    mask_filter_window = mask_back_window & mask_core_window

    n_filter_sources = int(np.sum(mask_filter_window))
    if n_filter_sources == 0:
        raise ValueError("A selected HSC window has no filter sources.")

    sum_w_filter = np.sum(weights_window[mask_filter_window])
    sum_w2_filter = np.sum(weights_window[mask_filter_window] ** 2)
    n_eff_window = (
        sum_w_filter**2
        / (filter_area_arcmin2 * sum_w2_filter)
    )

    return {
        "row_indices": rows,
        "symmetry_index": int(symmetry_index),
        "symmetry_name": HSC_SQUARE_SYMMETRY_NAMES[int(symmetry_index)],
        "xg": x_window,
        "yg": y_window,
        "z_gal": z_window,
        "catalogue_weight_raw": catalogue_weight_raw_window,
        "weight_raw": weight_raw_window,
        "weights": weights_window,
        "shape_variance_distortion": shape_variance_distortion_window,
        "shape_variance_shear": shape_variance_shear_window,
        "shape_std_shear": np.sqrt(shape_variance_shear_window),
        "sigma_e_hsc": catalogue_columns["sigma_e"][rows],
        "rms_e_hsc": catalogue_columns["rms_e"][rows],
        "z_risk": catalogue_columns["z_risk"][rows],
        "hsc_y3_zbin": catalogue_columns["hsc_y3_zbin"][rows],
        "mask_front": mask_front_window,
        "mask_back": mask_back_window,
        "mask_source": mask_back_window.copy(),
        "mask_core": mask_core_window,
        "mask_filter_sources": mask_filter_window,
        "Ngal": len(rows),
        "N_front": int(np.sum(mask_front_window)),
        "N_back": int(np.sum(mask_back_window)),
        "N_filter_sources": n_filter_sources,
        "n_gal_raw_arcmin2": len(rows) / area_arcmin2,
        "n_back_raw_arcmin2": (
            np.sum(mask_back_window) / area_arcmin2
        ),
        "n_filter_raw_arcmin2": (
            n_filter_sources / filter_area_arcmin2
        ),
        "n_eff_arcmin2": n_eff_window,
    }


def assign_sources_to_kTNG_planes(z_gal, mask_source, z_planes):
    '''
    Assign each source to the nearest kappaTNG redshift plane.
    Clip out-of-range redshifts to the end planes and return flags identifying those sources.
    '''
    below_map = z_gal < z_planes[0]
    above_map = z_gal > z_planes[-1]
    z_clipped = np.clip(z_gal, z_planes[0], z_planes[-1])

    right = np.searchsorted(z_planes, z_clipped, side="left")
    right = np.clip(right, 1, len(z_planes) - 1)
    left = right - 1

    choose_right = (
        np.abs(z_clipped - z_planes[right])
        < np.abs(z_clipped - z_planes[left])
    )
    plane_index = np.where(choose_right, right, left)

    plane_index[z_clipped <= z_planes[0]] = 0
    plane_index[z_clipped >= z_planes[-1]] = len(z_planes) - 1
    assigned_redshift = z_planes[plane_index]

    unique_planes = np.unique(plane_index[mask_source])
    source_indices = {
        int(plane): np.flatnonzero(
            mask_source & (plane_index == plane)
        )
        for plane in unique_planes
    }
    return {
        "below_map": below_map,
        "above_map": above_map,
        "plane_index": plane_index,
        "assigned_redshift": assigned_redshift,
        "unique_planes": unique_planes,
        "source_indices": source_indices,
    }


def sample_filter_bank_at_galaxies(
    filter_grid_bank,
    x_axis_arcmin,
    y_axis_arcmin,
    xgal_arcmin,
    ygal_arcmin,
):
    '''
    Interpolate all filter grids at the galaxy positions in one HSC window.
    Use the nearest edge values in the outer half-pixels and zero outside the full footprint.
    '''
    n_angles, ny, nx = filter_grid_bank.shape
    if nx != len(x_axis_arcmin) or ny != len(y_axis_arcmin):
        raise ValueError("Filter grid and coordinate axes are inconsistent.")

    dx_axis = float(x_axis_arcmin[1] - x_axis_arcmin[0])
    dy_axis = float(y_axis_arcmin[1] - y_axis_arcmin[0])

    fx = (
        np.asarray(xgal_arcmin, dtype=np.float64)
        - x_axis_arcmin[0]
    ) / dx_axis
    fy = (
        np.asarray(ygal_arcmin, dtype=np.float64)
        - y_axis_arcmin[0]
    ) / dy_axis

    valid = (
        np.isfinite(fx)
        & np.isfinite(fy)
        & (fx >= -0.5)
        & (fx <= nx - 0.5)
        & (fy >= -0.5)
        & (fy <= ny - 0.5)
    )

    sampled = np.zeros(
        (n_angles, len(xgal_arcmin)),
        dtype=np.float32,
    )
    if not np.any(valid):
        return sampled

    fx_valid = np.clip(fx[valid], 0.0, nx - 1.0)
    fy_valid = np.clip(fy[valid], 0.0, ny - 1.0)

    ix0 = np.floor(fx_valid).astype(np.int64)
    iy0 = np.floor(fy_valid).astype(np.int64)
    ix0 = np.minimum(ix0, nx - 2)
    iy0 = np.minimum(iy0, ny - 2)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    tx = (fx_valid - ix0).astype(np.float32)
    ty = (fy_valid - iy0).astype(np.float32)

    bank_flat = filter_grid_bank.reshape(n_angles, ny * nx)
    index_00 = iy0 * nx + ix0
    index_10 = iy0 * nx + ix1
    index_01 = iy1 * nx + ix0
    index_11 = iy1 * nx + ix1

    interpolated = (
        bank_flat[:, index_00] * ((1.0 - tx) * (1.0 - ty))[None, :]
        + bank_flat[:, index_10] * (tx * (1.0 - ty))[None, :]
        + bank_flat[:, index_01] * ((1.0 - tx) * ty)[None, :]
        + bank_flat[:, index_11] * (tx * ty)[None, :]
    )
    sampled[:, valid] = interpolated.astype(np.float32, copy=False)
    return sampled


def build_weighted_filter_for_window(
    filter_grid_bank,
    x_axis_arcmin,
    y_axis_arcmin,
    window,
):
    '''
    Sample the common filters at galaxy positions and apply the analysis weights and source mask.
    Return the sampled filters, weighted filters, and their normalization sums.
    '''
    psi_gal = sample_filter_bank_at_galaxies(
        filter_grid_bank,
        x_axis_arcmin,
        y_axis_arcmin,
        window["xg"],
        window["yg"],
    )
    weights_filter = np.where(
        window["mask_filter_sources"],
        window["weights"],
        0.0,
    ).astype(np.float32)
    weighted_filter = (
        psi_gal * weights_filter[None, :]
    ).astype(np.float32, copy=False)

    denominator = np.sum(
        weighted_filter,
        axis=1,
        dtype=np.float64,
    )
    if np.any(~np.isfinite(denominator)):
        raise RuntimeError("A window has a non-finite filter denominator.")
    if np.any(np.abs(denominator) < 1e-30):
        raise RuntimeError("A window has a near-zero filter denominator.")

    return psi_gal, weighted_filter, denominator

########################################################################
#TNG snapshot, cosmology, and regular analysis grid
########################################################################

basePath = "/virgotng/universe/IllustrisTNG/TNG100-3/output/"
snapNum = 60
h = 0.6774

snap_file = f"{basePath}/snapdir_{snapNum:03d}/snap_{snapNum:03d}.0.hdf5"

with h5py.File(snap_file, "r") as f:
    header = f["Header"].attrs
    boxsize_ckpc_h = float(header["BoxSize"])
    redshift = float(header["Redshift"])
    a = float(header["Time"])
    omega_m = float(header["Omega0"])
    omega_l = float(header["OmegaLambda"])
    dm_particle_mass = float(header["MassTable"][1]) * 1e10 / h

print(f"Snapshot = {snapNum}")
print(f"z_lens = {redshift:.6f}")
print(f"a = {a:.5f}")
print(f"BoxSize = {boxsize_ckpc_h:.1f} ckpc/h")
print(f"DM particle mass = {dm_particle_mass:.3e} Msun")

cosmo_astropy = FlatLambdaCDM(
    H0=100.0 * h,
    Om0=omega_m,
    Ob0=0.0486,
)

z_lens = redshift
DA = cosmo_astropy.angular_diameter_distance(z_lens).to_value(u.Mpc)

Rcut_Mpc_h = R_CUT_MPC_H
Rcut_ckpc_h = Rcut_Mpc_h * 1000.0

Lx_ckpc_h = 2.0 * Rcut_ckpc_h
Ly_ckpc_h = 2.0 * Rcut_ckpc_h

Nx, Ny = 512, 512

dx_ckpc_h = Lx_ckpc_h / Nx
dy_ckpc_h = Ly_ckpc_h / Ny

x_ckpc_h = (
    np.arange(Nx, dtype=float) + 0.5
) * dx_ckpc_h - Rcut_ckpc_h
y_ckpc_h = (
    np.arange(Ny, dtype=float) + 0.5
) * dy_ckpc_h - Rcut_ckpc_h

xg_ckpc_h, yg_ckpc_h = np.meshgrid(
    x_ckpc_h,
    y_ckpc_h,
    indexing="xy",
)

#Convert comoving ckpc/h coordinates to physical kpc.
x_kpc = x_ckpc_h / h / (1.0 + z_lens)
y_kpc = y_ckpc_h / h / (1.0 + z_lens)
xg_kpc, yg_kpc = np.meshgrid(x_kpc, y_kpc, indexing="xy")

Lx_kpc = Lx_ckpc_h / h / (1.0 + z_lens)
Ly_kpc = Ly_ckpc_h / h / (1.0 + z_lens)

dx_kpc = dx_ckpc_h / h / (1.0 + z_lens)
dy_kpc = dy_ckpc_h / h / (1.0 + z_lens)

arcmin_per_rad = (u.rad).to(u.arcmin)

x_arcmin = (x_kpc / 1000.0) / DA * arcmin_per_rad
y_arcmin = (y_kpc / 1000.0) / DA * arcmin_per_rad

dx_arcmin = (dx_kpc / 1000.0) / DA * arcmin_per_rad
dy_arcmin = (dy_kpc / 1000.0) / DA * arcmin_per_rad

xg_arcmin, yg_arcmin = np.meshgrid(
    x_arcmin,
    y_arcmin,
    indexing="xy",
)

field_width_x_arcmin = Nx * dx_arcmin
field_width_y_arcmin = Ny * dy_arcmin
A_arcmin2 = field_width_x_arcmin * field_width_y_arcmin

hc_Mpc_h = HC_MPC_H
hc_Mpc = hc_Mpc_h / h / (1.0 + z_lens)
hc_arcmin = (hc_Mpc / DA) * arcmin_per_rad

Lfil_kpc = (Rcut_ckpc_h / 2.0) / h / (1.0 + z_lens)
Lfil_Mpc = Lfil_kpc / 1000.0
Lfil_arcmin = (Lfil_Mpc / DA) * arcmin_per_rad

dtheta_deg = 1.0
thetas_deg = np.arange(0.0, 360.0, dtheta_deg)
N_angles = len(thetas_deg)

print("Grid:", Nx, "x", Ny)
print(f"Cut half-size = {Rcut_Mpc_h:.2f} Mpc/h")
print(f"Pixel size = {dx_ckpc_h:.2f} ckpc/h = {dx_kpc:.2f} kpc")
print(f"Pixel size = {dx_arcmin:.4f} arcmin")
print(
    f"Full pixel footprint = {field_width_x_arcmin:.3f} x "
    f"{field_width_y_arcmin:.3f} arcmin"
)
print(f"Geometric field area = {A_arcmin2:.3f} arcmin^2")
print(f"DA(z_lens) = {DA:.3f} Mpc")
print(f"hc = {hc_Mpc_h:.3f} Mpc/h = {hc_arcmin:.3f} arcmin")
print(f"Lfil = {Lfil_arcmin:.3f} arcmin")

########################################################################
#Load the central halo, dark matter particles, and subhaloes
########################################################################

halos = il.groupcat.loadHalos(
    basePath,
    snapNum,
    fields=["GroupPos", "Group_M_Crit200", "Group_R_Crit200"],
)

idx = 10
halo_pos = halos["GroupPos"][idx]

print("Selected halo index:", idx)
print(
    "Halo M200:",
    halos["Group_M_Crit200"][idx] * 1e10 / h,
    "Msun",
)

coords = il.snapshot.loadSubset(
    basePath,
    snapNum,
    "dm",
    fields=["Coordinates"],
)

subhalos = il.groupcat.loadSubhalos(
    basePath,
    snapNum,
    fields=["SubhaloPos"],
)
sub_pos = subhalos

########################################################################
#Central-halo radius and periodic particle selection
########################################################################

halo_R200_ckpc_h = float(halos["Group_R_Crit200"][idx])
halo_R200_kpc = halo_R200_ckpc_h / h / (1.0 + z_lens)
halo_R200_Mpc = halo_R200_kpc / 1000.0
theta_arcmin_halo = ((halo_R200_Mpc / DA) * arcmin_per_rad) *0.5

print(f"R200 = {halo_R200_Mpc:.3f} physical Mpc")
print(f"R200 angular size = {theta_arcmin_halo:.3f} arcmin")


def periodic_delta(position, centre, boxsize):
    '''Return offsets from the centre using the nearest periodic image of each position.'''
    delta = position - centre
    delta -= boxsize * np.round(delta / boxsize)
    return delta


dx_dm = periodic_delta(coords[:, 0], halo_pos[0], boxsize_ckpc_h)
dy_dm = periodic_delta(coords[:, 1], halo_pos[1], boxsize_ckpc_h)
dz_dm = periodic_delta(coords[:, 2], halo_pos[2], boxsize_ckpc_h)

mask_dm = (
    (np.abs(dx_dm) < Rcut_ckpc_h)
    & (np.abs(dy_dm) < Rcut_ckpc_h)
    & (np.abs(dz_dm) < Rcut_ckpc_h)
)

coords_cut = np.column_stack(
    [dx_dm[mask_dm], dy_dm[mask_dm], dz_dm[mask_dm]]
)

print("DM particles in the 20 Mpc/h cube:", coords_cut.shape[0])

########################################################################
#Load, clean, and construct the reusable HSC sampling-pattern bank
########################################################################

if not HSC_FILE.exists():
    raise FileNotFoundError(
        f"HSC catalogue not found: {HSC_FILE.resolve()}"
    )

hsc_raw = pd.read_csv(HSC_FILE, skipinitialspace=True)

#Remove the leading comment marker from the HSC object_id header.
#Strip surrounding whitespace from every column name.
hsc_raw.columns = (
    pd.Index(hsc_raw.columns)
    .astype(str)
    .str.strip()
    .str.removeprefix("#")
    .str.strip()
)

if hsc_raw.columns.duplicated().any():
    duplicate_columns = hsc_raw.columns[hsc_raw.columns.duplicated()].tolist()
    raise ValueError(f"Duplicate HSC columns after cleaning: {duplicate_columns}")

required_columns = {
    "object_id",
    "ra",
    "dec",
    "z_source",
    "z_risk",
    "weight",
    "sigma_e",
    "rms_e",
    "hsc_y3_zbin",
    "b_mode_mask",
}
missing_columns = required_columns.difference(hsc_raw.columns)

if missing_columns:
    raise KeyError(
        "The HSC catalogue is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )

numeric_columns = [
    "ra",
    "dec",
    "z_source",
    "z_risk",
    "weight",
    "sigma_e",
    "rms_e",
    "hsc_y3_zbin",
    "b_mode_mask",
]
for column in numeric_columns:
    hsc_raw[column] = pd.to_numeric(hsc_raw[column], errors="coerce")

finite_columns = [
    "ra",
    "dec",
    "z_source",
    "weight",
    "sigma_e",
    "rms_e",
]

finite_mask = np.ones(len(hsc_raw), dtype=bool)
for column in finite_columns:
    finite_mask &= np.isfinite(hsc_raw[column].to_numpy(dtype=float))

quality_mask = (
    finite_mask
    & (hsc_raw["weight"].to_numpy(dtype=float) > 0.0)
    & (hsc_raw["b_mode_mask"].to_numpy(dtype=float) == 1.0)
)

hsc_catalogue = hsc_raw.loc[quality_mask].reset_index(drop=True)

print("HSC rows read:", len(hsc_raw))
print(
    "Rows after finite/weight/b_mode_mask cuts:",
    len(hsc_catalogue),
)

if len(hsc_catalogue) == 0:
    raise ValueError("No HSC objects survive the catalogue cuts.")
if not 0 <= DATA_REALIZATION_INDEX < N_MC:
    raise ValueError("DATA_REALIZATION_INDEX must lie in [0, N_MC).")
if tuple(kTNG_universe_ids[DATA_REALIZATION_INDEX]) != headline_kTNG_id:
    raise ValueError(
        "DATA_REALIZATION_INDEX must identify the headline (LP, run)."
    )
if not 0 <= DATA_SHAPE_SPAWN_INDEX <= N_MC:
    raise ValueError("DATA_SHAPE_SPAWN_INDEX must lie in [0, N_MC].")
if not 0.0 < SHEAR_RESPONSIVITY <= 1.0:
    raise ValueError("The configured HSC shear responsivity is invalid.")

if (HSC_CENTER_RA_DEG is None) != (HSC_CENTER_DEC_DEG is None):
    raise ValueError(
        "Supply both HSC_CENTER_RA_DEG and HSC_CENTER_DEC_DEG, or neither."
    )

hsc_catalogue_columns = {
    column: hsc_catalogue[column].to_numpy()
    for column in required_columns
}

#Use an intrinsic ellipticity RMS of 0.4 per component for the simulated shapes.
#Recompute inverse-variance weights from this dispersion and each HSC measurement error.
#Keep the supplied HSC weights and rms_e values for catalogue diagnostics.
sigma_e_measurement_all = hsc_catalogue_columns["sigma_e"].astype(
    np.float64,
    copy=False,
)
analysis_shape_variance_distortion = (
    SIGMA_E_INTRINSIC_COMPONENT**2
    + sigma_e_measurement_all**2
)
analysis_shape_variance_shear = (
    analysis_shape_variance_distortion
    / (2.0 * SHEAR_RESPONSIVITY) ** 2
)
hsc_catalogue_columns["analysis_shape_variance_distortion"] = (
    analysis_shape_variance_distortion
)
hsc_catalogue_columns["analysis_shape_variance_shear"] = (
    analysis_shape_variance_shear
)
hsc_catalogue_columns["analysis_weight"] = (
    1.0 / analysis_shape_variance_distortion
)
hsc_unit_vectors = radec_to_unit_vectors(
    hsc_catalogue_columns["ra"],
    hsc_catalogue_columns["dec"],
)

#Use the full pixel footprint, not merely the range between pixel centres.
x_half_arcmin = 0.5 * field_width_x_arcmin
y_half_arcmin = 0.5 * field_width_y_arcmin

#The central circular exclusion is fully contained in every square field.
A_filter_arcmin2 = A_arcmin2 - np.pi * theta_arcmin_halo**2

forced_centre = None
if HSC_CENTER_RA_DEG is not None:
    forced_centre = (
        float(HSC_CENTER_RA_DEG),
        float(HSC_CENTER_DEC_DEG),
    )

hsc_window_specs, window_selection_info = (
    select_non_overlapping_hsc_windows(
        hsc_catalogue_columns,
        hsc_unit_vectors,
        N_MC,
        x_half_arcmin,
        y_half_arcmin,
        z_lens,
        theta_arcmin_halo,
        WINDOW_SELECTION_SEED,
        WINDOW_COVERAGE_BINS,
        WINDOW_GRID_PHASES_PER_AXIS,
        WINDOW_TILE_GAP_ARCMIN,
        forced_centre=forced_centre,
    )
)

N_HSC_BASE_WINDOWS = len(hsc_window_specs)

#Use each base window equally often, allowing a difference of at most one realization.
#Keep the displayed window at its configured base index with no rotation or reflection.
#A manually selected centre is stored as base window zero.
if not 0 <= DATA_BASE_WINDOW_INDEX < N_HSC_BASE_WINDOWS:
    raise ValueError("DATA_BASE_WINDOW_INDEX is outside the base bank.")
forced_data_base_window = DATA_BASE_WINDOW_INDEX
(
    realization_base_window_index,
    realization_symmetry_index,
) = build_balanced_hsc_schedule(
    N_MC,
    N_HSC_BASE_WINDOWS,
    WINDOW_SCHEDULE_SEED,
    DATA_REALIZATION_INDEX,
    forced_data_base_window=forced_data_base_window,
)
realization_symmetry_name = np.asarray([
    HSC_SQUARE_SYMMETRY_NAMES[index]
    for index in realization_symmetry_index
])
window_use_counts = np.bincount(
    realization_base_window_index,
    minlength=N_HSC_BASE_WINDOWS,
)

base_window_center_ra_deg = np.array([
    spec["centre_ra_deg"] for spec in hsc_window_specs
])
base_window_center_dec_deg = np.array([
    spec["centre_dec_deg"] for spec in hsc_window_specs
])
window_center_ra_deg = base_window_center_ra_deg[
    realization_base_window_index
]
window_center_dec_deg = base_window_center_dec_deg[
    realization_base_window_index
]

#Divide all analysis weights by the pooled mean background weight for numerical conditioning.
#This common factor cancels from the estimator, effective density, and shape-noise power.
pooled_background_weight_sum = 0.0
pooled_background_count = 0

for base_window_index, spec in enumerate(hsc_window_specs):
    rows = spec["row_indices"]
    background = hsc_catalogue_columns["z_source"][rows] > z_lens
    reuse_count = window_use_counts[base_window_index]
    pooled_background_weight_sum += reuse_count * np.sum(
        hsc_catalogue_columns["analysis_weight"][rows][background]
    )
    pooled_background_count += reuse_count * int(np.sum(background))

if pooled_background_count == 0:
    raise ValueError("No background sources exist in the selected windows.")

background_mean_weight = (
    pooled_background_weight_sum / pooled_background_count
)
if (
    not np.isfinite(background_mean_weight)
    or background_mean_weight <= 0.0
):
    raise ValueError("Invalid pooled mean background HSC weight.")

base_window_Ngal = np.zeros(N_HSC_BASE_WINDOWS, dtype=int)
base_window_N_front = np.zeros(N_HSC_BASE_WINDOWS, dtype=int)
base_window_N_back = np.zeros(N_HSC_BASE_WINDOWS, dtype=int)
base_window_N_filter_sources = np.zeros(
    N_HSC_BASE_WINDOWS,
    dtype=int,
)
base_window_n_gal_raw_arcmin2 = np.zeros(
    N_HSC_BASE_WINDOWS,
    dtype=float,
)
base_window_n_back_raw_arcmin2 = np.zeros(
    N_HSC_BASE_WINDOWS,
    dtype=float,
)
base_window_n_filter_raw_arcmin2 = np.zeros(
    N_HSC_BASE_WINDOWS,
    dtype=float,
)
base_window_n_eff_arcmin2 = np.zeros(
    N_HSC_BASE_WINDOWS,
    dtype=float,
)
base_window_mean_shape_std_shear = np.zeros(
    N_HSC_BASE_WINDOWS,
    dtype=float,
)

pooled_filter_sum_w = 0.0
pooled_filter_sum_w2 = 0.0
pooled_filter_sum_w2_shape_var_shear = 0.0

for base_window_index, spec in enumerate(hsc_window_specs):
    window_state = materialize_hsc_window(
        hsc_catalogue_columns,
        hsc_unit_vectors,
        spec,
        background_mean_weight,
        z_lens,
        theta_arcmin_halo,
        A_arcmin2,
        A_filter_arcmin2,
    )

    base_window_Ngal[base_window_index] = window_state["Ngal"]
    base_window_N_front[base_window_index] = window_state["N_front"]
    base_window_N_back[base_window_index] = window_state["N_back"]
    base_window_N_filter_sources[base_window_index] = (
        window_state["N_filter_sources"]
    )
    base_window_n_gal_raw_arcmin2[base_window_index] = (
        window_state["n_gal_raw_arcmin2"]
    )
    base_window_n_back_raw_arcmin2[base_window_index] = (
        window_state["n_back_raw_arcmin2"]
    )
    base_window_n_filter_raw_arcmin2[base_window_index] = (
        window_state["n_filter_raw_arcmin2"]
    )
    base_window_n_eff_arcmin2[base_window_index] = (
        window_state["n_eff_arcmin2"]
    )

    filter_mask = window_state["mask_filter_sources"]
    filter_weights = window_state["weights"][filter_mask]
    filter_shape_variance_shear = window_state[
        "shape_variance_shear"
    ][filter_mask]
    base_window_mean_shape_std_shear[base_window_index] = np.mean(
        np.sqrt(filter_shape_variance_shear)
    )
    reuse_count = window_use_counts[base_window_index]
    pooled_filter_sum_w += reuse_count * np.sum(filter_weights)
    pooled_filter_sum_w2 += reuse_count * np.sum(filter_weights**2)
    pooled_filter_sum_w2_shape_var_shear += reuse_count * np.sum(
        filter_weights**2 * filter_shape_variance_shear
    )

n_eff_arcmin2 = (
    pooled_filter_sum_w**2
    / (
        N_MC
        * A_filter_arcmin2
        * pooled_filter_sum_w2
    )
)
if not np.isfinite(n_eff_arcmin2) or n_eff_arcmin2 <= 0.0:
    raise ValueError("The pooled effective source density is invalid.")

effective_shape_variance_shear = (
    pooled_filter_sum_w2_shape_var_shear / pooled_filter_sum_w2
)
effective_shape_std_shear = np.sqrt(effective_shape_variance_shear)
if (
    not np.isfinite(effective_shape_std_shear)
    or effective_shape_std_shear <= 0.0
):
    raise ValueError("The pooled shear-unit shape RMS is invalid.")

#Realization-level diagnostics inherit the properties of their base window.
window_Ngal = base_window_Ngal[realization_base_window_index]
window_N_front = base_window_N_front[realization_base_window_index]
window_N_back = base_window_N_back[realization_base_window_index]
window_N_filter_sources = base_window_N_filter_sources[
    realization_base_window_index
]
window_n_gal_raw_arcmin2 = base_window_n_gal_raw_arcmin2[
    realization_base_window_index
]
window_n_back_raw_arcmin2 = base_window_n_back_raw_arcmin2[
    realization_base_window_index
]
window_n_filter_raw_arcmin2 = base_window_n_filter_raw_arcmin2[
    realization_base_window_index
]
window_n_eff_arcmin2 = base_window_n_eff_arcmin2[
    realization_base_window_index
]
window_mean_shape_std_shear = base_window_mean_shape_std_shear[
    realization_base_window_index
]

#Select the HSC sampling pattern used for the snapshot diagnostics and displayed observation.
data_base_window_index = int(
    realization_base_window_index[DATA_REALIZATION_INDEX]
)
data_symmetry_index = int(
    realization_symmetry_index[DATA_REALIZATION_INDEX]
)
data_symmetry_name = HSC_SQUARE_SYMMETRY_NAMES[data_symmetry_index]

data_window = materialize_hsc_window(
    hsc_catalogue_columns,
    hsc_unit_vectors,
    hsc_window_specs[data_base_window_index],
    background_mean_weight,
    z_lens,
    theta_arcmin_halo,
    A_arcmin2,
    A_filter_arcmin2,
    symmetry_index=data_symmetry_index,
)

data_rows = data_window["row_indices"]
hsc = hsc_catalogue.iloc[data_rows].copy().reset_index(drop=True)
hsc["x_arcmin"] = data_window["xg"]
hsc["y_arcmin"] = data_window["yg"]

hsc_center_ra_deg = base_window_center_ra_deg[data_base_window_index]
hsc_center_dec_deg = base_window_center_dec_deg[data_base_window_index]
hsc_center = SkyCoord(
    ra=hsc_center_ra_deg * u.deg,
    dec=hsc_center_dec_deg * u.deg,
    frame="icrs",
)

xg = data_window["xg"]
yg = data_window["yg"]
z_gal = data_window["z_gal"]
catalogue_weight_raw = data_window["catalogue_weight_raw"]
weight_raw = data_window["weight_raw"]
weights = data_window["weights"]
sigma_e_hsc = data_window["sigma_e_hsc"]
rms_e_hsc = data_window["rms_e_hsc"]
shape_variance_distortion = data_window["shape_variance_distortion"]
shape_variance_shear = data_window["shape_variance_shear"]
shape_std_shear = data_window["shape_std_shear"]
z_risk = data_window["z_risk"]
hsc_y3_zbin = data_window["hsc_y3_zbin"]

mask_front = data_window["mask_front"]
mask_back = data_window["mask_back"]
mask_source = data_window["mask_source"]
mask_core = data_window["mask_core"]
mask_filter_sources = data_window["mask_filter_sources"]

Ngal = data_window["Ngal"]
N_front = data_window["N_front"]
N_back = data_window["N_back"]
N_filter_sources = data_window["N_filter_sources"]

n_gal_raw_arcmin2 = data_window["n_gal_raw_arcmin2"]
n_back_raw_arcmin2 = data_window["n_back_raw_arcmin2"]
n_filter_raw_arcmin2 = data_window["n_filter_raw_arcmin2"]
n_eff_data_arcmin2 = data_window["n_eff_arcmin2"]

r_gal_arcmin = np.hypot(xg, yg)

print("Strictly non-overlapping HSC base windows:", N_HSC_BASE_WINDOWS)
print("Scheduled simulated universes:", N_MC)
print(
    "Base-window reuse count, min/max:",
    np.min(window_use_counts),
    np.max(window_use_counts),
)
print(
    "Valid complete windows in best grid phase:",
    window_selection_info["valid_windows_available"],
)
print(
    "Best grid phase (Dec, RA):",
    (
        window_selection_info["best_phase_y"],
        window_selection_info["best_phase_x"],
    ),
)
print(
    "Grid phases/centres tested:",
    window_selection_info["grid_phases_tested"],
    window_selection_info["grid_centres_tested"],
)
print(
    "Tile safety gap:",
    window_selection_info["tile_gap_arcmin"],
    "arcmin",
)
print(
    "Minimum selected centre separation:",
    f"{window_selection_info['minimum_selected_separation_arcmin']:.3f} "
    "arcmin",
)
print(
    "Selected catalogue rows are disjoint:",
    window_selection_info["rows_are_disjoint"],
)
print(
    "Data HSC window mapped to halo: "
    f"realization={DATA_REALIZATION_INDEX}, "
    f"base_window={data_base_window_index}, "
    f"symmetry={data_symmetry_name}, "
    f"RA={hsc_center_ra_deg:.8f} deg, "
    f"Dec={hsc_center_dec_deg:.8f} deg"
)
print("Data-window galaxies:", Ngal)
print("Data-window foreground galaxies:", N_front)
print("Data-window background sources:", N_back)
print("Data-window background sources outside R200:", N_filter_sources)
print(
    "Ellipticity model: Gaussian intrinsic component RMS =",
    SIGMA_E_INTRINSIC_COMPONENT,
)
print(f"HSC shear responsivity = {SHEAR_RESPONSIVITY:.6f}")
print(
    "Data-window measurement sigma_e, min/median/max:",
    np.min(sigma_e_hsc[mask_filter_sources]),
    np.median(sigma_e_hsc[mask_filter_sources]),
    np.max(sigma_e_hsc[mask_filter_sources]),
)
print(
    "Data-window shear-unit component noise, min/median/max:",
    np.min(shape_std_shear[mask_filter_sources]),
    np.median(shape_std_shear[mask_filter_sources]),
    np.max(shape_std_shear[mask_filter_sources]),
)
print(f"Data-window raw density = {n_gal_raw_arcmin2:.4f} arcmin^-2")
print(
    f"Data-window weighted effective density = "
    f"{n_eff_data_arcmin2:.4f} arcmin^-2"
)
print(
    f"Common pooled weighted effective density = "
    f"{n_eff_arcmin2:.4f} arcmin^-2"
)
print(
    "Ngal across scheduled realizations, min/median/max:",
    np.min(window_Ngal),
    np.median(window_Ngal),
    np.max(window_Ngal),
)
print("Data-window z_source min/max:", np.min(z_gal), np.max(z_gal))
print("z_risk is diagnostic only; no z_risk cut is applied.")

########################################################################
#HSC catalogue diagnostics
########################################################################

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].scatter(xg, yg, s=2, alpha=0.4)
axes[0].set_xlim(-x_half_arcmin, x_half_arcmin)
axes[0].set_ylim(-y_half_arcmin, y_half_arcmin)
axes[0].set_aspect("equal")
axes[0].set_xlabel("x [arcmin]")
axes[0].set_ylabel("y [arcmin]")
axes[0].set_title("Paired HSC data-window positions")

axes[1].hist(
    z_gal[mask_back],
    bins=50,
    weights=weights[mask_back],
    histtype="step",
    linewidth=1.5,
)
axes[1].axvline(
    z_lens,
    linestyle="--",
    color="k",
    label=f"z_lens = {z_lens:.3f}",
)
axes[1].set_xlabel("z_source")
axes[1].set_ylabel("weighted count")
axes[1].set_title("Weighted background-source distribution")
axes[1].legend()

analysis_variance_proxy = (
    SIGMA_E_INTRINSIC_COMPONENT**2 + sigma_e_hsc**2
)
axes[2].scatter(
    analysis_variance_proxy,
    weight_raw,
    s=3,
    alpha=0.35,
    label="analysis weight",
)
axes[2].scatter(
    analysis_variance_proxy,
    catalogue_weight_raw,
    s=3,
    alpha=0.20,
    label="supplied HSC weight",
)
axes[2].set_xlabel(r"$0.4^2+\mathrm{sigma\_e}^2$")
axes[2].set_ylabel("inverse-variance weight")
axes[2].set_yscale("log")
axes[2].set_title("Noise-model weight diagnostic")
axes[2].legend(markerscale=3)

plt.tight_layout()
fig.savefig(
    RESULTS_DIR / "hsc_catalogue_diagnostics.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)

print(
    "z_risk percentiles:",
    np.nanpercentile(z_risk, [0, 16, 50, 84, 100]),
)

########################################################################
#Validate the selected kappaTNG map realizations
########################################################################

z_kTNG = np.array([
    3.446699999999999764e-02, 6.951499999999999346e-02,
    1.051819999999999977e-01, 1.415099999999999969e-01,
    1.785450000000000093e-01, 2.163319999999999965e-01,
    2.549210000000000087e-01, 2.943629999999999858e-01,
    3.347129999999999828e-01, 3.760270000000000001e-01,
    4.183680000000000176e-01, 4.617990000000000150e-01,
    5.063870000000000315e-01, 5.522049999999999459e-01,
    5.993290000000000006e-01, 6.478380000000000249e-01,
    6.978180000000000494e-01, 7.493610000000000548e-01,
    8.025609999999999689e-01, 8.575220000000000065e-01,
    9.143520000000000536e-01, 9.731680000000000330e-01,
    1.034092999999999929e+00, 1.097259999999999902e+00,
    1.162811000000000039e+00, 1.230895999999999990e+00,
    1.301677999999999891e+00, 1.375329999999999941e+00,
    1.452039999999999997e+00, 1.532006999999999897e+00,
    1.615448999999999913e+00, 1.702596999999999916e+00,
    1.793703000000000047e+00, 1.889037999999999995e+00,
    1.988895999999999997e+00, 2.093595999999999790e+00,
    2.203481000000000023e+00, 2.318928000000000100e+00,
    2.440344000000000069e+00, 2.568175000000000097e+00,
])

kTNG_groups = [f"z{i:02d}" for i in range(len(z_kTNG))]

if len(kTNG_files) != N_MC:
    raise RuntimeError(
        f"Expected {N_MC} selected kappaTNG files, found {len(kTNG_files)}."
    )

kTNG_universe_ids = np.asarray(
    [ktng_universe_id(path) for path in kTNG_files],
    dtype=int,
)
kTNG_lp_numbers = kTNG_universe_ids[:, 0]
kTNG_run_numbers = kTNG_universe_ids[:, 1]
expected_kTNG_universe_ids = np.asarray(
    expected_all_kTNG_universe_ids[:N_MC],
    dtype=int,
)

if not np.array_equal(kTNG_universe_ids, expected_kTNG_universe_ids):
    raise RuntimeError(
        "Selected kappaTNG files are not the expected run-major sequence."
    )

if len({tuple(item) for item in kTNG_universe_ids}) != N_MC:
    raise RuntimeError("Duplicate (LP, run) kappaTNG universes were selected.")

if tuple(kTNG_universe_ids[DATA_REALIZATION_INDEX]) != headline_kTNG_id:
    raise RuntimeError("The configured headline (LP, run) was not selected.")

with h5py.File(kTNG_files[0], "r") as f:
    for group in kTNG_groups:
        for dataset in ("kappa", "gamma1", "gamma2"):
            key = f"{group}/{dataset}"
            if key not in f:
                raise KeyError(f"Missing dataset {key} in {kTNG_files[0]}")

    map_shape = tuple(f[f"{kTNG_groups[0]}/kappa"].shape)

if len(map_shape) != 2:
    raise ValueError(f"Expected 2D kappaTNG maps, found shape {map_shape}.")

kTNG_pixel_scale_arcmin = (
    KTNG_MAP_SIZE_DEG * 60.0 / map_shape[1]
)

#A small interpolation margin and odd dimensions give a unique patch centre.
N_patch_x = int(np.ceil(field_width_x_arcmin / kTNG_pixel_scale_arcmin)) + 3
N_patch_y = int(np.ceil(field_width_y_arcmin / kTNG_pixel_scale_arcmin)) + 3

if N_patch_x % 2 == 0:
    N_patch_x += 1
if N_patch_y % 2 == 0:
    N_patch_y += 1

patch_y_slice, patch_x_slice = centred_patch_slices(
    map_shape,
    N_patch_y,
    N_patch_x,
)

patch_half_x_arcmin = 0.5 * (N_patch_x - 1) * kTNG_pixel_scale_arcmin
patch_half_y_arcmin = 0.5 * (N_patch_y - 1) * kTNG_pixel_scale_arcmin

if (
    x_half_arcmin > patch_half_x_arcmin
    or y_half_arcmin > patch_half_y_arcmin
):
    raise RuntimeError("The central kappaTNG patch does not cover the HSC field.")

print("kappaTNG files used:", len(kTNG_files))
print("First file:", kTNG_files[0])
print("Last file:", kTNG_files[-1])
print(
    "Validated universe range:",
    tuple(kTNG_universe_ids[0]),
    "to",
    tuple(kTNG_universe_ids[-1]),
)
print(
    "Fixed headline LSS universe:",
    f"LP{DATA_LP_NUMBER:03d}_run{DATA_RUN_NUMBER:03d}",
)
print("Full map shape:", map_shape)
print(f"kappaTNG pixel scale = {kTNG_pixel_scale_arcmin:.5f} arcmin/pixel")
print("Fixed central patch shape:", (N_patch_y, N_patch_x))
print("Each selected (LP, run) file contributes one fixed central line of sight.")

########################################################################
#Assign every HSC base window to kappaTNG source planes
########################################################################

all_source_plane_usage = np.zeros(len(z_kTNG), dtype=bool)
total_sources_below_map = 0
total_sources_above_map = 0

for base_window_index, spec in enumerate(hsc_window_specs):
    window_state = materialize_hsc_window(
        hsc_catalogue_columns,
        hsc_unit_vectors,
        spec,
        background_mean_weight,
        z_lens,
        theta_arcmin_halo,
        A_arcmin2,
        A_filter_arcmin2,
    )
    assignment = assign_sources_to_kTNG_planes(
        window_state["z_gal"],
        window_state["mask_source"],
        z_kTNG,
    )
    all_source_plane_usage[assignment["unique_planes"]] = True
    reuse_count = window_use_counts[base_window_index]
    total_sources_below_map += reuse_count * int(np.sum(
        assignment["below_map"] & window_state["mask_source"]
    ))
    total_sources_above_map += reuse_count * int(np.sum(
        assignment["above_map"] & window_state["mask_source"]
    ))

all_source_plane_indices = np.flatnonzero(all_source_plane_usage)

#Assign source planes for the displayed HSC window.
data_plane_assignment = assign_sources_to_kTNG_planes(
    z_gal,
    mask_source,
    z_kTNG,
)
below_map = data_plane_assignment["below_map"]
above_map = data_plane_assignment["above_map"]
idx_kTNG_gal = data_plane_assignment["plane_index"]
z_kTNG_gal = data_plane_assignment["assigned_redshift"]
unique_source_planes = data_plane_assignment["unique_planes"]
plane_source_indices = data_plane_assignment["source_indices"]

print("Source planes used by all base windows:", all_source_plane_indices)
print("Data-window source planes:", unique_source_planes)
print(
    "Data-window assigned source-plane z range:",
    z_kTNG_gal[mask_source].min(),
    z_kTNG_gal[mask_source].max(),
)
print(
    "Total background sources below map range:",
    total_sources_below_map,
)
print(
    "Total background sources above map range:",
    total_sources_above_map,
)

if total_sources_above_map > 0:
    warnings.warn(
        "Some selected HSC sources lie above the highest kappaTNG plane "
        "and are assigned to z=2.568175."
    )

########################################################################
#Snapshot convergence and shear for the required source planes
########################################################################

if not 0 <= PLOT_SOURCE_PLANE_INDEX < len(z_kTNG):
    raise ValueError("PLOT_SOURCE_PLANE_INDEX is outside the kappaTNG grid.")

D_l = cosmo_astropy.angular_diameter_distance(z_lens)

Sigma_crit_kTNG = np.full(len(z_kTNG), np.inf)
kappa_snap_planes = {}
gamma1_snap_planes = {}
gamma2_snap_planes = {}

x_dm_cut = coords_cut[:, 0]
y_dm_cut = coords_cut[:, 1]

x_edges_ckpc_h = np.linspace(-Rcut_ckpc_h, Rcut_ckpc_h, Nx + 1)
y_edges_ckpc_h = np.linspace(-Rcut_ckpc_h, Rcut_ckpc_h, Ny + 1)

H, _, _ = np.histogram2d(
    x_dm_cut,
    y_dm_cut,
    bins=[x_edges_ckpc_h, y_edges_ckpc_h],
)
H = H.T

pixel_area_kpc2_physical = (
    dx_ckpc_h / h / (1.0 + z_lens)
) * (
    dy_ckpc_h / h / (1.0 + z_lens)
)

Sigma = H * dm_particle_mass / pixel_area_kpc2_physical

dx_rad = np.deg2rad(dx_arcmin / 60.0)
dy_rad = np.deg2rad(dy_arcmin / 60.0)

kx_1d = 2.0 * np.pi * fftfreq(Nx, d=dx_rad)
ky_1d = 2.0 * np.pi * fftfreq(Ny, d=dy_rad)
kx, ky = np.meshgrid(kx_1d, ky_1d, indexing="xy")

k2 = kx**2 + ky**2
k2[0, 0] = 1.0

snapshot_plane_indices = np.unique(
    np.r_[all_source_plane_indices, PLOT_SOURCE_PLANE_INDEX]
)

for plane_idx in snapshot_plane_indices:
    zs = z_kTNG[plane_idx]

    if zs <= z_lens:
        kappa_snap_planes[plane_idx] = np.zeros((Ny, Nx), dtype=np.float32)
        gamma1_snap_planes[plane_idx] = np.zeros((Ny, Nx), dtype=np.float32)
        gamma2_snap_planes[plane_idx] = np.zeros((Ny, Nx), dtype=np.float32)
        continue

    D_s = cosmo_astropy.angular_diameter_distance(zs)
    D_ls = cosmo_astropy.angular_diameter_distance_z1z2(z_lens, zs)

    Sigma_crit_i = (
        const.c**2 / (4.0 * np.pi * const.G)
    ) * (
        D_s / (D_l * D_ls)
    )
    Sigma_crit_i = Sigma_crit_i.to_value(u.Msun / u.kpc**2)
    Sigma_crit_kTNG[plane_idx] = Sigma_crit_i

    kappa_i = Sigma / Sigma_crit_i
    kappa_i -= np.mean(kappa_i)

    kappa_hat = fft2(kappa_i)
    gamma1_hat = ((kx**2 - ky**2) / k2) * kappa_hat
    gamma2_hat = ((2.0 * kx * ky) / k2) * kappa_hat
    gamma1_hat[0, 0] = 0.0
    gamma2_hat[0, 0] = 0.0

    kappa_snap_planes[plane_idx] = kappa_i.astype(np.float32)
    gamma1_snap_planes[plane_idx] = np.real(ifft2(gamma1_hat)).astype(np.float32)
    gamma2_snap_planes[plane_idx] = np.real(ifft2(gamma2_hat)).astype(np.float32)

finite_sigma_crit = Sigma_crit_kTNG[np.isfinite(Sigma_crit_kTNG)]
print("Snapshot lensing planes ready:", len(snapshot_plane_indices))
print(
    "Sigma_crit finite min/max:",
    finite_sigma_crit.min(),
    finite_sigma_crit.max(),
)

########################################################################
#Snapshot shear visualization
########################################################################

plot_plane_idx = PLOT_SOURCE_PLANE_INDEX

gamma1_plot = gamma1_snap_planes[plot_plane_idx]
gamma2_plot = gamma2_snap_planes[plot_plane_idx]
gamma_mag = np.hypot(gamma1_plot, gamma2_plot)

step = 10
xw_kpc = xg_kpc[::step, ::step]
yw_kpc = yg_kpc[::step, ::step]

g1w = gamma1_plot[::step, ::step]
g2w = gamma2_plot[::step, ::step]
gw = np.hypot(g1w, g2w)

phi = 0.5 * np.arctan2(g2w, g1w)

gw_clip = np.percentile(gw, 99)
gw = np.minimum(gw, gw_clip)
gw_norm = gw / max(np.max(gw), 1e-30)

L0_kpc = 350.0
whisker_length = L0_kpc * (0.05 + 0.95 * gw_norm**0.25)
whisker_dx = 0.5 * whisker_length * np.cos(phi)
whisker_dy = 0.5 * whisker_length * np.sin(phi)

segments = []
for iy in range(xw_kpc.shape[0]):
    for ix in range(xw_kpc.shape[1]):
        segments.append([
            (
                xw_kpc[iy, ix] - whisker_dx[iy, ix],
                yw_kpc[iy, ix] - whisker_dy[iy, ix],
            ),
            (
                xw_kpc[iy, ix] + whisker_dx[iy, ix],
                yw_kpc[iy, ix] + whisker_dy[iy, ix],
            ),
        ])

#Full pixel footprint, including half a pixel beyond the outer centres.
extent_kpc = [
    -0.5 * Lx_kpc,
    0.5 * Lx_kpc,
    -0.5 * Ly_kpc,
    0.5 * Ly_kpc,
]

fig, ax = plt.subplots(figsize=(7, 7))

im = ax.imshow(
    gamma_mag,
    origin="lower",
    extent=extent_kpc,
    cmap="viridis",
    vmin=np.percentile(gamma_mag, 5),
    vmax=np.percentile(gamma_mag, 99),
    aspect="equal",
    interpolation="bilinear",
)

whiskers = LineCollection(
    segments,
    colors="black",
    linewidths=0.60,
    alpha=0.75,
    clip_on=True,
)
ax.add_collection(whiskers)
whiskers.set_clip_path(ax.patch)

ax.scatter(0.0, 0.0, c="cyan", s=25, marker="*")
ax.set_xlim(extent_kpc[0], extent_kpc[1])
ax.set_ylim(extent_kpc[2], extent_kpc[3])
ax.set_xlabel("x [kpc]")
ax.set_ylabel("y [kpc]")
ax.set_title(
    f"Snapshot shear: lens z={z_lens:.3f}, "
    f"source plane z={z_kTNG[plot_plane_idx]:.3f}"
)

plt.colorbar(im, ax=ax, fraction=0.046, label=r"$|\gamma|$")
plt.tight_layout()
fig.savefig(
    RESULTS_DIR / "snapshot_shear_whiskers.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

########################################################################
#Sample the fixed snapshot signal at the paired data-window positions
########################################################################

gamma1_snapshot_gal = np.zeros(Ngal, dtype=np.float32)
gamma2_snapshot_gal = np.zeros(Ngal, dtype=np.float32)
kappa_snapshot_gal = np.zeros(Ngal, dtype=np.float32)

for plane_idx, source_indices in plane_source_indices.items():
    gamma1_snapshot_gal[source_indices] = sample_grid_at_galaxies(
        gamma1_snap_planes[plane_idx],
        x_arcmin,
        y_arcmin,
        xg[source_indices],
        yg[source_indices],
    )
    gamma2_snapshot_gal[source_indices] = sample_grid_at_galaxies(
        gamma2_snap_planes[plane_idx],
        x_arcmin,
        y_arcmin,
        xg[source_indices],
        yg[source_indices],
    )
    kappa_snapshot_gal[source_indices] = sample_grid_at_galaxies(
        kappa_snap_planes[plane_idx],
        x_arcmin,
        y_arcmin,
        xg[source_indices],
        yg[source_indices],
    )

if not (
    np.all(np.isfinite(gamma1_snapshot_gal[mask_source]))
    and np.all(np.isfinite(gamma2_snapshot_gal[mask_source]))
    and np.all(np.isfinite(kappa_snapshot_gal[mask_source]))
):
    raise RuntimeError(
        "Some HSC source positions fall outside the snapshot grid."
    )

print("Snapshot lensing fields sampled at HSC positions.")
print("Filter sources:", int(np.sum(mask_filter_sources)))

########################################################################
#Common analytic nonlinear LSS convergence power C_kappa(ell)
########################################################################

Omega_b_ccl = 0.0486
sigma8_ccl = 0.8159
n_s_ccl = 0.9667

cosmo_ccl = ccl.Cosmology(
    Omega_c=omega_m - Omega_b_ccl,
    Omega_b=Omega_b_ccl,
    h=h,
    sigma8=sigma8_ccl,
    n_s=n_s_ccl,
    m_nu=0.0,
    transfer_function="boltzmann_camb",
    matter_power_spectrum="camb",
    extra_parameters={
        "camb": {
            "halofit_version": "mead",
            "kmax": 100.0,
        }
    },
)

#Pool the weighted source-redshift distributions using each window reuse count.
#Use this common distribution to calculate the LSS power for the full filter ensemble.
nz_edges = np.linspace(0.0, z_kTNG.max(), 161)
nz_weighted = np.zeros(len(nz_edges) - 1, dtype=np.float64)

for base_window_index, spec in enumerate(hsc_window_specs):
    window_state = materialize_hsc_window(
        hsc_catalogue_columns,
        hsc_unit_vectors,
        spec,
        background_mean_weight,
        z_lens,
        theta_arcmin_halo,
        A_arcmin2,
        A_filter_arcmin2,
    )
    z_for_window = np.clip(
        window_state["z_gal"][window_state["mask_filter_sources"]],
        0.0,
        z_kTNG.max(),
    )
    weights_for_window = window_state["weights"][
        window_state["mask_filter_sources"]
    ]
    window_histogram, _ = np.histogram(
        z_for_window,
        bins=nz_edges,
        weights=weights_for_window,
    )
    nz_weighted += (
        window_use_counts[base_window_index] * window_histogram
    )

z_nz = 0.5 * (nz_edges[:-1] + nz_edges[1:])
dz_nz = np.diff(nz_edges)
dndz_weighted = nz_weighted / dz_nz

nz_norm = np.trapezoid(dndz_weighted, z_nz)
if not np.isfinite(nz_norm) or nz_norm <= 0.0:
    raise RuntimeError("The pooled weighted HSC n(z) cannot be normalized.")
dndz_weighted /= nz_norm

wl_tracer = ccl.WeakLensingTracer(
    cosmo_ccl,
    dndz=(z_nz, dndz_weighted),
    has_shear=True,
)

ell_nyquist_x = np.pi / dx_rad
ell_nyquist_y = np.pi / dy_rad
ell_max = 1.05 * np.hypot(ell_nyquist_x, ell_nyquist_y)

ell_theory = np.unique(
    np.ceil(np.geomspace(2.0, ell_max, 2500)).astype(int)
)

#Calculate the angular convergence power from the weighted source distribution.
C_ell_lss = cosmo_ccl.angular_cl(
    wl_tracer,
    wl_tracer,
    ell_theory,
)

valid_cl = np.isfinite(C_ell_lss) & (C_ell_lss > 0.0)
if not np.any(valid_cl):
    raise RuntimeError("CCL returned no finite positive C_kappa(ell).")

print("Common analytic C_kappa(ell) ready from pooled weighted HSC n(z).")
print("ell range:", ell_theory[0], "to", ell_theory[-1])
print("C_ell range:", np.nanmin(C_ell_lss), "to", np.nanmax(C_ell_lss))

########################################################################
#Total analytic noise power on the mirrored FFT grid
########################################################################

Nx_ext = 2 * Nx
Ny_ext = 2 * Ny

ellx_ext = 2.0 * np.pi * np.fft.fftfreq(Nx_ext, d=dx_rad)
elly_ext = 2.0 * np.pi * np.fft.fftfreq(Ny_ext, d=dy_rad)
ellx_grid_ext, elly_grid_ext = np.meshgrid(
    ellx_ext,
    elly_ext,
    indexing="xy",
)
ell_grid_ext = np.hypot(ellx_grid_ext, elly_grid_ext)

P_lss_extended = np.interp(
    ell_grid_ext.ravel(),
    ell_theory,
    C_ell_lss,
    left=0.0,
    right=C_ell_lss[-1],
).reshape(Ny_ext, Nx_ext)
P_lss_extended[0, 0] = 0.0

#Calculate the white shape-noise power for the weighted source catalogue.
#The variances are in shear units after division by (2R)^2.
#Check that the effective-density expression agrees with the direct weighted catalogue sum.
n_eff_sr = n_eff_arcmin2 * arcmin_per_rad**2
P_shape_white = effective_shape_variance_shear / n_eff_sr
total_filter_area_sr = (
    N_MC * A_filter_arcmin2 / arcmin_per_rad**2
)
P_shape_white_direct = (
    total_filter_area_sr
    * pooled_filter_sum_w2_shape_var_shear
    / pooled_filter_sum_w**2
)
if not np.isclose(P_shape_white, P_shape_white_direct, rtol=1e-12):
    raise RuntimeError("Inconsistent weighted shape-noise power.")

#Increase the noise power at high angular frequencies to account for finite galaxy sampling.
#Set this scale using the weighted effective source density.
ell_arcmin_ext = ell_grid_ext / arcmin_per_rad
sampling_exponent = (
    ell_arcmin_ext**2 / (n_eff_arcmin2 * np.log(2.0))
)
P_shape_extended = P_shape_white * np.exp(
    np.clip(sampling_exponent, 0.0, 100.0)
)

P_total_extended = P_lss_extended + P_shape_extended
P_total_extended_safe = make_safe_power(P_total_extended)

ell_flat = ell_grid_ext.ravel()
lss_flat = P_lss_extended.ravel()
shape_flat = P_shape_extended.ravel()

positive_modes = ell_flat > 0.0
ell_bins = np.geomspace(
    ell_flat[positive_modes].min(),
    ell_flat.max(),
    31,
)
ell_centres = np.sqrt(ell_bins[:-1] * ell_bins[1:])

lss_binned = np.full(len(ell_centres), np.nan)
shape_binned = np.full(len(ell_centres), np.nan)

for ibin in range(len(ell_centres)):
    in_bin = (
        (ell_flat >= ell_bins[ibin])
        & (ell_flat < ell_bins[ibin + 1])
    )
    if np.any(in_bin):
        lss_binned[ibin] = np.mean(lss_flat[in_bin])
        shape_binned[ibin] = np.mean(shape_flat[in_bin])

noise_power_figure = plt.figure(figsize=(7, 4))
plt.loglog(ell_theory[valid_cl], C_ell_lss[valid_cl], label=r"analytic $C_\kappa$")
plt.loglog(ell_centres, lss_binned, "o", ms=3, label="LSS on FFT grid")
plt.xlabel(r"$\ell$")
plt.ylabel("Angular noise power")
plt.title("Analytic total-noise model used by the matched filter")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
noise_power_figure.savefig(
    RESULTS_DIR / "analytic_noise_power.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(noise_power_figure)

print(
    f"Intrinsic distortion component RMS = "
    f"{SIGMA_E_INTRINSIC_COMPONENT:.6f}"
)
print(f"HSC shear responsivity = {SHEAR_RESPONSIVITY:.6f}")
print(
    f"Effective shear-unit component RMS = "
    f"{effective_shape_std_shear:.6f}"
)
print(f"White shape-noise level = {P_shape_white:.6e}")

########################################################################
#Build one common mirrored total-noise matched-filter bank
########################################################################

#Build the same filter grids for all realizations.
#Sample these grids at the galaxy positions assigned to each realization.
Psi_grid_all = np.empty(
    (N_angles, Ny, Nx),
    dtype=np.float32,
)

for j, theta_deg in enumerate(thetas_deg):
    tau_grid = tau_template(
        theta_deg,
        xg_arcmin,
        yg_arcmin,
        hc_arcmin,
        Lfil_arcmin,
        apply_gaussian_window=APPLY_GAUSSIAN_WINDOW,
    )

    Psi_grid, _ = matched_filter_from_tau_mirrored(
        tau_grid,
        P_total_extended_safe,
    )
    Psi_grid_all[j] = Psi_grid.astype(np.float32)

#Build the realization-specific weighted filter for the paired data window.
Psi_gal_all, weighted_filter, filter_denominator = (
    build_weighted_filter_for_window(
        Psi_grid_all,
        x_arcmin,
        y_arcmin,
        data_window,
    )
)
weights_filter = np.where(
    mask_filter_sources,
    weights,
    0.0,
).astype(np.float32)

print("Common mirrored total-noise filter grid ready.")
print("Psi_grid_all shape:", Psi_grid_all.shape)
print("Paired data-window Psi_gal_all shape:", Psi_gal_all.shape)
print(
    "Paired data-window estimator denominator min/max:",
    filter_denominator.min(),
    filter_denominator.max(),
)

########################################################################
#Fixed snapshot-only response (diagnostic signal curve)
########################################################################

red_snapshot = 1.0 - kappa_snapshot_gal.astype(np.float64)
valid_snapshot = (
    mask_filter_sources
    & np.isfinite(red_snapshot)
    & (np.abs(red_snapshot) > 1e-6)
)

g1_snapshot = np.zeros(Ngal, dtype=np.float64)
g2_snapshot = np.zeros(Ngal, dtype=np.float64)

g1_snapshot[valid_snapshot] = (
    gamma1_snapshot_gal[valid_snapshot] / red_snapshot[valid_snapshot]
)
g2_snapshot[valid_snapshot] = (
    gamma2_snapshot_gal[valid_snapshot] / red_snapshot[valid_snapshot]
)

Gamma_t_snapshot, Gamma_x_snapshot, _ = apply_filter_bank(
    g1_snapshot,
    g2_snapshot,
    weighted_filter,
    thetas_deg,
    valid_mask=valid_snapshot | ~mask_filter_sources,
)

idx_snapshot_peak = int(np.nanargmax(np.abs(Gamma_t_snapshot)))
print("Snapshot-only peak angle:", thetas_deg[idx_snapshot_peak])
print("Snapshot-only peak response:", Gamma_t_snapshot[idx_snapshot_peak])
print(
    "Snapshot-only cross response at peak:",
    Gamma_x_snapshot[idx_snapshot_peak],
)

########################################################################
#Allocate the N_MC-realization noise Monte Carlo
########################################################################

Gamma_t_noise_all = np.full(
    (N_MC, N_angles),
    np.nan,
    dtype=np.float64,
)
Gamma_x_noise_all = np.full_like(Gamma_t_noise_all, np.nan)

near_critical_lss_count = np.zeros(N_MC, dtype=int)
max_abs_lss_reduced_shear = np.full(N_MC, np.nan, dtype=float)
minimum_abs_filter_denominator_mc = np.full(
    N_MC,
    np.nan,
    dtype=float,
)

seed_sequence = np.random.SeedSequence(RNG_SEED)
child_seeds = seed_sequence.spawn(N_MC + 1)
data_seed = child_seeds[DATA_SHAPE_SPAWN_INDEX]
mc_shape_spawn_indices = np.delete(
    np.arange(N_MC + 1, dtype=int),
    DATA_SHAPE_SPAWN_INDEX,
)
mc_seeds = [child_seeds[index] for index in mc_shape_spawn_indices]

if not 0 <= DATA_REALIZATION_INDEX < N_MC:
    raise ValueError("DATA_REALIZATION_INDEX must lie in [0, N_MC).")
if len(mc_seeds) != N_MC:
    raise RuntimeError("The Monte Carlo shape-seed schedule is incomplete.")
if DATA_SHAPE_SPAWN_INDEX in mc_shape_spawn_indices:
    raise RuntimeError("The data shape seed leaked into the noise Monte Carlo.")

gamma1_lss_data = None
gamma2_lss_data = None
kappa_lss_data = None

print("Noise Monte Carlo arrays allocated:", Gamma_t_noise_all.shape)
print(
    "Fixed data shape spawn index (excluded from MC):",
    DATA_SHAPE_SPAWN_INDEX,
)

########################################################################
#Parallel LSS and shape-noise realizations
########################################################################
#Each worker reads one kappaTNG file and processes its assigned realization.
#Store returned results by realization index so completion order does not change the ensemble.

MC_CHECKPOINT_VERSION = 2
completed_noise_mc = np.zeros(N_MC, dtype=bool)


def _save_noise_mc_checkpoint():
    '''
    Save completed Monte Carlo results and restart settings through a temporary file.
    Replace the checkpoint only after the temporary file has been written.
    '''
    temporary_path = MC_CHECKPOINT_PATH.with_name(
        MC_CHECKPOINT_PATH.stem + ".tmp.npz"
    )
    np.savez(
        temporary_path,
        checkpoint_version=MC_CHECKPOINT_VERSION,
        completed_mask=completed_noise_mc,
        Gamma_t_noise_all=Gamma_t_noise_all,
        Gamma_x_noise_all=Gamma_x_noise_all,
        near_critical_lss_count=near_critical_lss_count,
        max_abs_lss_reduced_shear=max_abs_lss_reduced_shear,
        minimum_abs_filter_denominator_mc=(
            minimum_abs_filter_denominator_mc
        ),
        data_lss_present=(gamma1_lss_data is not None),
        gamma1_lss_data=(
            np.empty(0, dtype=np.float32)
            if gamma1_lss_data is None
            else gamma1_lss_data
        ),
        gamma2_lss_data=(
            np.empty(0, dtype=np.float32)
            if gamma2_lss_data is None
            else gamma2_lss_data
        ),
        kappa_lss_data=(
            np.empty(0, dtype=np.float32)
            if kappa_lss_data is None
            else kappa_lss_data
        ),
        N_MC=N_MC,
        RNG_SEED=RNG_SEED,
        intrinsic_distortion_component_rms=(
            SIGMA_E_INTRINSIC_COMPONENT
        ),
        shear_responsivity=SHEAR_RESPONSIVITY,
        hc_mpc_h=HC_MPC_H,
        r_cut_mpc_h=R_CUT_MPC_H,
        apply_gaussian_window=APPLY_GAUSSIAN_WINDOW,
        thetas_deg=thetas_deg,
        kTNG_lp_numbers=kTNG_lp_numbers,
        kTNG_run_numbers=kTNG_run_numbers,
        kTNG_file_paths=np.asarray(
            [str(path) for path in kTNG_files]
        ),
        base_window_index=realization_base_window_index,
        symmetry_index=realization_symmetry_index,
        window_center_ra_deg=window_center_ra_deg,
        window_center_dec_deg=window_center_dec_deg,
        mc_shape_spawn_index=mc_shape_spawn_indices,
    )
    os.replace(temporary_path, MC_CHECKPOINT_PATH)


if MC_RESUME and MC_CHECKPOINT_PATH.exists():
    with np.load(MC_CHECKPOINT_PATH, allow_pickle=False) as checkpoint:
        required_fields = {
            "checkpoint_version",
            "completed_mask",
            "Gamma_t_noise_all",
            "Gamma_x_noise_all",
            "near_critical_lss_count",
            "max_abs_lss_reduced_shear",
            "minimum_abs_filter_denominator_mc",
            "data_lss_present",
            "gamma1_lss_data",
            "gamma2_lss_data",
            "kappa_lss_data",
            "N_MC",
            "RNG_SEED",
            "intrinsic_distortion_component_rms",
            "shear_responsivity",
            "hc_mpc_h",
            "r_cut_mpc_h",
            "apply_gaussian_window",
            "thetas_deg",
            "kTNG_lp_numbers",
            "kTNG_run_numbers",
            "kTNG_file_paths",
            "base_window_index",
            "symmetry_index",
            "window_center_ra_deg",
            "window_center_dec_deg",
            "mc_shape_spawn_index",
        }
        missing_fields = sorted(required_fields - set(checkpoint.files))
        if missing_fields:
            raise RuntimeError(
                "The Monte Carlo checkpoint is incomplete: "
                + ", ".join(missing_fields)
            )

        checkpoint_mismatches = []
        scalar_checks = {
            "checkpoint_version": MC_CHECKPOINT_VERSION,
            "N_MC": N_MC,
            "RNG_SEED": RNG_SEED,
            "intrinsic_distortion_component_rms": (
                SIGMA_E_INTRINSIC_COMPONENT
            ),
            "shear_responsivity": SHEAR_RESPONSIVITY,
            "hc_mpc_h": HC_MPC_H,
            "r_cut_mpc_h": R_CUT_MPC_H,
            "apply_gaussian_window": APPLY_GAUSSIAN_WINDOW,
        }
        for name, expected_value in scalar_checks.items():
            if not np.array_equal(
                np.asarray(checkpoint[name]),
                np.asarray(expected_value),
            ):
                checkpoint_mismatches.append(name)

        array_checks = {
            "thetas_deg": thetas_deg,
            "kTNG_lp_numbers": kTNG_lp_numbers,
            "kTNG_run_numbers": kTNG_run_numbers,
            "kTNG_file_paths": np.asarray(
                [str(path) for path in kTNG_files]
            ),
            "base_window_index": realization_base_window_index,
            "symmetry_index": realization_symmetry_index,
            "window_center_ra_deg": window_center_ra_deg,
            "window_center_dec_deg": window_center_dec_deg,
            "mc_shape_spawn_index": mc_shape_spawn_indices,
        }
        for name, expected_array in array_checks.items():
            if not np.array_equal(checkpoint[name], expected_array):
                checkpoint_mismatches.append(name)

        expected_result_shape = (N_MC, N_angles)
        if checkpoint["Gamma_t_noise_all"].shape != expected_result_shape:
            checkpoint_mismatches.append("Gamma_t_noise_all shape")
        if checkpoint["Gamma_x_noise_all"].shape != expected_result_shape:
            checkpoint_mismatches.append("Gamma_x_noise_all shape")
        if checkpoint["completed_mask"].shape != (N_MC,):
            checkpoint_mismatches.append("completed_mask shape")

        if checkpoint_mismatches:
            raise RuntimeError(
                "The existing Monte Carlo checkpoint does not match this "
                "notebook configuration. Set MC_RESUME=False for one run "
                "to replace it. Mismatches: "
                + ", ".join(checkpoint_mismatches)
            )

        completed_noise_mc[:] = checkpoint["completed_mask"]
        Gamma_t_noise_all[:] = checkpoint["Gamma_t_noise_all"]
        Gamma_x_noise_all[:] = checkpoint["Gamma_x_noise_all"]
        near_critical_lss_count[:] = (
            checkpoint["near_critical_lss_count"]
        )
        max_abs_lss_reduced_shear[:] = (
            checkpoint["max_abs_lss_reduced_shear"]
        )
        minimum_abs_filter_denominator_mc[:] = (
            checkpoint["minimum_abs_filter_denominator_mc"]
        )

        if bool(checkpoint["data_lss_present"]):
            gamma1_lss_data = checkpoint["gamma1_lss_data"].copy()
            gamma2_lss_data = checkpoint["gamma2_lss_data"].copy()
            kappa_lss_data = checkpoint["kappa_lss_data"].copy()

    if (
        completed_noise_mc[DATA_REALIZATION_INDEX]
        and gamma1_lss_data is None
    ):
        raise RuntimeError(
            "The checkpoint marks the headline universe complete but does "
            "not contain its retained LSS arrays."
        )

    print(
        "Resumed Monte Carlo checkpoint:",
        int(np.sum(completed_noise_mc)),
        "of",
        N_MC,
        "realizations already complete.",
    )


def _run_one_noise_realization(n_mc):
    '''
    Sample one kappaTNG realization using its assigned HSC window and add random shape noise.
    Filter both components and return the responses and diagnostics, without the snapshot signal.
    Retain the selected LSS fields when they are needed for the displayed mock observation.
    '''
    n_mc = int(n_mc)
    file_path = kTNG_files[n_mc]
    base_window_index_mc = int(
        realization_base_window_index[n_mc]
    )
    symmetry_index_mc = int(realization_symmetry_index[n_mc])

    window_mc = materialize_hsc_window(
        hsc_catalogue_columns,
        hsc_unit_vectors,
        hsc_window_specs[base_window_index_mc],
        background_mean_weight,
        z_lens,
        theta_arcmin_halo,
        A_arcmin2,
        A_filter_arcmin2,
        symmetry_index=symmetry_index_mc,
    )
    Ngal_mc = window_mc["Ngal"]
    mask_source_mc = window_mc["mask_source"]
    mask_filter_sources_mc = window_mc["mask_filter_sources"]

    plane_assignment_mc = assign_sources_to_kTNG_planes(
        window_mc["z_gal"],
        mask_source_mc,
        z_kTNG,
    )

    psi_gal_mc, weighted_filter_mc, denominator_mc = (
        build_weighted_filter_for_window(
            Psi_grid_all,
            x_arcmin,
            y_arcmin,
            window_mc,
        )
    )
    minimum_denominator_mc = float(np.min(np.abs(denominator_mc)))
    del psi_gal_mc

    gamma1_lss = np.zeros(Ngal_mc, dtype=np.float32)
    gamma2_lss = np.zeros(Ngal_mc, dtype=np.float32)
    kappa_lss = np.zeros(Ngal_mc, dtype=np.float32)

    with h5py.File(file_path, "r") as map_file:
        for plane_idx, source_indices in (
            plane_assignment_mc["source_indices"].items()
        ):
            group = kTNG_groups[plane_idx]
            kappa_patch = map_file[
                f"{group}/kappa"
            ][patch_y_slice, patch_x_slice]
            gamma1_patch = map_file[
                f"{group}/gamma1"
            ][patch_y_slice, patch_x_slice]
            gamma2_patch = map_file[
                f"{group}/gamma2"
            ][patch_y_slice, patch_x_slice]

            x_source = window_mc["xg"][source_indices]
            y_source = window_mc["yg"][source_indices]
            kappa_lss[source_indices] = sample_centred_patch(
                kappa_patch,
                kTNG_pixel_scale_arcmin,
                x_source,
                y_source,
            )
            gamma1_lss[source_indices] = sample_centred_patch(
                gamma1_patch,
                kTNG_pixel_scale_arcmin,
                x_source,
                y_source,
            )
            gamma2_lss[source_indices] = sample_centred_patch(
                gamma2_patch,
                kTNG_pixel_scale_arcmin,
                x_source,
                y_source,
            )

    if not (
        np.all(np.isfinite(kappa_lss[mask_source_mc]))
        and np.all(np.isfinite(gamma1_lss[mask_source_mc]))
        and np.all(np.isfinite(gamma2_lss[mask_source_mc]))
    ):
        raise RuntimeError(
            f"Non-finite sampled LSS values in {file_path} "
            f"for HSC base window {base_window_index_mc} "
            f"with symmetry {realization_symmetry_name[n_mc]}."
        )

    red_lss = 1.0 - kappa_lss.astype(np.float64)
    valid_lss = (
        mask_filter_sources_mc
        & np.isfinite(red_lss)
        & (np.abs(red_lss) > 1e-6)
    )
    near_critical_count_mc = int(np.sum(
        mask_filter_sources_mc & ~valid_lss
    ))

    g1_lss = np.zeros(Ngal_mc, dtype=np.float64)
    g2_lss = np.zeros(Ngal_mc, dtype=np.float64)
    g1_lss[valid_lss] = gamma1_lss[valid_lss] / red_lss[valid_lss]
    g2_lss[valid_lss] = gamma2_lss[valid_lss] / red_lss[valid_lss]

    max_abs_lss_mc = np.nan
    if np.any(valid_lss):
        max_abs_lss_mc = float(np.max(
            np.hypot(g1_lss[valid_lss], g2_lss[valid_lss])
        ))

    rng_mc = np.random.default_rng(mc_seeds[n_mc])
    shape_noise_1_mc, shape_noise_2_mc = draw_hsc_shear_noise(
        rng_mc,
        window_mc["sigma_e_hsc"],
        SIGMA_E_INTRINSIC_COMPONENT,
        SHEAR_RESPONSIVITY,
    )

    gamma_t_mc, gamma_x_mc, _ = apply_filter_bank(
        g1_lss + shape_noise_1_mc,
        g2_lss + shape_noise_2_mc,
        weighted_filter_mc,
        thetas_deg,
        valid_mask=valid_lss | ~mask_filter_sources_mc,
    )

    retained_data_lss = None
    if n_mc == DATA_REALIZATION_INDEX:
        pairing_is_correct = (
            np.array_equal(
                window_mc["row_indices"],
                data_window["row_indices"],
            )
            and window_mc["symmetry_index"] == data_symmetry_index
            and np.allclose(window_mc["xg"], data_window["xg"])
            and np.allclose(window_mc["yg"], data_window["yg"])
        )
        if not pairing_is_correct:
            raise RuntimeError(
                "The retained data LSS was not paired with the data HSC "
                "sampling pattern."
            )
        retained_data_lss = (
            gamma1_lss,
            gamma2_lss,
            kappa_lss,
        )

    return (
        n_mc,
        gamma_t_mc,
        gamma_x_mc,
        near_critical_count_mc,
        max_abs_lss_mc,
        minimum_denominator_mc,
        retained_data_lss,
    )


pending_indices = np.flatnonzero(~completed_noise_mc).astype(int).tolist()
print(
    f"Monte Carlo workers: {MC_N_WORKERS} "
    f"(CPUs visible/allocated: {ALLOCATED_CPU_COUNT})."
)
print(
    f"Pending realizations: {len(pending_indices)}/{N_MC}. "
    f"Checkpoint: {MC_CHECKPOINT_PATH}"
)

if MC_N_WORKERS > 1 and "fork" not in mp.get_all_start_methods():
    raise RuntimeError(
        "Parallel notebook execution requires the Linux 'fork' start "
        "method used on VERA."
    )

run_start_time = time.perf_counter()
processed_this_run = 0
last_checkpoint_total = int(np.sum(completed_noise_mc))
worker_pool = None

try:
    if MC_N_WORKERS == 1:
        result_iterator = map(_run_one_noise_realization, pending_indices)
    else:
        worker_pool = mp.get_context("fork").Pool(
            processes=MC_N_WORKERS
        )
        result_iterator = worker_pool.imap_unordered(
            _run_one_noise_realization,
            pending_indices,
            chunksize=1,
        )

    for result in result_iterator:
        (
            n_mc,
            gamma_t_mc,
            gamma_x_mc,
            near_critical_count_mc,
            max_abs_lss_mc,
            minimum_denominator_mc,
            retained_data_lss,
        ) = result

        if completed_noise_mc[n_mc]:
            raise RuntimeError(
                f"Realization {n_mc} was returned more than once."
            )

        Gamma_t_noise_all[n_mc] = gamma_t_mc
        Gamma_x_noise_all[n_mc] = gamma_x_mc
        near_critical_lss_count[n_mc] = near_critical_count_mc
        max_abs_lss_reduced_shear[n_mc] = max_abs_lss_mc
        minimum_abs_filter_denominator_mc[n_mc] = (
            minimum_denominator_mc
        )

        if retained_data_lss is not None:
            (
                gamma1_lss_data,
                gamma2_lss_data,
                kappa_lss_data,
            ) = retained_data_lss

        completed_noise_mc[n_mc] = True
        processed_this_run += 1
        completed_total = int(np.sum(completed_noise_mc))

        if (
            SAVE_CHECKPOINT_EVERY
            and completed_total - last_checkpoint_total
            >= SAVE_CHECKPOINT_EVERY
        ):
            _save_noise_mc_checkpoint()
            last_checkpoint_total = completed_total

        if (
            processed_this_run % PRINT_EVERY == 0
            or completed_total == N_MC
        ):
            elapsed_seconds = time.perf_counter() - run_start_time
            rate_per_minute = (
                60.0 * processed_this_run / max(elapsed_seconds, 1e-12)
            )
            remaining = N_MC - completed_total
            eta_minutes = remaining / max(rate_per_minute, 1e-12)
            print(
                f"Combined noise: processed {completed_total}/{N_MC}; "
                f"{rate_per_minute:.2f} files/min; "
                f"ETA {eta_minutes / 60.0:.2f} h"
            )

    if worker_pool is not None:
        worker_pool.close()
        worker_pool.join()
        worker_pool = None

except BaseException:
    if worker_pool is not None:
        worker_pool.terminate()
        worker_pool.join()
    if np.any(completed_noise_mc):
        _save_noise_mc_checkpoint()
        print(
            "Interrupted run checkpointed at",
            int(np.sum(completed_noise_mc)),
            "of",
            N_MC,
        )
    raise

_save_noise_mc_checkpoint()

if not np.all(completed_noise_mc):
    raise RuntimeError("The Monte Carlo did not return every realization.")
if gamma1_lss_data is None:
    raise RuntimeError("The requested display LSS realization was not retained.")

print(
    f"All {N_MC} independent LSS/shape-noise universes "
    "have been filtered through the scheduled HSC patterns."
)

########################################################################
#Numerical sigma and covariance from combined noise
########################################################################

Sigma_t_MC = np.nanstd(Gamma_t_noise_all, axis=0, ddof=1)
Sigma_x_MC = np.nanstd(Gamma_x_noise_all, axis=0, ddof=1)

Gamma_t_noise_mean = np.nanmean(Gamma_t_noise_all, axis=0)
Gamma_x_noise_mean = np.nanmean(Gamma_x_noise_all, axis=0)

Cov_t_MC = np.cov(Gamma_t_noise_all, rowvar=False, ddof=1)
Cov_x_MC = np.cov(Gamma_x_noise_all, rowvar=False, ddof=1)

standard_error_t = Sigma_t_MC / np.sqrt(N_MC)
standard_error_x = Sigma_x_MC / np.sqrt(N_MC)

zscore_mean_t = Gamma_t_noise_mean / (standard_error_t + 1e-30)
zscore_mean_x = Gamma_x_noise_mean / (standard_error_x + 1e-30)

SNR_t_snapshot = Gamma_t_snapshot / (Sigma_t_MC + 1e-30)
SNR_x_snapshot = Gamma_x_snapshot / (Sigma_x_MC + 1e-30)
idx_snapshot_snr_peak = int(np.nanargmax(np.abs(SNR_t_snapshot)))

print("Combined numerical sigma ready from", N_MC, "realizations.")
print("mean Sigma_t_MC =", np.nanmean(Sigma_t_MC))
print("mean Sigma_x_MC =", np.nanmean(Sigma_x_MC))
print(
    "Snapshot-only expected peak S/N:",
    SNR_t_snapshot[idx_snapshot_snr_peak],
    "at",
    thetas_deg[idx_snapshot_snr_peak],
    "deg",
)
print("largest |noise mean / standard error|, tangential =",
      np.nanmax(np.abs(zscore_mean_t)))
print("largest |noise mean / standard error|, cross =",
      np.nanmax(np.abs(zscore_mean_x)))
print("maximum near-critical LSS source count =",
      np.max(near_critical_lss_count))
print("maximum sampled |g_LSS| =",
      np.nanmax(max_abs_lss_reduced_shear))

########################################################################
#Single observed realization: snapshot + LSS + shape noise
########################################################################

rng_data = np.random.default_rng(data_seed)
shape_noise_1_data, shape_noise_2_data = draw_hsc_shear_noise(
    rng_data,
    sigma_e_hsc,
    SIGMA_E_INTRINSIC_COMPONENT,
    SHEAR_RESPONSIVITY,
)

gamma1_total_data = (
    gamma1_snapshot_gal.astype(np.float64)
    + gamma1_lss_data.astype(np.float64)
)
gamma2_total_data = (
    gamma2_snapshot_gal.astype(np.float64)
    + gamma2_lss_data.astype(np.float64)
)
kappa_total_data = (
    kappa_snapshot_gal.astype(np.float64)
    + kappa_lss_data.astype(np.float64)
)

red_total_data = 1.0 - kappa_total_data
valid_reduced_data = (
    mask_source
    & np.isfinite(red_total_data)
    & (np.abs(red_total_data) > 1e-6)
)
valid_total_data = mask_filter_sources & valid_reduced_data

g1_total_data = np.zeros(Ngal, dtype=np.float64)
g2_total_data = np.zeros(Ngal, dtype=np.float64)

g1_total_data[valid_reduced_data] = (
    gamma1_total_data[valid_reduced_data]
    / red_total_data[valid_reduced_data]
)
g2_total_data[valid_reduced_data] = (
    gamma2_total_data[valid_reduced_data]
    / red_total_data[valid_reduced_data]
)

e1_observed_data = g1_total_data + shape_noise_1_data
e2_observed_data = g2_total_data + shape_noise_2_data

Gamma_t_data, Gamma_x_data, data_filter_denominator = apply_filter_bank(
    e1_observed_data,
    e2_observed_data,
    weighted_filter,
    thetas_deg,
    valid_mask=valid_total_data | ~mask_filter_sources,
)

SNR_t_data = Gamma_t_data / (Sigma_t_MC + 1e-30)
SNR_x_data = Gamma_x_data / (Sigma_x_MC + 1e-30)

idx_gamma_peak = int(np.nanargmax(np.abs(Gamma_t_data)))
idx_snr_peak = idx_gamma_peak

print("Displayed LSS universe:", kTNG_files[DATA_REALIZATION_INDEX])
print("Displayed shape spawn index:", DATA_SHAPE_SPAWN_INDEX)
print(
    "Noisy-data S/N at snapshot-only expected peak:",
    SNR_t_data[idx_snapshot_snr_peak],
)
print("Gamma_t peak angle:", thetas_deg[idx_gamma_peak])
print("Gamma_t peak value:", Gamma_t_data[idx_gamma_peak])
print("S/N at Gamma_t peak:", SNR_t_data[idx_gamma_peak])
print("Cross S/N at Gamma_t peak:", SNR_x_data[idx_gamma_peak])

########################################################################
#Matched-filter curves for the complete single realization
########################################################################

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    thetas_deg,
    Gamma_t_data,
    linewidth=1.8,
    label=r"single realization $\Gamma_+(\theta)$",
)

ax.plot(
    thetas_deg,
    Gamma_x_data,
    linewidth=1.2,
    label=r"single realization $\Gamma_\times(\theta)$",
)

ax.fill_between(
    thetas_deg,
    Gamma_t_data - Sigma_t_MC,
    Gamma_t_data + Sigma_t_MC,
    alpha=0.25,
    label=r"$\Gamma_+\pm\sigma_{\rm MC}$",
)


ax.axhline(0.0, color="k", linewidth=1.0)

ax.set_xlabel(r"$\theta$ [deg]")
ax.set_ylabel("Noise-consistent HSC-weighted estimator")
ax.set_title(
    "Single realization: snapshot + LSS + intrinsic/measurement noise"
)
ax.legend()

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "single_realization_matched_filter.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

########################################################################
#Noise distribution at the snapshot-expected S/N peak
########################################################################

diagnostic_angle_index = idx_snapshot_snr_peak
diagnostic_angle = thetas_deg[diagnostic_angle_index]
noise_responses_at_peak = Gamma_t_noise_all[:, diagnostic_angle_index]

noise_histogram_figure = plt.figure(figsize=(7, 4))
plt.hist(noise_responses_at_peak, bins=25, alpha=0.75)
plt.axvline(
    Gamma_t_data[diagnostic_angle_index],
    color="k",
    linestyle="--",
    linewidth=1.5,
    label="complete single realization",
)
plt.axvline(
    Gamma_t_snapshot[diagnostic_angle_index],
    color="tab:red",
    linestyle=":",
    linewidth=1.5,
    label="snapshot-only signal",
)
plt.xlabel("Matched-filter response")
plt.ylabel("Number of noise realizations")
plt.title(
    f"Noise distribution at snapshot-expected peak: "
    f"theta={diagnostic_angle:.1f} deg"
)
plt.legend()
plt.tight_layout()
noise_histogram_figure.savefig(
    RESULTS_DIR / "noise_response_distribution.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(noise_histogram_figure)

########################################################################
#Projected density, signed shear, and local filter contributions
########################################################################
#Display the signed shear component projected along the snapshot-expected filament direction.
#The magnitude |g| is non-negative, so random shape noise also appears bright in that map.
#Keeping the sign and orientation shows the shear pattern and its cancellations.
#The separate contribution map also includes each galaxy weight and filter value.

spatial_diagnostic_index = idx_snapshot_snr_peak
spatial_diagnostic_theta_deg = float(
    thetas_deg[spatial_diagnostic_index]
)
spatial_diagnostic_angle_2 = 2.0 * np.deg2rad(
    spatial_diagnostic_theta_deg
)
spatial_cos_2 = np.cos(spatial_diagnostic_angle_2)
spatial_sin_2 = np.sin(spatial_diagnostic_angle_2)

snapshot_projected_component = (
    -spatial_cos_2 * g1_snapshot
    -spatial_sin_2 * g2_snapshot
)
observed_projected_component = (
    -spatial_cos_2 * e1_observed_data
    -spatial_sin_2 * e2_observed_data
)

x_edges_arcmin = np.linspace(
    -x_half_arcmin,
    x_half_arcmin,
    Nx + 1,
)
y_edges_arcmin = np.linspace(
    -y_half_arcmin,
    y_half_arcmin,
    Ny + 1,
)

snapshot_component_display, snapshot_support_display = (
    smooth_weighted_catalogue_field(
        snapshot_projected_component[valid_snapshot],
        xg[valid_snapshot],
        yg[valid_snapshot],
        weights[valid_snapshot],
        x_edges_arcmin,
        y_edges_arcmin,
        DISPLAY_SMOOTHING_PIXELS,
    )
)
observed_component_display, observed_support_display = (
    smooth_weighted_catalogue_field(
        observed_projected_component[valid_total_data],
        xg[valid_total_data],
        yg[valid_total_data],
        weights[valid_total_data],
        x_edges_arcmin,
        y_edges_arcmin,
        DISPLAY_SMOOTHING_PIXELS,
    )
)

snapshot_support_cut = (
    0.02 * np.nanmax(snapshot_support_display)
)
observed_support_cut = (
    0.02 * np.nanmax(observed_support_display)
)
snapshot_component_display[
    snapshot_support_display < snapshot_support_cut
] = np.nan
observed_component_display[
    observed_support_display < observed_support_cut
] = np.nan

#Calculate each galaxy contribution to the normalized tangential response at the chosen angle.
#Their sum before display smoothing must reproduce the matched-filter response.
local_filter_contribution = np.zeros(Ngal, dtype=np.float64)
local_filter_contribution[valid_total_data] = (
    weighted_filter[
        spatial_diagnostic_index,
        valid_total_data,
    ]
    * observed_projected_component[valid_total_data]
    / data_filter_denominator[spatial_diagnostic_index]
)
local_filter_contribution_sum = np.sum(
    local_filter_contribution,
    dtype=np.float64,
)

if not np.isclose(
    local_filter_contribution_sum,
    Gamma_t_data[spatial_diagnostic_index],
    rtol=1e-6,
    atol=1e-10,
):
    raise RuntimeError(
        "The spatial contribution map does not reproduce Gamma_t_data."
    )

local_contribution_display, _, _ = np.histogram2d(
    yg[valid_total_data],
    xg[valid_total_data],
    bins=[y_edges_arcmin, x_edges_arcmin],
    weights=local_filter_contribution[valid_total_data],
)
local_contribution_display = gaussian_filter(
    local_contribution_display,
    sigma=DISPLAY_SMOOTHING_PIXELS,
    mode="constant",
)
local_contribution_display[
    observed_support_display < observed_support_cut
] = np.nan

#Use the projected snapshot dark matter density as a reference for the actual simulated structure.
#This map is independent of the source galaxies and is not a lensing reconstruction.
#A logarithmic display makes both the dense halo and fainter surrounding filaments visible.
#Apply the positive density floor only to the display map.
positive_surface_density = Sigma[
    np.isfinite(Sigma) & (Sigma > 0.0)
]
if positive_surface_density.size == 0:
    raise RuntimeError("The snapshot surface-density map is empty.")

surface_density_floor = np.percentile(
    positive_surface_density,
    1.0,
)
log_surface_density_display = np.log10(
    np.maximum(Sigma, surface_density_floor)
)

template_spatial_diagnostic = tau_template(
    spatial_diagnostic_theta_deg,
    xg_arcmin,
    yg_arcmin,
    hc_arcmin,
    Lfil_arcmin,
    apply_gaussian_window=APPLY_GAUSSIAN_WINDOW,
)
template_levels = (
    np.array([0.25, 0.50, 0.75])
    * np.nanmax(template_spatial_diagnostic)
)

def robust_symmetric_limit(field, percentile=99.0):
    '''Choose a positive colour-scale limit from a percentile of the absolute finite field values.'''
    finite_values = np.asarray(field)[np.isfinite(field)]
    if finite_values.size == 0:
        raise RuntimeError("A spatial diagnostic field is empty.")
    return max(
        np.percentile(np.abs(finite_values), percentile),
        np.finfo(np.float64).eps,
    )


snapshot_component_limit = robust_symmetric_limit(
    snapshot_component_display,
    percentile=99.0,
)
observed_component_limit = robust_symmetric_limit(
    observed_component_display,
    percentile=99.0,
)
local_contribution_limit = robust_symmetric_limit(
    local_contribution_display,
    percentile=99.0,
)

circle_angle = np.linspace(0.0, 2.0 * np.pi, 500)
axis_length_kpc = (
    0.92
    * min(
        0.5 * Lx_kpc,
        0.5 * Ly_kpc,
    )
)
expected_theta_rad = np.deg2rad(spatial_diagnostic_theta_deg)
noisy_theta_rad = np.deg2rad(thetas_deg[idx_snr_peak])

expected_axis_endpoint = axis_length_kpc * np.array([
    np.cos(expected_theta_rad),
    np.sin(expected_theta_rad),
])
noisy_axis_endpoint = axis_length_kpc * np.array([
    np.cos(noisy_theta_rad),
    np.sin(noisy_theta_rad),
])


def add_spatial_guides(
    ax,
    contour_color,
    show_noisy_direction=False,
):
    '''
    Overlay the template contours, halo centre, full R200 circle, and snapshot-expected direction.
    Optionally show the direction of the noisy signal-to-noise peak.
    '''
    ax.contour(
        x_kpc,
        y_kpc,
        template_spatial_diagnostic,
        levels=template_levels,
        colors=contour_color,
        linewidths=0.65,
        alpha=0.55,
    )
    ax.plot(
        [0.0, expected_axis_endpoint[0]],
        [0.0, expected_axis_endpoint[1]],
        color="gold",
        linewidth=1.8,
        label=(
            f"snapshot-expected direction "
            f"({spatial_diagnostic_theta_deg:.0f} deg)"
        ),
    )
    if show_noisy_direction:
        ax.plot(
            [0.0, noisy_axis_endpoint[0]],
            [0.0, noisy_axis_endpoint[1]],
            color="cyan",
            linestyle="--",
            linewidth=1.4,
            label=f"noisy S/N peak ({thetas_deg[idx_snr_peak]:.0f} deg)",
        )
    ax.plot(
        halo_R200_kpc * np.cos(circle_angle),
        halo_R200_kpc * np.sin(circle_angle),
        color="gold",
        linewidth=1.2,
    )
    ax.scatter(
        0.0,
        0.0,
        c="magenta",
        s=28,
        marker="*",
        zorder=6,
    )
    ax.set_xlim(extent_kpc[0], extent_kpc[1])
    ax.set_ylim(extent_kpc[2], extent_kpc[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")


fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 11),
    constrained_layout=True,
)

surface_vmin = np.nanpercentile(
    log_surface_density_display,
    2.0,
)
surface_vmax = np.nanpercentile(
    log_surface_density_display,
    99.8,
)
im_surface = axes[0, 0].imshow(
    log_surface_density_display,
    origin="lower",
    extent=extent_kpc,
    cmap="magma",
    vmin=surface_vmin,
    vmax=surface_vmax,
    aspect="equal",
    interpolation="bilinear",
)
add_spatial_guides(
    axes[0, 0],
    contour_color="white",
    show_noisy_direction=True,
)
axes[0, 0].set_title(
    "1. Projected snapshot matter: filament truth"
)
axes[0, 0].legend(
    loc="lower left",
    fontsize=8,
    framealpha=0.80,
)
fig.colorbar(
    im_surface,
    ax=axes[0, 0],
    fraction=0.046,
    pad=0.04,
    label=r"$\log_{10}\Sigma\ [{\rm M_\odot\,kpc^{-2}}]$",
)

im_snapshot_component = axes[0, 1].imshow(
    snapshot_component_display,
    origin="lower",
    extent=extent_kpc,
    cmap="RdBu_r",
    vmin=-snapshot_component_limit,
    vmax=snapshot_component_limit,
    aspect="equal",
    interpolation="bilinear",
)
add_spatial_guides(
    axes[0, 1],
    contour_color="black",
)
axes[0, 1].set_title(
    r"2. Snapshot-only projected shear $g_+(\theta_\star)$"
)
fig.colorbar(
    im_snapshot_component,
    ax=axes[0, 1],
    fraction=0.046,
    pad=0.04,
    label=r"smoothed snapshot $g_+$",
)

im_observed_component = axes[1, 0].imshow(
    observed_component_display,
    origin="lower",
    extent=extent_kpc,
    cmap="RdBu_r",
    vmin=-observed_component_limit,
    vmax=observed_component_limit,
    aspect="equal",
    interpolation="bilinear",
)
add_spatial_guides(
    axes[1, 0],
    contour_color="black",
)
axes[1, 0].set_title(
    r"3. Noisy HSC shear projected onto $\theta_\star$"
)
fig.colorbar(
    im_observed_component,
    ax=axes[1, 0],
    fraction=0.046,
    pad=0.04,
    label=r"smoothed observed $g_+$ proxy",
)

im_contribution = axes[1, 1].imshow(
    local_contribution_display,
    origin="lower",
    extent=extent_kpc,
    cmap="RdBu_r",
    vmin=-local_contribution_limit,
    vmax=local_contribution_limit,
    aspect="equal",
    interpolation="bilinear",
)
add_spatial_guides(
    axes[1, 1],
    contour_color="black",
)
axes[1, 1].set_title(
    r"4. Local contribution to matched-filter $\Gamma_+$"
)
fig.colorbar(
    im_contribution,
    ax=axes[1, 1],
    fraction=0.046,
    pad=0.04,
    label=r"smoothed local contribution to $\Gamma_+$",
)

fig.suptitle(
    (
        f"LP100_run001 matched-filter anatomy: "
        f"theta_star={spatial_diagnostic_theta_deg:.1f} deg, "
        f"snapshot S/N={SNR_t_snapshot[spatial_diagnostic_index]:.2f}, "
        f"noisy S/N={SNR_t_data[spatial_diagnostic_index]:.2f}\n"
        f"HSC base window {data_base_window_index}, "
        f"{data_symmetry_name}; display smoothing only"
    ),
    fontsize=13,
)

fig.savefig(
    RESULTS_DIR / "matched_filter_spatial_diagnostic.png",
    dpi=350,
    bbox_inches="tight",
    facecolor="white",
)
#Save the same signed spatial diagnostic under both output filenames.
fig.savefig(
    RESULTS_DIR / "complete_observed_field_contour.png",
    dpi=350,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)

print(
    "Spatial diagnostic angle (snapshot-expected S/N peak):",
    spatial_diagnostic_theta_deg,
    "deg",
)
print(
    "Local contribution sum / Gamma_t_data:",
    local_filter_contribution_sum,
    Gamma_t_data[spatial_diagnostic_index],
)

########################################################################
#Filament inference at the strongest positive response
########################################################################
#Follow the fitting method in Section 6.5 and Eq. (4) of Shinde & Dell'Antonio.
#Fit the filament profile to shear projected along the strongest positive response direction.
#Use sources on the forward branch within the configured transverse distance.
#Use a constant Gaussian noise estimate and uniform priors with the configured MCMC settings.
#Keep the same central exclusion as the matched filter, currently half the halo R200 radius.
#The fitted data retain both LSS and shape noise.

detected_peak_index = idx_gamma_peak
detected_theta_deg = float(thetas_deg[detected_peak_index])
detected_peak_snr = float(SNR_t_data[detected_peak_index])

if not np.isfinite(detected_peak_snr) or detected_peak_snr <= 0.0:
    raise RuntimeError("No positive tangential S/N peak is available for MCMC.")

detected_theta_rad = np.deg2rad(detected_theta_deg)
detected_angle_2 = 2.0 * detected_theta_rad
g_t_observed_detected = (
    -np.cos(detected_angle_2) * e1_observed_data
    -np.sin(detected_angle_2) * e2_observed_data
)

x_parallel_arcmin, h_signed_arcmin = rotate_coords(
    xg,
    yg,
    detected_theta_rad,
)
h_detected_mpc = (
    np.abs(h_signed_arcmin)
    * np.deg2rad(1.0 / 60.0)
    * DA
)

mcmc_data_mask = (
    valid_total_data
    & (x_parallel_arcmin > 0.0)
    & (h_detected_mpc < MCMC_HREF_MPC)
)

if np.sum(mcmc_data_mask) < 2 * MCMC_N_WALKERS:
    raise RuntimeError(
        "Too few valid sources in the paper's 1.5 Mpc MCMC window: "
        f"{np.sum(mcmc_data_mask)}."
    )

mcmc_h_mpc = h_detected_mpc[mcmc_data_mask]
mcmc_g_t = g_t_observed_detected[mcmc_data_mask]

#Estimate one Gaussian noise scale from the two observed shear-component dispersions.
#The likelihood uses this common scale without per-galaxy analysis weights.
mcmc_gaussian_noise_sigma = 0.5 * (
    np.std(e1_observed_data[valid_total_data], ddof=1)
    + np.std(e2_observed_data[valid_total_data], ddof=1)
)

if (
    not np.isfinite(mcmc_gaussian_noise_sigma)
    or mcmc_gaussian_noise_sigma <= 0.0
):
    raise RuntimeError("The MCMC Gaussian noise estimate is invalid.")


def paper_filament_profile(h_mpc, kappa_0, h_c_mpc):
    '''
    Return the fitted filament profile: a flat core followed by a declining outer profile.
    The parameters are the central amplitude kappa_0 and core width h_c_mpc.
    '''
    h_mpc = np.asarray(h_mpc, dtype=np.float64)
    profile = np.full(h_mpc.shape, kappa_0, dtype=np.float64)
    outside_core = h_mpc > h_c_mpc
    profile[outside_core] = kappa_0 / (
        1.0
        + ((h_mpc[outside_core] - h_c_mpc) / h_c_mpc) ** 2
    )
    return profile


def filament_log_prior(parameters):
    '''Return a uniform log prior within the amplitude and width bounds, or minus infinity outside.'''
    kappa_0, h_c_mpc = parameters
    if (
        MCMC_KAPPA0_BOUNDS[0] < kappa_0 < MCMC_KAPPA0_BOUNDS[1]
        and MCMC_HC_BOUNDS_MPC[0] < h_c_mpc < MCMC_HC_BOUNDS_MPC[1]
    ):
        return 0.0
    return -np.inf


def filament_log_likelihood(parameters):
    '''Compare the filament profile with the projected shear using a constant Gaussian noise variance.'''
    kappa_0, h_c_mpc = parameters
    model = paper_filament_profile(
        mcmc_h_mpc,
        kappa_0,
        h_c_mpc,
    )
    residual = mcmc_g_t - model
    variance = mcmc_gaussian_noise_sigma**2
    return -0.5 * np.sum(
        residual**2 / variance + np.log(2.0 * np.pi * variance)
    )


def filament_log_posterior(parameters):
    '''Add the log prior and log likelihood, rejecting parameters outside the prior bounds.'''
    log_prior = filament_log_prior(parameters)
    if not np.isfinite(log_prior):
        return -np.inf
    return log_prior + filament_log_likelihood(parameters)


mcmc_rng = np.random.default_rng(MCMC_RANDOM_SEED)
initial_position = np.column_stack([
    mcmc_rng.uniform(
        MCMC_KAPPA0_BOUNDS[0],
        MCMC_KAPPA0_BOUNDS[1],
        MCMC_N_WALKERS,
    ),
    mcmc_rng.uniform(
        MCMC_HC_BOUNDS_MPC[0],
        MCMC_HC_BOUNDS_MPC[1],
        MCMC_N_WALKERS,
    ),
])

filament_sampler = emcee.EnsembleSampler(
    MCMC_N_WALKERS,
    2,
    filament_log_posterior,
)
filament_sampler.random_state = np.random.RandomState(
    MCMC_RANDOM_SEED + 1
).get_state()

print(
    f"Running emcee at theta={detected_theta_deg:.1f} deg "
    f"(S/N={detected_peak_snr:.3f})..."
)
filament_sampler.run_mcmc(
    initial_position,
    MCMC_N_STEPS,
)

filament_chain = filament_sampler.get_chain()
filament_log_probability = filament_sampler.get_log_prob()
filament_samples = filament_sampler.get_chain(
    discard=MCMC_BURN_IN,
    flat=True,
)

if (
    filament_samples.shape
    != (MCMC_N_WALKERS * (MCMC_N_STEPS - MCMC_BURN_IN), 2)
    or not np.all(np.isfinite(filament_samples))
):
    raise RuntimeError("The flattened post-burn-in MCMC chain is invalid.")

filament_percentiles = np.percentile(
    filament_samples,
    [16.0, 50.0, 84.0],
    axis=0,
)
filament_median = filament_percentiles[1]
filament_error_low = filament_median - filament_percentiles[0]
filament_error_high = filament_percentiles[2] - filament_median

mcmc_acceptance_fraction = float(
    np.mean(filament_sampler.acceptance_fraction)
)
if not 0.15 <= mcmc_acceptance_fraction <= 0.70:
    warnings.warn(
        "Mean emcee acceptance fraction is outside the broad 0.15-0.70 "
        f"diagnostic range: {mcmc_acceptance_fraction:.3f}."
    )

try:
    mcmc_autocorrelation_time = filament_sampler.get_autocorr_time(
        discard=MCMC_BURN_IN,
        tol=0,
    )
except emcee.autocorr.AutocorrError as error:
    mcmc_autocorrelation_time = np.full(2, np.nan)
    warnings.warn(f"Could not estimate MCMC autocorrelation time: {error}")

mcmc_post_burn_steps = MCMC_N_STEPS - MCMC_BURN_IN
mcmc_effective_samples = np.full(2, np.nan)
finite_tau = np.isfinite(mcmc_autocorrelation_time)
mcmc_effective_samples[finite_tau] = (
    MCMC_N_WALKERS
    * mcmc_post_burn_steps
    / mcmc_autocorrelation_time[finite_tau]
)

if np.any(
    finite_tau
    & (mcmc_post_burn_steps < 50.0 * mcmc_autocorrelation_time)
):
    warnings.warn(
        "The paper's fixed 2000-step chain is shorter than 50 integrated "
        "autocorrelation times for at least one parameter. Interpret the "
        "reported credible intervals cautiously or extend MCMC_N_STEPS."
    )

print(
    "kappa_0 = "
    f"{filament_median[0]:.4f} "
    f"+{filament_error_high[0]:.4f} "
    f"-{filament_error_low[0]:.4f}"
)
print(
    "h_c = "
    f"{filament_median[1]:.3f} "
    f"+{filament_error_high[1]:.3f} "
    f"-{filament_error_low[1]:.3f} Mpc"
)
print(f"Mean acceptance fraction = {mcmc_acceptance_fraction:.3f}")
print("Autocorrelation time [steps] =", mcmc_autocorrelation_time)
print("Approximate effective samples =", mcmc_effective_samples)

#Show the joint posterior and marginal distributions of filament amplitude and width.
posterior_color = "#8A2BE2"
posterior_labels = [r"$\kappa_0$", r"$h_c\ [{\rm Mpc}]$"]
corner_figure = corner.corner(
    filament_samples,
    labels=posterior_labels,
    bins=100,
    smooth=1.0,
    smooth1d=1.0,
    levels=[0.68, 0.95],
    plot_datapoints=False,
    plot_density=False,
    plot_contours=True,
    fill_contours=False,
    color=posterior_color,
    contour_kwargs={"linewidths": [1.1, 1.5]},
    hist_kwargs={"linewidth": 1.4},
    label_kwargs={"fontsize": 13},
)

corner_axes = np.asarray(corner_figure.axes).reshape((2, 2))
for parameter_index in range(2):
    axis = corner_axes[parameter_index, parameter_index]
    axis.axvline(
        filament_median[parameter_index],
        color=posterior_color,
        linestyle="--",
        linewidth=1.25,
        alpha=0.45,
    )

kappa_text = (
    rf"$\kappa_0={filament_median[0]:.3f}"
    rf"^{{+{filament_error_high[0]:.3f}}}"
    rf"_{{-{filament_error_low[0]:.3f}}}$"
)
hc_text = (
    rf"$h_c={filament_median[1]:.2f}"
    rf"^{{+{filament_error_high[1]:.2f}}}"
    rf"_{{-{filament_error_low[1]:.2f}}}\ {{\rm Mpc}}$"
)
corner_axes[0, 0].text(
    0.98,
    0.96,
    kappa_text,
    color=posterior_color,
    ha="right",
    va="top",
    transform=corner_axes[0, 0].transAxes,
    fontsize=10,
)
corner_axes[1, 1].text(
    0.98,
    0.96,
    hc_text,
    color=posterior_color,
    ha="right",
    va="top",
    transform=corner_axes[1, 1].transAxes,
    fontsize=10,
)

from matplotlib.lines import Line2D

corner_axes[0, 1].legend(
    handles=[
        Line2D(
            [0],
            [0],
            color=posterior_color,
            linewidth=2.0,
            label=rf"$\theta={detected_theta_deg:.0f}^\circ$",
        )
    ],
    loc="upper right",
    frameon=True,
    edgecolor="black",
    framealpha=1.0,
)
corner_figure.savefig(
    RESULTS_DIR / "filament_mcmc_corner.png",
    dpi=350,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(corner_figure)

#Plot the walker traces to inspect sampling behaviour and possible convergence problems.
trace_figure, trace_axes = plt.subplots(
    2,
    1,
    figsize=(9, 5.5),
    sharex=True,
    constrained_layout=True,
)
for parameter_index, (axis, label) in enumerate(
    zip(trace_axes, posterior_labels)
):
    axis.plot(
        filament_chain[:, :, parameter_index],
        color=posterior_color,
        alpha=0.12,
        linewidth=0.5,
    )
    axis.axvline(
        MCMC_BURN_IN,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="burn-in",
    )
    axis.set_ylabel(label)
trace_axes[0].legend(loc="upper right")
trace_axes[-1].set_xlabel("MCMC step")
trace_figure.suptitle(
    f"Filament posterior traces at theta={detected_theta_deg:.1f} deg"
)
trace_figure.savefig(
    RESULTS_DIR / "filament_mcmc_trace.png",
    dpi=250,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(trace_figure)

np.savez_compressed(
    RESULTS_DIR / "filament_mcmc_samples.npz",
    chain=filament_chain,
    log_probability=filament_log_probability,
    samples=filament_samples,
    percentiles=filament_percentiles,
    detected_peak_index=detected_peak_index,
    detected_theta_deg=detected_theta_deg,
    detected_peak_snr=detected_peak_snr,
    data_mask=mcmc_data_mask,
    h_mpc=mcmc_h_mpc,
    g_t=mcmc_g_t,
    gaussian_noise_sigma=mcmc_gaussian_noise_sigma,
    acceptance_fraction=mcmc_acceptance_fraction,
    autocorrelation_time=mcmc_autocorrelation_time,
    effective_samples=mcmc_effective_samples,
    n_walkers=MCMC_N_WALKERS,
    n_steps=MCMC_N_STEPS,
    burn_in=MCMC_BURN_IN,
)

mcmc_summary = pd.DataFrame({
    "quantity": [
        "detected_theta_deg",
        "detected_peak_SNR",
        "N_sources_in_fit",
        "Gaussian_noise_sigma",
        "kappa_0_median",
        "kappa_0_error_low",
        "kappa_0_error_high",
        "h_c_Mpc_median",
        "h_c_Mpc_error_low",
        "h_c_Mpc_error_high",
        "mean_acceptance_fraction",
        "tau_kappa_0_steps",
        "tau_h_c_steps",
        "effective_samples_kappa_0",
        "effective_samples_h_c",
    ],
    "value": [
        detected_theta_deg,
        detected_peak_snr,
        int(np.sum(mcmc_data_mask)),
        mcmc_gaussian_noise_sigma,
        filament_median[0],
        filament_error_low[0],
        filament_error_high[0],
        filament_median[1],
        filament_error_low[1],
        filament_error_high[1],
        mcmc_acceptance_fraction,
        mcmc_autocorrelation_time[0],
        mcmc_autocorrelation_time[1],
        mcmc_effective_samples[0],
        mcmc_effective_samples[1],
    ],
})
mcmc_summary.to_csv(
    RESULTS_DIR / "filament_mcmc_summary.csv",
    index=False,
)

########################################################################
#Recovered filament orientation on the simulated matter distribution
########################################################################
#Overlay the recovered filament direction on the projected snapshot dark matter density.
#Select subhaloes from the same periodic cube, with side length set by twice R_CUT_MPC_H.
#These simulated objects provide a positional comparison with the surrounding halo structure.

dx_sub = periodic_delta(sub_pos[:, 0], halo_pos[0], boxsize_ckpc_h)
dy_sub = periodic_delta(sub_pos[:, 1], halo_pos[1], boxsize_ckpc_h)
dz_sub = periodic_delta(sub_pos[:, 2], halo_pos[2], boxsize_ckpc_h)
subhalo_cube_mask = (
    (np.abs(dx_sub) < Rcut_ckpc_h)
    & (np.abs(dy_sub) < Rcut_ckpc_h)
    & (np.abs(dz_sub) < Rcut_ckpc_h)
)

subhalo_x_mpc = (
    dx_sub[subhalo_cube_mask] / h / (1.0 + z_lens) / 1000.0
)
subhalo_y_mpc = (
    dy_sub[subhalo_cube_mask] / h / (1.0 + z_lens) / 1000.0
)

extent_mpc = np.asarray(extent_kpc, dtype=float) / 1000.0
axis_length_mpc = axis_length_kpc / 1000.0
detected_direction = np.array([
    np.cos(detected_theta_rad),
    np.sin(detected_theta_rad),
])
arrow_start_mpc = 1.03 * halo_R200_Mpc * detected_direction
arrow_end_mpc = axis_length_mpc * detected_direction

orientation_figure, orientation_axis = plt.subplots(
    figsize=(7.5, 7.0),
    constrained_layout=True,
)
orientation_image = orientation_axis.imshow(
    log_surface_density_display,
    origin="lower",
    extent=extent_mpc,
    cmap="gray_r",
    vmin=surface_vmin,
    vmax=surface_vmax,
    aspect="equal",
    interpolation="bilinear",
)

orientation_axis.scatter(
    subhalo_x_mpc,
    subhalo_y_mpc,
    s=10,
    facecolors="none",
    edgecolors="#32A852",
    linewidths=0.55,
    alpha=0.45,
    label="simulated subhaloes",
    zorder=3,
)
orientation_axis.plot(
    halo_R200_Mpc * np.cos(circle_angle)*0.5,
    halo_R200_Mpc * np.sin(circle_angle)*0.5,
    color="red",
    linestyle="--",
    linewidth=1.35,
    label=r"central $R_{200}$ exclusion",
    zorder=4,
)
orientation_axis.scatter(
    0.0,
    0.0,
    marker="*",
    s=115,
    c="gold",
    edgecolors="black",
    linewidths=0.7,
    label="central halo",
    zorder=6,
)
orientation_axis.annotate(
    "",
    xy=arrow_end_mpc,
    xytext=arrow_start_mpc,
    arrowprops={
        "arrowstyle": "-|>",
        "color": "#8A2BE2",
        "linewidth": 2.8,
        "mutation_scale": 18,
    },
    zorder=7,
)
orientation_axis.plot(
    [],
    [],
    color="#8A2BE2",
    linewidth=2.8,
    label=(
        rf"recovered $\theta={detected_theta_deg:.0f}^\circ$ "
        rf"($S/N={detected_peak_snr:.2f}$)"
    ),
)

orientation_axis.set_xlim(extent_mpc[0], extent_mpc[1])
orientation_axis.set_ylim(extent_mpc[2], extent_mpc[3])
orientation_axis.set_aspect("equal")
orientation_axis.set_xlabel(r"$x\ [{\rm Mpc}]$")
orientation_axis.set_ylabel(r"$y\ [{\rm Mpc}]$")
orientation_axis.set_title(
    "Strongest positive matched-filter orientation"
)
orientation_axis.legend(
    loc="lower left",
    fontsize=8.5,
    framealpha=0.88,
)
orientation_figure.colorbar(
    orientation_image,
    ax=orientation_axis,
    fraction=0.046,
    pad=0.04,
    label=r"$\log_{10}\Sigma\ [{\rm M_\odot\,kpc^{-2}}]$",
)
orientation_figure.savefig(
    RESULTS_DIR / "filament_orientation_overlay.png",
    dpi=350,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(orientation_figure)

print("Subhaloes shown in orientation overlay:", np.sum(subhalo_cube_mask))
print("Recovered orientation shown:", detected_theta_deg, "deg")

########################################################################
#Save the numerical result
########################################################################

np.savez_compressed(
    RESULTS_DIR / f"matched_filter_hsc_{N_MC}_analytic_results.npz",
    thetas_deg=thetas_deg,
    Gamma_t_noise_all=Gamma_t_noise_all,
    Gamma_x_noise_all=Gamma_x_noise_all,
    Gamma_t_noise_mean=Gamma_t_noise_mean,
    Gamma_x_noise_mean=Gamma_x_noise_mean,
    Gamma_t_snapshot=Gamma_t_snapshot,
    Gamma_x_snapshot=Gamma_x_snapshot,
    Gamma_t_data=Gamma_t_data,
    Gamma_x_data=Gamma_x_data,
    Sigma_t_MC=Sigma_t_MC,
    Sigma_x_MC=Sigma_x_MC,
    Cov_t_MC=Cov_t_MC,
    Cov_x_MC=Cov_x_MC,
    SNR_t_data=SNR_t_data,
    SNR_x_data=SNR_x_data,
    SNR_t_snapshot=SNR_t_snapshot,
    SNR_x_snapshot=SNR_x_snapshot,
    ell_theory=ell_theory,
    C_ell_lss=C_ell_lss,
    P_shape_white=P_shape_white,
    P_shape_white_direct=P_shape_white_direct,
    effective_shape_variance_shear=effective_shape_variance_shear,
    effective_shape_std_shear=effective_shape_std_shear,
    n_gal_raw_arcmin2=n_gal_raw_arcmin2,
    n_back_raw_arcmin2=n_back_raw_arcmin2,
    n_filter_raw_arcmin2=n_filter_raw_arcmin2,
    n_eff_data_arcmin2=n_eff_data_arcmin2,
    n_eff_pooled_arcmin2=n_eff_arcmin2,
    Ngal=Ngal,
    N_back=N_back,
    N_filter_sources=N_filter_sources,
    intrinsic_distortion_component_rms=(
        SIGMA_E_INTRINSIC_COMPONENT
    ),
    shear_responsivity=SHEAR_RESPONSIVITY,
    hc_mpc_h=HC_MPC_H,
    N_MC=N_MC,
    MC_N_WORKERS=MC_N_WORKERS,
    completed_noise_mc=completed_noise_mc,
    N_HSC_BASE_WINDOWS=N_HSC_BASE_WINDOWS,
    RNG_SEED=RNG_SEED,
    WINDOW_SELECTION_SEED=WINDOW_SELECTION_SEED,
    WINDOW_SCHEDULE_SEED=WINDOW_SCHEDULE_SEED,
    data_lp_number=DATA_LP_NUMBER,
    data_run_number=DATA_RUN_NUMBER,
    kTNG_lp_numbers=kTNG_lp_numbers,
    kTNG_run_numbers=kTNG_run_numbers,
    data_realization_index=DATA_REALIZATION_INDEX,
    data_shape_spawn_index=DATA_SHAPE_SPAWN_INDEX,
    mc_shape_spawn_indices=mc_shape_spawn_indices,
    data_base_window_index=data_base_window_index,
    data_symmetry_index=data_symmetry_index,
    snapshot_snr_peak_index=idx_snapshot_snr_peak,
    noisy_snr_peak_index=idx_snr_peak,
    spatial_diagnostic_index=spatial_diagnostic_index,
    spatial_diagnostic_theta_deg=spatial_diagnostic_theta_deg,
    snapshot_projected_component=snapshot_projected_component,
    observed_projected_component=observed_projected_component,
    local_filter_contribution=local_filter_contribution,
    local_filter_contribution_sum=local_filter_contribution_sum,
    local_contribution_display=local_contribution_display,
    log_surface_density_display=log_surface_density_display,
    data_sigma_e_measurement=sigma_e_hsc,
    data_analysis_weight_raw=weight_raw,
    data_catalogue_weight_raw=catalogue_weight_raw,
    data_shape_variance_distortion=shape_variance_distortion,
    data_shape_variance_shear=shape_variance_shear,
    realization_base_window_index=realization_base_window_index,
    realization_symmetry_index=realization_symmetry_index,
    realization_symmetry_name=realization_symmetry_name,
    base_window_use_counts=window_use_counts,
    base_window_center_ra_deg=base_window_center_ra_deg,
    base_window_center_dec_deg=base_window_center_dec_deg,
    base_window_Ngal=base_window_Ngal,
    base_window_N_back=base_window_N_back,
    base_window_N_filter_sources=base_window_N_filter_sources,
    base_window_n_eff_arcmin2=base_window_n_eff_arcmin2,
    base_window_mean_shape_std_shear=(
        base_window_mean_shape_std_shear
    ),
    window_center_ra_deg=window_center_ra_deg,
    window_center_dec_deg=window_center_dec_deg,
    window_Ngal=window_Ngal,
    window_N_back=window_N_back,
    window_N_filter_sources=window_N_filter_sources,
    window_n_eff_arcmin2=window_n_eff_arcmin2,
    window_mean_shape_std_shear=window_mean_shape_std_shear,
)

angle_summary = pd.DataFrame({
    "theta_deg": thetas_deg,
    "Gamma_t_data": Gamma_t_data,
    "Gamma_x_data": Gamma_x_data,
    "Gamma_t_snapshot": Gamma_t_snapshot,
    "Gamma_x_snapshot": Gamma_x_snapshot,
    "Gamma_t_noise_mean": Gamma_t_noise_mean,
    "Gamma_x_noise_mean": Gamma_x_noise_mean,
    "sigma_t_MC": Sigma_t_MC,
    "sigma_x_MC": Sigma_x_MC,
    "SNR_t_data": SNR_t_data,
    "SNR_x_data": SNR_x_data,
    "SNR_t_snapshot": SNR_t_snapshot,
    "SNR_x_snapshot": SNR_x_snapshot,
})
angle_summary.to_csv(
    RESULTS_DIR / "matched_filter_angle_summary.csv",
    index=False,
)

window_summary = pd.DataFrame({
    "realization_index": np.arange(N_MC),
    "base_window_index": realization_base_window_index,
    "square_symmetry_index": realization_symmetry_index,
    "square_symmetry_name": realization_symmetry_name,
    "centre_ra_deg": window_center_ra_deg,
    "centre_dec_deg": window_center_dec_deg,
    "N_selected_all": window_Ngal,
    "N_foreground": window_N_front,
    "N_background": window_N_back,
    "N_filter_sources": window_N_filter_sources,
    "raw_density_all_arcmin-2": window_n_gal_raw_arcmin2,
    "raw_density_background_arcmin-2": window_n_back_raw_arcmin2,
    "raw_density_filter_arcmin-2": window_n_filter_raw_arcmin2,
    "weighted_n_eff_filter_arcmin-2": window_n_eff_arcmin2,
    "mean_shape_component_rms_shear": (
        window_mean_shape_std_shear
    ),
    "paired_LP_number": kTNG_lp_numbers,
    "paired_run_number": kTNG_run_numbers,
    "paired_universe_id": [
        f"LP{lp:03d}_run{run:03d}"
        for lp, run in zip(kTNG_lp_numbers, kTNG_run_numbers)
    ],
    "MC_shape_spawn_index": mc_shape_spawn_indices,
    "provides_headline_LSS": (
        np.arange(N_MC) == DATA_REALIZATION_INDEX
    ),
})
window_summary.to_csv(
    RESULTS_DIR / "hsc_realization_schedule.csv",
    index=False,
)

base_window_summary = pd.DataFrame({
    "base_window_index": np.arange(N_HSC_BASE_WINDOWS),
    "centre_ra_deg": base_window_center_ra_deg,
    "centre_dec_deg": base_window_center_dec_deg,
    "realization_use_count": window_use_counts,
    "N_selected_all": base_window_Ngal,
    "N_foreground": base_window_N_front,
    "N_background": base_window_N_back,
    "N_filter_sources": base_window_N_filter_sources,
    "raw_density_all_arcmin-2": base_window_n_gal_raw_arcmin2,
    "raw_density_background_arcmin-2": base_window_n_back_raw_arcmin2,
    "raw_density_filter_arcmin-2": base_window_n_filter_raw_arcmin2,
    "weighted_n_eff_filter_arcmin-2": base_window_n_eff_arcmin2,
    "mean_shape_component_rms_shear": (
        base_window_mean_shape_std_shear
    ),
    "is_data_base_window": (
        np.arange(N_HSC_BASE_WINDOWS) == data_base_window_index
    ),
})
base_window_summary.to_csv(
    RESULTS_DIR / "independent_hsc_base_window_summary.csv",
    index=False,
)

catalogue_summary = pd.DataFrame({
    "quantity": [
        "geometric_area_per_window_arcmin2",
        "filter_area_per_window_arcmin2",
        "N_independent_HSC_base_windows",
        "N_noise_realizations",
        "minimum_base_window_reuse_count",
        "maximum_base_window_reuse_count",
        "data_lp_number",
        "data_run_number",
        "data_realization_index",
        "data_shape_spawn_index",
        "data_base_window_index",
        "data_square_symmetry",
        "data_window_N_selected_all",
        "data_window_N_background",
        "data_window_N_filter_sources",
        "data_window_raw_density_all_arcmin-2",
        "data_window_raw_density_background_arcmin-2",
        "data_window_raw_density_filter_arcmin-2",
        "data_window_weighted_n_eff_filter_arcmin-2",
        "pooled_weighted_n_eff_filter_arcmin-2",
        "intrinsic_distortion_component_rms",
        "shear_responsivity",
        "effective_shape_component_rms_shear",
        "data_median_measurement_sigma_e",
        "data_median_shape_component_rms_shear",
        "white_shape_noise_power",
        "filament_core_scale_mpc_h",
        "N_kappaTNG_noise_realizations",
        "snapshot_expected_peak_angle_deg",
        "snapshot_expected_peak_SNR",
        "noisy_SNR_at_snapshot_expected_peak",
        "noisy_peak_angle_deg",
        "noisy_peak_SNR",
        "MCMC_detected_theta_deg",
        "MCMC_detected_peak_SNR",
        "MCMC_kappa_0_median",
        "MCMC_h_c_Mpc_median",
    ],
    "value": [
        A_arcmin2,
        A_filter_arcmin2,
        N_HSC_BASE_WINDOWS,
        N_MC,
        np.min(window_use_counts),
        np.max(window_use_counts),
        DATA_LP_NUMBER,
        DATA_RUN_NUMBER,
        DATA_REALIZATION_INDEX,
        DATA_SHAPE_SPAWN_INDEX,
        data_base_window_index,
        data_symmetry_name,
        Ngal,
        N_back,
        N_filter_sources,
        n_gal_raw_arcmin2,
        n_back_raw_arcmin2,
        n_filter_raw_arcmin2,
        n_eff_data_arcmin2,
        n_eff_arcmin2,
        SIGMA_E_INTRINSIC_COMPONENT,
        SHEAR_RESPONSIVITY,
        effective_shape_std_shear,
        np.median(sigma_e_hsc[mask_filter_sources]),
        np.median(shape_std_shear[mask_filter_sources]),
        P_shape_white,
        HC_MPC_H,
        N_MC,
        thetas_deg[idx_snapshot_snr_peak],
        SNR_t_snapshot[idx_snapshot_snr_peak],
        SNR_t_data[idx_snapshot_snr_peak],
        thetas_deg[idx_snr_peak],
        SNR_t_data[idx_snr_peak],
        detected_theta_deg,
        detected_peak_snr,
        filament_median[0],
        filament_median[1],
    ],
})
catalogue_summary.to_csv(
    RESULTS_DIR / "catalogue_and_noise_summary.csv",
    index=False,
)

print("Saved numerical products in:", RESULTS_DIR.resolve())

########################################################################
#Automated consistency checks
########################################################################

analysis_variance_all = hsc_catalogue_columns[
    "analysis_shape_variance_distortion"
]
analysis_variance_shear_all = hsc_catalogue_columns[
    "analysis_shape_variance_shear"
]
analysis_weight_all = hsc_catalogue_columns["analysis_weight"]
expected_variance_all = (
    SIGMA_E_INTRINSIC_COMPONENT**2
    + hsc_catalogue_columns["sigma_e"]**2
)

checks = {
    "clean_object_id_header": "object_id" in hsc_raw.columns,
    "fixed_intrinsic_component_RMS_is_0p4": np.isclose(
        SIGMA_E_INTRINSIC_COMPONENT,
        0.4,
    ),
    "responsivity_matches_fixed_dispersion": np.isclose(
        SHEAR_RESPONSIVITY,
        1.0 - SIGMA_E_INTRINSIC_COMPONENT**2,
    ),
    "analysis_variance_matches_shape_draw": np.allclose(
        analysis_variance_all,
        expected_variance_all,
    ),
    "analysis_weights_inverse_of_shape_variance": np.allclose(
        analysis_weight_all * analysis_variance_all,
        1.0,
    ),
    "shape_variance_converted_to_shear_units": np.allclose(
        analysis_variance_shear_all,
        analysis_variance_all / (2.0 * SHEAR_RESPONSIVITY) ** 2,
    ),
    "weighted_shape_power_discrete_equivalence": np.isclose(
        P_shape_white,
        P_shape_white_direct,
        rtol=1e-12,
    ),
    "filament_core_scale_is_requested_0p8": np.isclose(
        HC_MPC_H,
        0.8,
    ),
    "selected_HSC_base_windows": N_HSC_BASE_WINDOWS >= 1,
    "strictly_non_overlapping_base_windows": (
        window_selection_info["strictly_non_overlapping"]
    ),
    "base_window_catalogue_rows_disjoint": (
        window_selection_info["rows_are_disjoint"]
    ),
    "realization_schedule_has_N_MC_rows": (
        len(realization_base_window_index) == N_MC
        and len(realization_symmetry_index) == N_MC
    ),
    "realization_base_indices_valid": np.all(
        (realization_base_window_index >= 0)
        & (realization_base_window_index < N_HSC_BASE_WINDOWS)
    ),
    "balanced_base_window_reuse": (
        np.max(window_use_counts) - np.min(window_use_counts) <= 1
    ),
    "square_symmetry_indices_valid": np.all(
        (realization_symmetry_index >= 0)
        & (
            realization_symmetry_index
            < len(HSC_SQUARE_SYMMETRY_NAMES)
        )
    ),
    "displayed_window_has_identity_symmetry": data_symmetry_index == 0,
    "displayed_window_is_fixed_base_zero": (
        data_base_window_index == DATA_BASE_WINDOW_INDEX == 0
    ),
    "all_base_windows_have_filter_sources": np.all(
        base_window_N_filter_sources > 0
    ),
    "selected_N_MC_nested_files": len(kTNG_files) == N_MC,
    "parallel_MC_completed_every_index": np.all(
        completed_noise_mc
    ),
    "worker_count_within_CPU_allocation": (
        1 <= MC_N_WORKERS <= ALLOCATED_CPU_COUNT
    ),
    "one_unique_LP_run_per_realization": (
        len({tuple(item) for item in kTNG_universe_ids}) == N_MC
    ),
    "complete_selected_universe_sequence": np.array_equal(
        kTNG_universe_ids,
        expected_kTNG_universe_ids,
    ),
    "headline_LSS_is_fixed_LP100_run001": (
        DATA_LP_NUMBER == 100
        and DATA_RUN_NUMBER == 1
        and DATA_REALIZATION_INDEX == 99
        and tuple(kTNG_universe_ids[DATA_REALIZATION_INDEX]) == (100, 1)
    ),
    "data_shape_seed_excluded_from_MC": (
        DATA_SHAPE_SPAWN_INDEX not in mc_shape_spawn_indices
        and len(np.unique(mc_shape_spawn_indices)) == N_MC
    ),
    "headline_shape_seed_preserves_old_100_run": (
        data_seed.spawn_key == (100,)
    ),
    "positive_pooled_effective_density": n_eff_arcmin2 > 0.0,
    "background_sources_available": N_back > 0,
    "filter_sources_available": N_filter_sources > 0,
    "nonzero_data_filter_denominator": np.all(
        np.abs(filter_denominator) > 1e-30
    ),
    "nonzero_all_MC_filter_denominators": np.all(
        minimum_abs_filter_denominator_mc > 1e-30
    ),
    "all_MC_tangential_finite": np.all(
        np.isfinite(Gamma_t_noise_all)
    ),
    "all_MC_cross_finite": np.all(
        np.isfinite(Gamma_x_noise_all)
    ),
    "all_sigma_positive": np.all(Sigma_t_MC > 0.0),
    "covariance_shape_correct": Cov_t_MC.shape == (
        N_angles,
        N_angles,
    ),
    "single_realization_finite": (
        np.all(np.isfinite(Gamma_t_data))
        and np.all(np.isfinite(Gamma_x_data))
    ),
    "data_window_pairing_preserved": (
        np.array_equal(
            data_window["row_indices"],
            hsc_window_specs[data_base_window_index]["row_indices"],
        )
        and data_window["symmetry_index"] == data_symmetry_index
    ),
    "spatial_diagnostic_uses_expected_peak": (
        spatial_diagnostic_index == idx_snapshot_snr_peak
    ),
    "spatial_contributions_reproduce_estimator": np.isclose(
        local_filter_contribution_sum,
        Gamma_t_data[spatial_diagnostic_index],
        rtol=1e-6,
        atol=1e-10,
    ),
    "spatial_signed_fields_available": (
        np.any(np.isfinite(snapshot_component_display))
        and np.any(np.isfinite(observed_component_display))
        and np.any(np.isfinite(local_contribution_display))
    ),
    "MCMC_uses_strongest_positive_peak": (
        detected_peak_index == int(np.nanargmax(Gamma_t_data))
        and Gamma_t_data[detected_peak_index] > 0.0
    ),
    "MCMC_fit_window_is_populated": (
        np.sum(mcmc_data_mask) >= 2 * MCMC_N_WALKERS
    ),
    "MCMC_samples_are_finite": np.all(np.isfinite(filament_samples)),
    "MCMC_acceptance_fraction_is_finite": np.isfinite(
        mcmc_acceptance_fraction
    ),
    "orientation_overlay_has_subhaloes": np.any(subhalo_cube_mask),
    "snapshot_plot_covers_full_box": np.allclose(
        extent_kpc,
        [-0.5 * Lx_kpc, 0.5 * Lx_kpc,
         -0.5 * Ly_kpc, 0.5 * Ly_kpc],
    ),
}

for name, passed in checks.items():
    print(f"{name:42s}: {passed}")

failed_checks = [name for name, passed in checks.items() if not passed]
if failed_checks:
    raise RuntimeError(
        "Pipeline checks failed: " + ", ".join(failed_checks)
    )

print("All automated consistency checks passed.")

expected_figure_paths = [
    RESULTS_DIR / "hsc_catalogue_diagnostics.png",
    RESULTS_DIR / "snapshot_shear_whiskers.png",
    RESULTS_DIR / "analytic_noise_power.png",
    RESULTS_DIR / "single_realization_matched_filter.png",
    RESULTS_DIR / "noise_response_distribution.png",
    RESULTS_DIR / "matched_filter_spatial_diagnostic.png",
    RESULTS_DIR / "complete_observed_field_contour.png",
    RESULTS_DIR / "filament_mcmc_corner.png",
    RESULTS_DIR / "filament_mcmc_trace.png",
    RESULTS_DIR / "filament_orientation_overlay.png",
]
missing_or_empty_figures = [
    path
    for path in expected_figure_paths
    if not path.is_file() or path.stat().st_size == 0
]
if missing_or_empty_figures:
    raise RuntimeError(
        "Missing or empty saved figures: "
        + ", ".join(str(path) for path in missing_or_empty_figures)
    )

print(f"Verified {len(expected_figure_paths)} saved PNG files:")
for figure_path in expected_figure_paths:
    print("  ", figure_path)
print("All figures and numerical products are in:", RESULTS_DIR.resolve())
