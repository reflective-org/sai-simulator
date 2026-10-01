"""Shared constants and lookup tables for the SAI simulator backend.

Defines the regional model name list, grid dimensions, SAI start-year floor, dashboard
temperature-target range, the per-variable ``VAR2INFO`` display table (title, units, colormap,
SO2-overlay color), and the ``exposurevar2var`` mapping from exposure-variable names to their
underlying daily climate variables.
"""
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


REGIONAL_MODEL_NAMES = [
    "CESM2-WACCM",
    "MIROC-ES2H",
    "UKESM1-1-LL",
]

# A model can publish two distinct no-SAI runs, which serve two different roles:
#
#   * **SAI-delta rebase reference** — the run the model's SAI experiments branched from.
#     The delta fit must rebase each anchor against it (see ``ANCHORS`` in
#     scripts/fit_delta.py), so that choice is pinned there, not here.
#   * **projection baseline** — the standard ScenarioMIP SSP2-4.5 run that the
#     climate-change-only emulator trains on (scripts/fit_map.py) and that backs the
#     dashboard's historical series. That is what this table selects.
#
# CESM2-WACCM's single baseline run serves both roles. MIROC-ES2H ships them separately:
# its GeoMIP ``baseline`` (2020-2084) is the G6 branch point, while ``ssp245`` (2015-2100)
# is the ScenarioMIP run — longer, and continuous with the historical run.
#
# Values are preference-ordered: the first file present in a variable's processed
# directory wins. MIROC publishes no ScenarioMIP evaporation or sea ice, so those
# variables have no ``ssp245`` output and fall back to the branch-point baseline.
REGIONAL_MODEL2PROJECTION_BASELINE_FILES = {
    "CESM2-WACCM": ("output_gauss-baseline.nc",),
    "MIROC-ES2H": ("output_gauss-ssp245-miroc.nc", "output_gauss-baseline-miroc.nc"),
    # UKESM's single het-chem SSP2-4.5 run serves both roles (CESM-style).
    "UKESM1-1-LL": ("output_gauss-baseline-ukesm.nc",),
}

# Variables a model cannot contribute to the PROJECTION side (the fit_map
# emulators, the multi-model projection mean, and the multi-model historical
# reference). UKESM1-1-LL's historical publishes no evaporative flux, so its
# p-e series starts in 2015: no 1850-1900 rebase window and nothing to
# contribute to the historical mean — and _load_multi_model_baseline averages
# models over their OVERLAPPING years, so including it would erase everyone's
# pre-2015 p-e. UKESM's p-e still feeds the SAI delta fit (fit_delta.ANCHORS),
# which rebases against the fully-covered branch-point baseline instead.
# (icefrac needs no entry: its projection side is CESM-only by construction —
# see backend/historical.py.)
REGIONAL_MODEL_VAR_EXCLUSIONS: dict[str, tuple[str, ...]] = {
    # Empty since UKESM's historical QFLX was sourced from CMIP6 ESGF hfls: it
    # previously sat out p-e's projection side because its evaporative flux began
    # in 2015, and _load_multi_model_baseline inner-joins years, so a 2015-start
    # series left the <= 2014 historical reference EMPTY rather than merely short.
}


def regional_models_for_var(var: str) -> list[str]:
    """The regional models that participate in ``var``'s projection side.

    Exposure variables are looked up under two names — the count variable
    (``tas_above_35``, the processed/emulator directory name) and its exposure
    alias (``exposure_above_35``, the dashboard name) — so both resolve to the
    same list; exclusions are declared on the count-variable name.
    """
    base = exposurevar2var.get(var, var)
    excluded = REGIONAL_MODEL_VAR_EXCLUSIONS.get(base, ())
    return [m for m in REGIONAL_MODEL_NAMES if m not in excluded]


# The first regional model is the primary: its grid is the target grid, its
# processed baseline must exist, and fit_map / the cache pipeline treat
# position 0 specially. It can never be excluded from a variable.
for _var, _models in REGIONAL_MODEL_VAR_EXCLUSIONS.items():
    if REGIONAL_MODEL_NAMES[0] in _models:
        raise ValueError(f"the primary model {REGIONAL_MODEL_NAMES[0]} cannot be excluded ({_var})")
    if not set(_models) <= set(REGIONAL_MODEL_NAMES):
        raise ValueError(f"unknown model in REGIONAL_MODEL_VAR_EXCLUSIONS[{_var!r}]: {_models}")

# Provenance string written into exported NetCDFs (see backend/get_outputs.py).
DEFAULT_MODEL_NAME = (
    " + ".join(REGIONAL_MODEL_NAMES)
    + " (equal-weight mean; per-variable exclusions in REGIONAL_MODEL_VAR_EXCLUSIONS)"
)


def resolve_projection_baseline_file(variable_dir: str | Path, model: str) -> Path | None:
    """Return ``model``'s projection-baseline file inside ``variable_dir``, or ``None``.

    Walks :data:`REGIONAL_MODEL2PROJECTION_BASELINE_FILES` in preference order and returns
    the first file that exists, so a variable missing from the preferred (ScenarioMIP) store
    falls back to the model's other no-SAI run. ``None`` means the model has no processed
    baseline for this variable at all.
    """
    for filename in REGIONAL_MODEL2PROJECTION_BASELINE_FILES[model]:
        candidate = Path(variable_dir) / filename
        if candidate.exists():
            return candidate
    return None
NUM_EMULATORS = 100

# Per-cell significance: values in ``regional_*_p_values`` at or above this are "not
# significant" and get hatched. What the values are (raw or FDR-adjusted p-values) is
# defined in backend/p_values.py; this is the only place the threshold lives.
SIGNIFICANCE_THRESHOLD = 0.05

# The dashboard's default SAI deployment. scripts/cache.py precomputes the with-SAI
# p-values for exactly this deployment; any other start year / ramp-up is computed live,
# because the with-SAI field (and so its significance) depends on both.
DEFAULT_RAMP_UP = 10
# CESM2-WACCM grid dimensions (all other models are regridded onto this grid)
NUM_LAT = 192
NUM_LON = 288
MIN_SAI_START = 2035
DEFAULT_SAI_START_YEAR = MIN_SAI_START

# Reference periods for temperature/precipitation anomalies.
# PREINDUSTRIAL_* is the frame the *projections* and the SAI ``temp_target`` live in (the
# FaIR global temperature, the FaIR->SMIP emulator inputs, and the icefrac model all assume
# anomalies vs 1850-1900). Do not change it without retraining/recaching the models.
# DISPLAY_BASELINE_* is the "current climate" period the dashboard *presents* anomalies and
# significance against; it shifts what the user sees, not the underlying physics. The window
# ends at 2014 because the historical model data stops there (see get_historical_model).
PREINDUSTRIAL_START = 1850
PREINDUSTRIAL_END = 1900
DISPLAY_BASELINE_START = 1995
DISPLAY_BASELINE_END = 2014

MIN_TARGET = 0.5
MAX_TARGET = 2.5
STEP_TARGET = 0.1

VAR2INFO = {
    "tas": ("Temperature in °C", "Temperature (°C)", "coolwarm", "#00FF00"),
    "p-e": (
        "Water Availability in mm/day",
        "Water Availability (mm/day)",
        LinearSegmentedColormap.from_list(
            "brown_to_white_to_blue",
            ["brown", "khaki", "white", "deepskyblue", "mediumblue"],
            N=100
        ),
        "red"
    ),
    "tasmin": ("Minimum Temperature in °C", "Min Temperature (°C)", "coolwarm", "#00FF00"),
    "tasmax": ("Maximum Temperature in °C", "Max Temperature (°C)", "coolwarm","#00FF00"),
    "exposure_above_40": (
        "Person-Days Above 40°C (Millions)",
        "Exposure to > 40°C (million person-days)",
        LinearSegmentedColormap.from_list(
            "white_to_red", 
            ["white", "red"], 
            N=100
        ),
        "#00FF00"
    ),
    "exposure_above_35": (
        "Person-Days Above 35°C (Millions)",
        "Exposure to > 35°C (million person-days)",
        LinearSegmentedColormap.from_list(
            "white_to_red",
            ["white", "red"],
            N=100
        ),
        "#00FF00"
    ),
    "exposure_below_0": (
        "Person-Days Below 0°C (Millions)",
        "Exposure to < 0°C (million person-days)",
        LinearSegmentedColormap.from_list(
            "white_to_blue",
            ["white", "deepskyblue", "mediumblue"],
            N=100
        ),
        "#00FF00"
    ),
    "exposure_above_10": (
        "Person-Days Above 10mm/day (Millions)", 
        "Exposure to > 10mm (million mm-days)", 
        LinearSegmentedColormap.from_list(
            "white_to_blue", 
            ["white", "deepskyblue", "mediumblue"], 
            N=100
        ),
        "red"
    ),
    "exposure_above_20": (
        "Person-Days Above 20mm/day (Millions)",
        "Exposure to > 20mm (million mm-days)",
        LinearSegmentedColormap.from_list(
            "white_to_blue", 
            ["white", "deepskyblue", "mediumblue"], 
            N=100
        ), 
        "red"
    ),
    "icefrac": (
        "Ice Fraction", # title
        "Ice Fraction", # description
        plt.get_cmap('Blues_r', 10),
        "#00FF00"       # color
    ),
}

exposurevar2var = {
    "exposure_above_40": "tas_above_40",
    "exposure_above_35": "tas_above_35",
    "exposure_below_0": "tas_below_0",
    "exposure_above_10": "pr_above_10",
    "exposure_above_20": "pr_above_20"
}
