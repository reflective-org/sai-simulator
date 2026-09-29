"""FaIR setup, calibration, and runner helpers for the SAI simulator.

Builds a :class:`fair_wrap.custom_fair.FAIR` instance pre-loaded with RCMIP emissions,
the chris-roadmap calibrated parameter ensemble (100 members, sampled with a fixed seed),
solar / volcanic forcing time series, and optionally a CESM-tuned override of the climate
configs. The runner is pickled per scenario under ``fair_wrap/f_no_sai/<ssp>.pkl`` so
repeated dashboard requests for the same SSP reuse the same fully-built FaIR object instead
of re-downloading and re-allocating.

Module-level constants: ``SIM_START_YEAR`` / ``SIM_END_YEAR`` define the simulation horizon;
``NUM_MEMBERS`` controls ensemble size; ``FANCY_SSP_TITLES`` and ``REVERSE_FANCY_SSP_TITLES``
translate between the short FaIR scenario keys and the dashboard's display strings.
"""
import pooch
import numpy as np
import pandas as pd
import pickle as pkl
import os
from pathlib import Path
from fair_wrap.custom_fair import FAIR
from fair_wrap.custom_fair.io import read_properties
from fair_wrap.custom_fair.interface import fill, initialise


# SIM_START_YEAR = 1950
# Have to use 1750 for now because we don't have CESM2 calibrated initial conditions for 1950
SIM_START_YEAR = 1750
IGNORE_DIR = SIM_START_YEAR == 1750
SIM_END_YEAR = 2100
NUM_MEMBERS = 100
SCENARIOS = ["ssp119", "ssp126", "ssp534-over", "ssp434", "ssp245", "ssp460", "ssp370", "ssp585"]
CANVAS_START_YEAR = 2024
CANVAS_END_YEAR = SIM_END_YEAR
FANCY_SSP_TITLES = {
    "ssp119": r"SSP1-1.9",
    "ssp126": r"SSP1-2.6",
    "ssp434": r"SSP4-3.4",
    "ssp534-over": r"SSP5-3.4",
    "ssp245": r"SSP2-4.5",
    "ssp460": r"SSP4-6.0",
    "ssp370": r"SSP3-Baseline",
    "ssp585": r"SSP5-Baseline",
}
REVERSE_FANCY_SSP_TITLES = {
    title: scenario for scenario, title in FANCY_SSP_TITLES.items()
}

HERE = os.path.dirname(os.path.realpath(__file__))
FAIR_DIR = Path(HERE)
F_INIT_DIR = FAIR_DIR / "f_no_sai"
F_INIT_DIR.mkdir(exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def get_df_emis() -> pd.DataFrame:
    """Download (and cache) the RCMIP v5.1.0 annual-mean emissions CSV and return it as a DataFrame.

    :mod:`pooch` verifies the MD5 (``md5:4044106f55ca65b094670e7577eaf9b3``) so a corrupted
    download is refused. A spoofed ``User-Agent`` is passed because Zenodo rejects the
    default :mod:`urllib` UA.
    """
    rcmip_emissions_file = pooch.retrieve(
        url="https://zenodo.org/records/4589756/files/rcmip-emissions-annual-means-v5-1-0.csv",
        # original url: "doi:10.5281/zenodo.4589756/rcmip-emissions-annual-means-v5-1-0.csv",
        known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
        downloader=pooch.HTTPDownloader(headers=headers),
    )
    df_emis = pd.read_csv(rcmip_emissions_file)
    return df_emis


def get_df_configs(ensemble: bool = True) -> pd.DataFrame:
    """Download (and cache) the FaIR 1.2.0 calibrated-constrained parameter ensemble.

    When ``ensemble=True`` (the default) returns ``NUM_MEMBERS`` rows sampled with a fixed
    ``random_state=42`` so runs are reproducible. When ``False``, returns a single-row median
    of the full ensemble for deterministic / debugging runs.
    """
    fair_params_1_2_0_obj = pooch.retrieve(
        url = 'https://zenodo.org/record/8399112/files/calibrated_constrained_parameters.csv',
        known_hash = 'md5:de3b83432b9d071efdd1427ad31e9076',
        downloader=pooch.HTTPDownloader(headers=headers),
    )
    df_configs = pd.read_csv(fair_params_1_2_0_obj, index_col=0)
    if not ensemble:
        df_configs = pd.DataFrame(df_configs.median(axis=0)).T
    else:
        df_configs = df_configs.sample(NUM_MEMBERS, random_state=42)
    return df_configs


def get_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One-stop loader for the four input DataFrames FaIR needs: emissions, configs, solar, and volcanic forcing."""
    df_emis = get_df_emis()
    df_configs = get_df_configs()
    df_solar, df_volcanic = get_aerosols()
    return df_emis, df_configs, df_solar, df_volcanic


def set_up_fair(
    df_emis: pd.DataFrame,
    df_configs: pd.DataFrame,
    sim_start_year: int,
    sim_end_year: int,
    scenario: str | None = None,
) -> FAIR:
    """Build and populate a :class:`FAIR` instance for the requested scenario.

    Walks the canonical FaIR init sequence (``define_time`` / ``define_scenarios`` /
    ``define_configs`` / ``define_species`` / ``allocate`` / ``fill_from_rcmip``) using
    ``species_configs_properties_calibration1.2.0.csv``, and then patches the NOx emissions
    so the four GFED fire sectors are summed into a single AFOLU/agriculture+FFI series with
    the standard 46.006/30.006 mass conversion. Returns the populated FaIR instance ready
    for the caller to apply aerosol and config overrides.
    """
    ## 1. Create FaIR instance
    f = FAIR(ch4_method="Thornhill2021")

    ## 2. Define time horizon
    f.define_time(sim_start_year, sim_end_year, 1)  # start, end, step

    ## 3. Define scenarios
    if scenario is None:
        scenarios = SCENARIOS
    else:
        scenarios = [scenario]
    f.define_scenarios(scenarios)

    ## 4. Define configs
    configs = df_configs.index  # this is used as a label for the "config" axis
    f.define_configs(configs)

    ## 5. Define species and properties
    species, properties = read_properties(filename=FAIR_DIR / 'species_configs_properties_calibration1.2.0.csv')
    f.define_species(species, properties)

    ## 6. Modify run options

    ## 7. Create input and output xarrays
    f.allocate()

    ## 8. Fill in data
    ### 8a. emissions, solar forcing, and volcanic forcing
    f.fill_from_rcmip()

    gfed_sectors = [
        "Emissions|NOx|MAGICC AFOLU|Agricultural Waste Burning",
        "Emissions|NOx|MAGICC AFOLU|Forest Burning",
        "Emissions|NOx|MAGICC AFOLU|Grassland Burning",
        "Emissions|NOx|MAGICC AFOLU|Peat Burning",
    ]
    for scenario in scenarios:
        f.emissions.loc[dict(specie="NOx", scenario=scenario)] = (
            df_emis.loc[
                (df_emis["Scenario"] == scenario)
                & (df_emis["Region"] == "World")
                & (df_emis["Variable"].isin(gfed_sectors)),
                str(sim_start_year):str(sim_end_year),
            ]
            .interpolate(axis=1)
            .values.squeeze()
            .sum(axis=0)
            * 46.006
            / 30.006
            + df_emis.loc[
                (df_emis["Scenario"] == scenario)
                & (df_emis["Region"] == "World")
                & (df_emis["Variable"] == "Emissions|NOx|MAGICC AFOLU|Agriculture"),
                str(sim_start_year):str(sim_end_year),
            ]
            .interpolate(axis=1)
            .values.squeeze()
            + df_emis.loc[
                (df_emis["Scenario"] == scenario)
                & (df_emis["Region"] == "World")
                & (df_emis["Variable"] == "Emissions|NOx|MAGICC Fossil and Industrial"),
                str(sim_start_year):str(sim_end_year),
            ]
            .interpolate(axis=1)
            .values.squeeze()
        )[:sim_end_year-sim_start_year, None]

    return f


def get_aerosols() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (and cache) the solar and volcanic ERF timebound CSVs used to drive FaIR forcing.

    Sources: Chris Smith's :mod:`fair-add-hfc` for solar (``solar_erf_timebounds.csv``) and
    :mod:`fair-calibrate` for volcanic (``volcanic_erf_timebounds_cmip6.csv``). MD5 hashes
    are pinned so a substitute upstream artifact is refused.
    """
    solar_obj = pooch.retrieve(
        url = 'https://raw.githubusercontent.com/chrisroadmap/fair-add-hfc/main/data/solar_erf_timebounds.csv',
        known_hash = 'md5:98f6f4c5309d848fea89803683441acf',
    )

    volcanic_obj = pooch.retrieve(
        url = 'https://raw.githubusercontent.com/chrisroadmap/fair-calibrate/refs/heads/main/data/forcing/volcanic_erf_timebounds_cmip6.csv',
        known_hash = 'md5:4e467954c3ab5a18fc61cfa20ce41e74',
    )

    df_solar = pd.read_csv(solar_obj, index_col="year")
    df_volcanic = pd.read_csv(volcanic_obj)

    return df_solar, df_volcanic


def fill_aerosols(
    f: FAIR,
    df_configs: pd.DataFrame,
    df_solar: pd.DataFrame,
    df_volcanic: pd.DataFrame,
    sai_input: np.ndarray | None,
    sim_start_year: int,
    sim_end_year: int,
) -> FAIR:
    """Fill the ``Volcanic`` and ``Solar`` forcing arrays on ``f``, optionally injecting SAI ERF.

    When ``sai_input`` is provided (an ``(N, 2)`` array of ``[year, AOD]`` rows), the
    volcanic ERF for the SAI window is replaced with ``-0.28 * AOD`` to represent the
    aerosol-induced cooling. Solar forcing is loaded from the 1750-2300 timebound table; the
    forcing-trend shape ramps from 0 → 1 over 1750-2020 to match the CMIP6 calibration. Both
    series are then sliced to the requested simulation window and broadcast against the
    per-config scale factors from ``df_configs``.
    """
    # Aerosol injections
    df_volcanic = df_volcanic.copy()
    if sai_input is not None:
        sai_start_year = int(sai_input[0, 0])
        sai_end_year = int(sai_input[-1, 0])
        df_volcanic.loc[
            (df_volcanic['timebounds'] >= sai_start_year) &
            (df_volcanic['timebounds'] <= sai_end_year),
            'erf'
        ] = -0.28 * sai_input[:, 1]

    solar_forcing = np.zeros(551)
    volcanic_forcing = np.zeros(551)
    volcanic_forcing[:352] = df_volcanic.erf.values[:352]
    solar_forcing = df_solar["erf"].loc[1750:2300].values

    trend_shape = np.ones(551)
    trend_shape[:271] = np.linspace(0, 1, 271)

    volcanic_start_year = df_volcanic['year'].min()
    start_index = sim_start_year - volcanic_start_year
    end_index = sim_end_year - volcanic_start_year + 1

    fill(
        f.forcing,
        volcanic_forcing[start_index:end_index, None, None] * df_configs["fscale_Volcanic"].values.squeeze(),
        specie="Volcanic",
    )
    fill(
        f.forcing,
        solar_forcing[start_index:end_index, None, None] * df_configs["fscale_solar_amplitude"].values.squeeze()
        + trend_shape[start_index:end_index, None, None] * df_configs["fscale_solar_trend"].values.squeeze(),
        specie="Solar",
    )

    return f


# Named climate-response calibrations selectable via ``calibrate_to``.
#
# "constrained" (the default) keeps the per-member values of the FaIR v1.2.0
# calibrated-constrained ensemble that ``get_df_configs`` already samples, so the
# run spans the assessed climate-response uncertainty and ``get_global_temp``
# averages across it.
#
# "cesm" overwrites every member with one Cummins 3-layer parameter set tuned so
# FaIR's global ``tas`` tracks CESM2-WACCM. That was the default while CESM was the
# only regional model. It is retained — switchable, not deleted — because the
# regional emulators' *patterns* are still CESM-derived in part, so reproducing
# pre-2026-08 output requires it. Note its ``clim_kappa1`` (0.649) sits below the
# 5th percentile of the constrained ensemble, i.e. a more sensitive climate than
# almost the whole assessed range.
#
# Provenance of the CESM numbers is unrecorded in this repo; they predate it.
#
# Scope: ``calibrate_to`` sets only the no-SAI global *warming* magnitude. The SAI
# response chain — the GAUSS-fit Tg-per-K SO2 slopes, the per-K regional delta slopes,
# and the custom-injection impulse responses (CESM_params.mat) — remains CESM-derived
# regardless of the calibration. Under "constrained" the projected warming (hence the
# ``temp_diff`` that sizes injections) is smaller than under "cesm", so reported SO2
# amounts shrink and SAI start years move later relative to pre-flip output; the analytic
# and custom-injection modes stay mutually consistent because their CESM-derived factors
# cancel to first order.
CLIMATE_CALIBRATIONS: dict[str, dict[str, float] | None] = {
    "constrained": None,  # keep the ensemble's own per-member climate configs
    "cesm": {
        "clim_c1": 4.830712, "clim_c2": 6.397801, "clim_c3": 71.294445,
        "clim_kappa1": 0.649448, "clim_kappa2": 3.876076, "clim_kappa3": 0.900808,
        "clim_epsilon": 1.678583,
        "clim_gamma": 2.796757,
        "clim_sigma_eta": 0.713515,
        "clim_sigma_xi": 0.491656,
        "clim_F_4xCO2": 8.175245,
    },
}

DEFAULT_CALIBRATION = "constrained"


def apply_climate_calibration(df_configs: pd.DataFrame, calibrate_to: str) -> pd.DataFrame:
    """Return a copy of ``df_configs`` with the named calibration's climate columns applied.

    ``calibrate_to="constrained"`` keeps the input's values: the sampled ensemble keeps its
    own per-member climate response. Any other name must be a key of
    :data:`CLIMATE_CALIBRATIONS`, and collapses every member onto that single set. The input
    frame is never mutated, so calling with ``"cesm"`` and then ``"constrained"`` on the same
    ensemble frame yields the two distinct calibrations (not CESM twice).
    """
    if calibrate_to not in CLIMATE_CALIBRATIONS:
        raise ValueError(
            f"Unknown calibrate_to={calibrate_to!r}; "
            f"expected one of {sorted(CLIMATE_CALIBRATIONS)}."
        )
    df_configs = df_configs.copy()
    overrides = CLIMATE_CALIBRATIONS[calibrate_to]
    if overrides is not None:
        missing = set(overrides) - set(df_configs.columns)
        if missing:
            # Plain ``df[col] = value`` would silently append a new column on a typo'd
            # key while the real column keeps its ensemble values — refuse instead.
            raise ValueError(
                f"Calibration {calibrate_to!r} overrides columns not present in "
                f"df_configs: {sorted(missing)}."
            )
        for column, value in overrides.items():
            df_configs[column] = value
    return df_configs


def fill_in_configs(
    f: FAIR,
    df_configs: pd.DataFrame,
    co2_input: np.ndarray | None = None,
    scenario: str | None = None,
    calibrate_to: str = DEFAULT_CALIBRATION,
) -> FAIR:
    """Apply the climate-config + species-config calibrations and (optionally) override CO2 FFI emissions.

    ``calibrate_to`` selects a climate-response calibration from
    :data:`CLIMATE_CALIBRATIONS` (see there for what each means); the default keeps the
    calibrated-constrained ensemble's own per-member values. The species configs are
    filled from ``species_configs_properties_calibration1.2.0.csv`` and the per-config
    aerosol-direct, aerosol-indirect, ozone, and forcing-scale factors regardless.

    When ``co2_input`` is supplied (an ``(N, 2)`` array of ``[year, Gt CO2]`` rows covering
    ``[CANVAS_START_YEAR, CANVAS_END_YEAR]``), the CO2 FFI emissions are overridden in that
    window and CO2 AFOLU is zeroed so the sum still represents total CO2 emissions.
    """
    df_configs = apply_climate_calibration(df_configs, calibrate_to)

    ### 8b. Fill in climate_configs
    fill(f.climate_configs["ocean_heat_capacity"], df_configs.loc[:, "clim_c1":"clim_c3"].values)
    fill(
        f.climate_configs["ocean_heat_transfer"],
        df_configs.loc[:, "clim_kappa1":"clim_kappa3"].values,
    )
    fill(f.climate_configs["deep_ocean_efficacy"], df_configs["clim_epsilon"].values.squeeze())
    fill(f.climate_configs["gamma_autocorrelation"], df_configs["clim_gamma"].values.squeeze())
    fill(f.climate_configs["sigma_eta"], df_configs["clim_sigma_eta"].values.squeeze())
    fill(f.climate_configs["sigma_xi"], df_configs["clim_sigma_xi"].values.squeeze())
    fill(f.climate_configs["seed"], df_configs["seed"])
    fill(f.climate_configs["stochastic_run"], True)
    fill(f.climate_configs["use_seed"], True)
    fill(f.climate_configs["forcing_4co2"], df_configs["clim_F_4xCO2"])

    ### 8c. Fill in species_configs
    f.fill_species_configs(filename=FAIR_DIR / 'species_configs_properties_calibration1.2.0.csv')
    # carbon cycle
    fill(f.species_configs["iirf_0"], df_configs["cc_r0"].values.squeeze(), specie="CO2")
    fill(f.species_configs["iirf_airborne"], df_configs["cc_rA"].values.squeeze(), specie="CO2")
    fill(f.species_configs["iirf_uptake"], df_configs["cc_rU"].values.squeeze(), specie="CO2")
    fill(f.species_configs["iirf_temperature"], df_configs["cc_rT"].values.squeeze(), specie="CO2")

    # aerosol indirect
    fill(f.species_configs["aci_scale"], df_configs["aci_beta"].values.squeeze())
    fill(f.species_configs["aci_shape"], df_configs["aci_shape_so2"].values.squeeze(), specie="Sulfur")
    fill(f.species_configs["aci_shape"], df_configs["aci_shape_bc"].values.squeeze(), specie="BC")
    fill(f.species_configs["aci_shape"], df_configs["aci_shape_oc"].values.squeeze(), specie="OC")

    # aerosol direct
    for specie in [
        "BC", 
        "CH4", 
        "N2O",
        "NH3", 
        "NOx",
        "OC", 
        "Sulfur", 
        "VOC",
        "Equivalent effective stratospheric chlorine"
    ]:
        fill(f.species_configs["erfari_radiative_efficiency"], df_configs[f"ari_{specie}"], specie=specie)

    # forcing scaling
    for specie in [
        "CO2", 
        "CH4", 
        "N2O", 
        "Stratospheric water vapour",
        "Contrails", 
        "Light absorbing particles on snow and ice", 
        "Land use"
    ]:
        fill(f.species_configs["forcing_scale"], df_configs[f"fscale_{specie}"].values.squeeze(), specie=specie)
    # the halogenated gases all take the same scale factor
    for specie in [
        "CFC-11",
        "CFC-12",
        "CFC-113",
        "CFC-114",
        "CFC-115",
        "HCFC-22",
        "HCFC-141b",
        "HCFC-142b",
        "CCl4",
        "CHCl3",
        "CH2Cl2",
        "CH3Cl",
        "CH3CCl3",
        "CH3Br",
        "Halon-1211",
        "Halon-1301",
        "Halon-2402",
        "CF4",
        "C2F6",
        "C3F8",
        "c-C4F8",
        "C4F10",
        "C5F12",
        "C6F14",
        "C7F16",
        "C8F18",
        "NF3",
        "SF6",
        "SO2F2",
        "HFC-125",
        "HFC-134a",
        "HFC-143a",
        "HFC-152a",
        "HFC-227ea",
        "HFC-23",
        "HFC-236fa",
        "HFC-245fa",
        "HFC-32",
        "HFC-365mfc",
        "HFC-4310mee",
    ]:
        fill(f.species_configs["forcing_scale"], df_configs["fscale_minorGHG"].values.squeeze(), specie=specie)

    # ozone
    for specie in ["CH4", "N2O", "Equivalent effective stratospheric chlorine", "CO", "VOC", "NOx"]:
        fill(f.species_configs["ozone_radiative_efficiency"], df_configs[f"o3_{specie}"], specie=specie)

    # initial value of CO2 concentration (but not baseline for forcing calculations)
    fill(
        f.species_configs["baseline_concentration"], 
        df_configs["cc_co2_concentration_1750"].values.squeeze(), 
        specie="CO2"
    )

    if co2_input is not None:
        # Only fill in f.emissions for CO2 FFI between the canvas years
        f.emissions.loc[
            dict(
                specie="CO2 FFI",
                scenario=scenario,
                timepoints=np.arange(CANVAS_START_YEAR, CANVAS_END_YEAR+1) - 0.5
            )
        ] = co2_input[:, [1]]
        # Set AFOLU emissions to zero for the same time period
        # So that the sum of CO2 FFI and CO2 AFOLU is the total CO2 emissions
        f.emissions.loc[
            dict(
                specie="CO2 AFOLU",
                scenario=scenario,
                timepoints=np.arange(CANVAS_START_YEAR, CANVAS_END_YEAR+1) - 0.5
            )
        ] = 0

    return f


def initialize_run(f: FAIR, initial_conditions: dict | None = None) -> FAIR:
    """Initialize FaIR's concentration / forcing / temperature / emissions buffers in place.

    Without ``initial_conditions``, every state buffer starts at zero except the concentration
    array, which is seeded from the species' ``baseline_concentration``. When
    ``initial_conditions`` is supplied (a dict of pre-loaded arrays from a cached FaIR run),
    each buffer is initialized from that dict and the gas-partitioning state is restored, so
    the run can pick up from a previously-computed timestep without re-doing the spin-up.
    """
    ### 8d. Initial conditions
    if initial_conditions is None:
        initialise(f.concentration, f.species_configs["baseline_concentration"])
        initialise(f.forcing, 0)
        initialise(f.temperature, 0)
        initialise(f.cumulative_emissions, 0)
        initialise(f.airborne_emissions, 0)
    else:
        initialise(f.concentration, initial_conditions["concentration"])
        initialise(f.forcing, initial_conditions["forcing"])
        initialise(f.temperature, initial_conditions["temperature"])
        initialise(f.cumulative_emissions, initial_conditions["cumulative_emissions"])
        initialise(f.airborne_emissions, initial_conditions["airborne_emissions"])
        f.gas_partitions = initial_conditions["gas_partitions"].copy()

    return f


def get_runner(
    df_emis: pd.DataFrame,
    df_configs: pd.DataFrame,
    df_solar: pd.DataFrame,
    df_volcanic: pd.DataFrame,
    co2_input: np.ndarray | None,
    ssp_scenario: str,
    initial_conditions: dict | None = None,
    calibrate_to: str = DEFAULT_CALIBRATION,
) -> FAIR:
    """High-level builder used by :mod:`backend.baseline`: returns a ready-to-``run`` FaIR instance.

    Caches the post-``set_up_fair`` instance under ``f_no_sai/<ssp>.pkl`` so repeated calls
    for the same SSP skip the expensive scenario-setup step (which downloads RCMIP, builds
    all species buffers, and patches NOx). The cached instance is then routed through
    :func:`fill_aerosols` (with no SAI), :func:`fill_in_configs` (with the
    calibrated-constrained ensemble by default), and :func:`initialize_run` (with
    optional restart conditions) before being handed back to the caller.

    Accepts the dashboard's fancy SSP titles (``"SSP2-4.5"`` etc.) and translates them to
    the FaIR short keys via :data:`REVERSE_FANCY_SSP_TITLES`.
    """
    ssp_scenario = REVERSE_FANCY_SSP_TITLES[ssp_scenario]
    f_init_path = F_INIT_DIR / f"{ssp_scenario}.pkl"
    if not f_init_path.exists():
        f_init = set_up_fair(
            df_emis,
            df_configs,
            sim_start_year=SIM_START_YEAR,
            sim_end_year=SIM_END_YEAR,
            scenario=ssp_scenario
        )
        with open(f_init_path, "wb") as f:
            pkl.dump(f_init, f)
    else:
        with open(f_init_path, "rb") as f:
            f_init = pkl.load(f)

    f = fill_aerosols(
        f_init,
        df_configs,
        df_solar,
        df_volcanic,
        None,
        sim_start_year=SIM_START_YEAR,
        sim_end_year=SIM_END_YEAR
    )

    f = fill_in_configs(f, df_configs, co2_input, ssp_scenario, calibrate_to)

    f = initialize_run(f, initial_conditions)

    return f
