import pooch
import numpy as np
import pandas as pd
import pickle as pkl
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
FAIR_DIR = Path("fair_wrap")
F_INIT_DIR = FAIR_DIR / "f_no_sai"
F_INIT_DIR.mkdir(exist_ok=True)


def get_df_emis():
    rcmip_emissions_file = pooch.retrieve(
        url="doi:10.5281/zenodo.4589756/rcmip-emissions-annual-means-v5-1-0.csv",
        known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
    )
    df_emis = pd.read_csv(rcmip_emissions_file)
    return df_emis


def get_df_configs(ensemble=True):
    fair_params_1_2_0_obj = pooch.retrieve(
        url = 'https://zenodo.org/record/8399112/files/calibrated_constrained_parameters.csv',
        known_hash = 'md5:de3b83432b9d071efdd1427ad31e9076',
    )
    df_configs = pd.read_csv(fair_params_1_2_0_obj, index_col=0)
    if not ensemble:
        df_configs = pd.DataFrame(df_configs.median(axis=0)).T
    else:
        df_configs = df_configs.sample(NUM_MEMBERS, random_state=42)
    return df_configs


def get_dataframes():
    df_emis = get_df_emis()
    df_configs = get_df_configs()
    df_solar, df_volcanic = get_aerosols()
    return df_emis, df_configs, df_solar, df_volcanic


def set_up_fair(df_emis, df_configs, sim_start_year, sim_end_year, scenario=None):

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


def get_aerosols():
    solar_obj = pooch.retrieve(
        url = 'https://raw.githubusercontent.com/chrisroadmap/fair-add-hfc/main/data/solar_erf_timebounds.csv',
        known_hash = 'md5:98f6f4c5309d848fea89803683441acf',
    )

    volcanic_obj = pooch.retrieve(
        url = 'https://raw.githubusercontent.com/chrisroadmap/fair-calibrate/main/data/forcing/volcanic_ERF_1750-2101_timebounds.csv',
        known_hash = 'md5:c0801f80f70195eb9567dbd70359219d',
    )

    df_solar = pd.read_csv(solar_obj, index_col="year")
    df_volcanic = pd.read_csv(volcanic_obj)

    return df_solar, df_volcanic


def fill_aerosols(f, df_configs, df_solar, df_volcanic, sai_input, sim_start_year, sim_end_year):
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
    volcanic_forcing[:352] = df_volcanic.erf.values
    solar_forcing = df_solar["erf"].loc[1750:2300].values

    trend_shape = np.ones(551)
    trend_shape[:271] = np.linspace(0, 1, 271)

    volcanic_start_year = df_volcanic['timebounds'].min()
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


def fill_in_configs(f, df_configs, co2_input=None, scenario=None, calibrate_to="cesm"):
    if calibrate_to == "cesm":
        df_configs.loc[:, "clim_c1":"clim_c3"] = [4.830712, 6.397801, 71.294445]
        df_configs.loc[:, "clim_kappa1":"clim_kappa3"] = [0.649448, 3.876076, 0.900808]
        df_configs["clim_epsilon"] = 1.678583
        df_configs["clim_gamma"] = 2.796757
        df_configs["clim_sigma_eta"] = 0.713515
        df_configs["clim_sigma_xi"] = 0.491656
        df_configs["clim_F_4xCO2"] = 8.175245

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


def initialize_run(f, initial_conditions=None):
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


def get_runner(df_emis, df_configs,
               df_solar, df_volcanic,
               co2_input, ssp_scenario,
               initial_conditions=None,
               calibrate_to="cesm"):

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
