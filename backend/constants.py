import os
import geopandas as gpd
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap


REGIONAL_MODEL_NAMES = [
    "CESM2-WACCM"
]
NUM_EMULATORS = 100
# CESM2-WACCM dimensions
NUM_LAT = 192
NUM_LON = 288
MIN_SAI_START = 2035

MIN_TARGET = 0.5
MAX_TARGET = 2.5
STEP_TARGET = 0.1

VAR2INFO = {
    "tas": ("Temperature in °C", "Temperature (°C)", "coolwarm", "#00FF00"),
    "p-e": ("Water Availability in mm/day", "Water Availability (mm/day)", LinearSegmentedColormap.from_list(
        "brown_to_white_to_blue", ["brown", "khaki", "white", "deepskyblue", "mediumblue"], N=100
        ), "red"),
    "tasmin": ("Minimum Temperature in °C", "Min Temperature (°C)", "coolwarm", "#00FF00"),
    "tasmax": ("Maximum Temperature in °C", "Max Temperature (°C)", "coolwarm","#00FF00"),
    "exposure_above_40": ("Person-Days Above 40°C (Millions)", "Exposure to > 40°C (million person-days)", LinearSegmentedColormap.from_list(
            "white_to_red", ["white", "red"], N=100
        ), "#00FF00"),
    "exposure_above_35": ("Person-Days Above 35°C (Millions)", "Exposure to > 35°C (million person-days)", LinearSegmentedColormap.from_list(
            "white_to_red", ["white", "red"], N=100
        ), "#00FF00"),
    "exposure_below_0": ("Person-Days Below 0°C (Millions)", "Exposure to < 0°C (million person-days)", LinearSegmentedColormap.from_list(
            "white_to_blue", ["white", "deepskyblue", "mediumblue"], N=100
        ), "#00FF00"),
    "exposure_above_10": ("Person-Days Above 10mm/day (Millions)", "Exposure to > 10mm (million mm-days)", LinearSegmentedColormap.from_list(
            "white_to_blue", ["white", "deepskyblue", "mediumblue"], N=100
        ), "red"),
    "exposure_above_20": ("Person-Days Above 20mm/day (Millions)", "Exposure to > 20mm (million mm-days)", LinearSegmentedColormap.from_list(
            "white_to_blue", ["white", "deepskyblue", "mediumblue"], N=100
        ), "red"),
}

exposurevar2var = {
    "exposure_above_40": "tas_above_40",
    "exposure_above_35": "tas_above_35",
    "exposure_below_0": "tas_below_0",
    "exposure_above_10": "pr_above_10",
    "exposure_above_20": "pr_above_20"
}
