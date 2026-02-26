import fire
import numpy as np
import xesmf as xe
import xarray as xr
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import re

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.utils import open_xarray_datasets, save_processed_data
from scripts.variability import process_geojson_variability


# --- Configuration --- #
# Define patterns once for efficiency
PATTERNS = {
    # Regex : Experiment Name
    r"hist": "historical",
    r"SSP.*DEFAULT": "1.5",
    r"SSP.*LOWER-0.5": "1.0",
    r"SSP.*LOWER-1.0": "0.5",
    r"SSP": "baseline" # Catch-all for SSP if other tags missing
}

# Define unit conversion factors here for easy maintenance
UNIT_SCALERS = {
    "PRECT": 86400 * 1000,   # m/s -> mm/day
    "QFLX": 86400,         # kg/m2/s -> mm/day
}

# Map user variables to aggregation methods
AGGREGATION_RULES = {
    "tasmin": "min",
    "tasmax": "max",
    "icefrac": "mean", # Special case handled in logic
    "default": "mean"  # Fallback
}

# Define the mapping between the variable name and the GAUSS variable name
var2esm_var = {
        "tas": "TREFHT",
        "pr": "PRECT",
        "aod": "AODVISstdn",
        "tasmin": "TREFHTMN",
        "tasmax": "TREFHTMX",
        "tas_above_35": "TREFHT_above_35",
        "tas_above_40": "TREFHT_above_40",
        "tas_below_0": "TREFHT_below_0",
        "pr_above_10": "PRECT_above_10",
        "pr_above_20": "PRECT_above_20",
        "p-e": "QFLX",
        "icefrac": "ICEFRAC",
    }

# --- Helper Functions --- #

# Helper to check if a variable is cumulative (like "days above 30C")
def is_cumulative(var_name):
    return "above" in var_name or "below" in var_name

def identify_file(filename):
    """Parses a filename to find member ID and experiment type."""
    
    # Extract Member ID
    member_match = re.search(r'\.(\d{3})\.', filename)
    if not member_match:
        return None, None
    member = member_match.group(1)

    # Identify Experiment
    # Iterate through patterns; return the first one that matches
    for pattern, exp_name in PATTERNS.items():
        if re.search(pattern, filename, re.IGNORECASE):
            return member, exp_name
            
    return member, None


def process_file(path: Path, var_name: str, exp: str):
    """
    Loads a file, slices time, converts units, and applies masks.
    Returns the processed DataArray or None if error.
    """
    try:
        # Load Data
        ds = open_xarray_datasets(path, decode_times=True)
        x = ds[var_name]

        # Time Slicing
        # Select date range based on experiment type
        time_range = slice("1850", "2014") if "historical" in exp else slice("2015", "2099")
        x = x.sel(time=time_range)

        # Special Logic: ICEFRAC (Northern Hemisphere September)
        if var_name == "ICEFRAC":
            # Mask data: Keep North & Sept, set others to NaN
            # Note: drop=False maintains the original time index shape
            is_sept_north = (x.lat > 0) & (x.time.dt.month == 9)
            x = x.where(is_sept_north, np.nan)

        # Unit Conversion
        if var_name in UNIT_SCALERS:
            x = x * UNIT_SCALERS[var_name]

        # Cleanup
        if 'height' in x.coords:
            x = x.drop_vars('height')
            
        # Standardize Time Format (cftime -> datetime64)
        # Using a safe cast that handles the 'noleap' calendar within pandas range
        if x.indexes['time'].dtype == 'O': # Check if object/cftime
            x['time'] = x['time'].values.astype("datetime64[s]").astype("datetime64[ns]")
             
        return x

    except Exception as e:
        print(f"\n\033[91mError processing {path.name}: {e}\033[0m")
        return None


def pair_precc_precl_files(data_dir, prect_member2paths):
    """
    Pairs PRECC and PRECL files from data_dir and populates prect_member2paths.
    
    Args:
        data_dir (Path): The directory containing the NetCDF files.
        prect_member2paths (dict): A dictionary (preferably defaultdict(list)) to store the pairs.
                                   Structure: {member_id: [(path_c, path_l), ...]}
    """
    print(f"Scanning {data_dir} for file pairs...")

    # 1. Gather files and create "Canonical Keys"
    # The key is the filename with the variable removed, e.g., 'case.h0..2000-01.nc'
    files_c = {f.name.replace('PRECC', ''): f for f in data_dir.glob("*.h0.PRECC*.nc")}
    files_l = {f.name.replace('PRECL', ''): f for f in data_dir.glob("*.h0.PRECL*.nc")}

    # 2. Use Sets to find matches and orphans
    keys_c = set(files_c.keys())
    keys_l = set(files_l.keys())

    valid_keys = sorted(list(keys_c & keys_l)) # Intersection (Both exist)
    missing_l = keys_c - keys_l                # In C but not L
    missing_c = keys_l - keys_c                # In L but not C

    # 3. Process the Valid Pairs
    pattern_member = re.compile(r'\.(\d{3})\.')
    
    for key in valid_keys:
        file_c = files_c[key]
        file_l = files_l[key]
        
        # Extract member ID
        match = pattern_member.search(file_c.name)
        if match:
            member = match.group(1)
            # Append tuple pair
            prect_member2paths[member].append((file_c, file_l))
        else:
            print(f"Skipping {file_c.name}: Could not extract member ID.")

    # 4. Report Results
    print("-" * 40)
    print(f"Processing Complete.")
    print(f"  Valid Pairs Found: {len(valid_keys)}")
    
    if missing_l:
        print(f"\n  [WARNING] {len(missing_l)} files have PRECC but are MISSING PRECL:")
        for key in sorted(list(missing_l)):
            print(f"    - {files_c[key].name}")

    if missing_c:
        print(f"\n  [WARNING] {len(missing_c)} files have PRECL but are MISSING PRECC:")
        for key in sorted(list(missing_c)):
            print(f"    - {files_l[key].name}")
    print("-" * 40)

    return prect_member2paths


def process_and_export_prect_per_file(prect_member2paths, output_dir, overwrite=False):
    """
    Reads pairs of PRECC/PRECL files, sums them to PRECT,
    and exports a SEPARATE file for each pair immediately.
    If overwrite is False, skip export if file already exists.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for member, file_pairs in prect_member2paths.items():
        print(f"Processing Member: {member} ({len(file_pairs)} pairs)")
        
        for file_c, file_l in file_pairs:
            try:
                # Generate Filename before processing, so we can check for existence
                new_name = file_c.name.replace('PRECC', 'PRECT')
                final_out_path = output_path / new_name

                if final_out_path.exists() and not overwrite:
                    print(f"  Skipping: {final_out_path.name} (already exists, use overwrite=True to replace)")
                    continue

                # Open individual datasets
                ds_c = open_xarray_datasets(file_c)
                ds_l = open_xarray_datasets(file_l)
                
                # Calculate PRECT
                prect_sum = ds_c['PRECC'] + ds_l['PRECL']

                # Float32 Optimization
                if prect_sum.dtype == 'float64':
                    prect_sum = prect_sum.astype('float32')

                # Construct Output Dataset
                ds_total = ds_c.copy()
                ds_total['PRECT'] = prect_sum
                
                # Update attributes
                ds_total['PRECT'].attrs = ds_c['PRECC'].attrs
                ds_total['PRECT'].attrs['long_name'] = 'Total Precipitation (PRECC + PRECL)'
                
                # Drop original variables
                ds_total = ds_total.drop_vars(['PRECC', 'PRECL'], errors='ignore')

                # Export
                print(f"  Exporting: {final_out_path.name}")
                
                encoding = {var: {'zlib': True, 'complevel': 5} for var in ds_total.data_vars}
                ds_total.to_netcdf(final_out_path, encoding=encoding)
                
                # Cleanup immediate memory
                ds_total.close()
                ds_c.close()
                ds_l.close()
                
            except Exception as e:
                print(f"  Error processing pair {file_c.name}: {e}")
                continue

    print("\nAll individual file exports complete.")


def get_yearly_aggregate(ds, var_name, esm_var):
    """
    Resamples data to yearly, applies the correct aggregation, 
    and REPLACES years that do not have full 12-month coverage with NaN.
    """
    print(f" var_name: {var_name}")

    # Resample
    resampler = ds.resample(time="1YE")

    # Check Completeness (Count valid months per year)
    # This creates a boolean mask: True if year has 12 valid months, False otherwise
    counts = resampler.count(dim="time")

    # Default requirement: 12 months. 
    # EXCEPTION: ICEFRAC (which is Sept-only in your logic) should not be filtered by 12 months.
    if var_name == "icefrac":
        is_complete = (counts >= 1) # Keep if we have at least the 1 month (Sept)
    else:
        is_complete = (counts == 12)

    # --- REPORT DROPPED YEARS ---
    # If is_complete is a Dataset, we must pull out the variable's mask
    # to get a 1D array aligned with 'time'
    if isinstance(is_complete, xr.Dataset):
        # Use the variable name to get the boolean DataArray
        mask_da = is_complete[esm_var]
    else:
        # It's already a DataArray
        mask_da = is_complete

    # Reduce 3D (Time, Lat, Lon) -> 1D (Time) for logging
    # We check if the year is valid "everywhere" or just pick a reference point.
    # Since missing files affect the whole globe, selecting the first spatial point is safe and fast.
    # We use .isel to grab the first lat/lon index.
    if 'lat' in mask_da.dims and 'lon' in mask_da.dims:
        mask_1d = mask_da.isel(lat=-21, lon=-57)
    elif 'lat' in mask_da.dims: # Handle edge case of 2D data
        mask_1d = mask_da.isel(lat=-21)
    else:
        mask_1d = mask_da

    # Identify years with incomplete data
    dropped_mask = ~mask_1d.values 

    if dropped_mask.any():
        # Now we can safely index the 1D time array
        dropped_years_vals = ds.resample(time="1YE").mean().time[dropped_mask].dt.year.values
        print(f"\033[93m  [WARNING] {len(dropped_years_vals)} incomplete years for {var_name}: {dropped_years_vals}\033[0m")
    # ----------------------------

    # Determine Method & Aggregate
    if is_cumulative(var_name):
        agg = resampler.sum()
    else:
        method = AGGREGATION_RULES.get(var_name, AGGREGATION_RULES["default"])

        if method == "min":
            agg = resampler.min()
        elif method == "max":
            agg = resampler.max()
        elif method == "mean":
            agg = resampler.mean()
            # Special fix for icefrac NaN handling
            if var_name == "icefrac":
                 agg = agg.fillna(0)
        else:
            agg = resampler.mean()

    # Filter Incomplete Years
    # where(mask, drop=True) keeps where mask is True, drops the rest.
    filtered_agg = agg.where(is_complete, drop=True)
    # Optional: Warn if data was dropped (Debugging)
    dropped_years = len(agg.time) - len(filtered_agg.time)
    if dropped_years > 0:
        print(f" Dropped {dropped_years} incomplete years for {var_name}")

    return filtered_agg


def validate_continuity(ds):
    """Asserts that monthly data is continuous (diff is exactly 1 month)."""
    time_diffs = np.diff(ds.time.values.astype("datetime64[M]"))
    one_month = np.timedelta64(1, 'M')
    if not np.all(time_diffs == one_month):
        raise ValueError("\033[91mData is not temporally continuous (missing months detected).\033[0m")

def merge_historical_and_baseline(exp2member2data_combined):
    """
    Merges 'historical' and 'baseline' into a single superset of members.

    Logic:
    1. Overlapping Members (e.g. Hist 1 & Base 1): Stitched together (1850-2100).
    2. Historical-Only (e.g. Hist 2): kept as is (1850-2014).
    3. Baseline-Only (e.g. Base 7): kept as is (2015-2100).

    The result is stored in 'baseline', and 'historical' is removed.
    """
    print("Merging Historical and Baseline data (Superset Mode)...")

    historical_data = exp2member2data_combined.pop('historical', {})
    baseline_data = exp2member2data_combined.setdefault('baseline', {})

    if not historical_data:
        print("  Note: No historical data found. Keeping baseline as is.")
        return

    # Identify all unique member IDs involved
    hist_ids = set(historical_data.keys())
    base_ids = set(baseline_data.keys())
    all_ids = sorted(list(hist_ids | base_ids))

    count_stitched = 0
    count_hist_only = 0
    count_base_only = 0

    for member in all_ids:
        # Case 1: Member exists in BOTH (Stitch them)
        if member in hist_ids and member in base_ids:
            ds_hist = historical_data[member].sel(time=slice(None, 2014))
            ds_base = baseline_data[member]

            try:
                # Concatenate along time
                ds_combined = xr.concat(
                    [ds_hist, ds_base],
                    dim="time",
                    join="outer"
                ).sortby("time")

                # Instead of using .dt accessor (which is unavailable for DataArray without datetime objects),
                # use values and numpy directly.
                # This fix properly extracts years even if xarray does not supply .dt
                years = ds_combined["time"].values
                # np.datetime64: convert to integer years
                # Handle if time is not datetime64
                try:
                    # This will succeed for datetimes
                    years_int = np.array([pd.Timestamp(y).year for y in years])
                except Exception:
                    try:
                        # Sometimes time values are already numeric year
                        years_int = years.astype(int)
                    except Exception:
                        years_int = np.arange(len(years))

                # Check for gaps
                if len(years_int) > 1 and np.any(np.diff(years_int) > 1):
                    print(f"\033[93m    [WARNING] Gap detected in stitched member {member}\033[0m")

                baseline_data[member] = ds_combined
                count_stitched += 1

            except Exception as e:
                print(f"\033[91m    Error stitching member {member}: {e}\033[0m")

        # Case 2: Member exists ONLY in Historical
        elif member in hist_ids:
            # Move it to baseline dict so it's included in the final ensemble
            baseline_data[member] = historical_data[member]
            count_hist_only += 1

        # Case 3: Member exists ONLY in Baseline
        elif member in base_ids:
            # It's already in baseline_data, just count it
            count_base_only += 1

    print("-" * 40)
    print(f"Merge Complete. Total Members: {len(baseline_data)}")
    print(f"  - Full Time Series (Stitched): {count_stitched}")
    print(f"  - Historical Only (1850-2014): {count_hist_only}")
    print(f"  - Baseline Only   (2015-2100): {count_base_only}")
    print("-" * 40)


def get_robust_regridder(ds_sample, target_grid):
    """
    Initializes an xesmf regridder. 
    Includes fallback logic for datasets with malformed lat/lon coordinates.
    """
    try:
        # Standard initialization
        return xe.Regridder(ds_sample, target_grid, 'bilinear', periodic=True)
    except Exception as e:
        print(f"\033[93m  Standard regridder failed ({e}). Attempting coordinate fix...\033[0m")
        # Fallback: Force standard lat/lon arrays if metadata is broken
        ds_fix = ds_sample.copy()
        ds_fix['lat'] = np.linspace(-90, 90, len(ds_sample.lat))
        ds_fix['lon'] = np.linspace(0, 360, len(ds_sample.lon))
        
        return xe.Regridder(ds_fix, target_grid, 'bilinear', periodic=True, ignore_degenerate=True)


def compute_and_save_variability(exp2member2data_combined, var_name, output_dir, overwrite=False):
    """
    Calculates and saves the model internal variability based on the 'baseline' experiment.
    
    Method:
    1. Global Weighted Mean (Latitudinal)
    2. Detrending (removing linear trend over time)
    3. Standard Deviation of the residuals (across time and members)
    """
    print("Computing model internal variability...")
    
    # 1. Extract Baseline Data
    if 'baseline' not in exp2member2data_combined:
        print("\033[93mWarning: No 'baseline' experiment found. Skipping variability calculation.\033[0m")
        return

    # Dictionary of {member: DataArray} -> List of DataArrays
    baseline_members = list(exp2member2data_combined['baseline'].values())
    
    if not baseline_members:
        print("Warning: Baseline experiment is empty.")
        return

    # 2. Concatenate Members
    # Result dim: (member, time, lat, lon)
    ds_baseline = xr.concat(baseline_members, dim="member", join="outer")

    # 3. Calculate Global Weighted Mean
    # Ensure we are working with the variable, not a Dataset wrapper
    if isinstance(ds_baseline, xr.Dataset):
        da = ds_baseline[var_name] # Extract DataArray
    else:
        da = ds_baseline # It is already a DataArray

    # Cosine weighting for latitude
    weights = np.cos(np.deg2rad(da.lat))
    global_mean = da.weighted(weights).mean(dim=("lat", "lon"))

    # 4. Filter Time Range (2025-2099)
    # Using slice ensures we don't crash if range is slightly smaller
    global_mean = global_mean.sel(time=slice(2025, 2099))

    # 5. Detrending (Linear)
    # We fit a degree-1 polynomial along the 'time' dimension
    print("  Detrending baseline data...")
    coeffs = global_mean.polyfit(dim='time', deg=1)
    
    # Evaluate the trend: y_trend = a*t + b
    fitted_trend = xr.polyval(global_mean['time'], coeffs.polyfit_coefficients)
    
    # Subtract trend to get residuals
    detrended_residuals = global_mean - fitted_trend

    # 6. Compute Standard Deviation
    # Taking std of all residuals across all members and time steps
    std_value = detrended_residuals.std().item()

    # 7. Save
    out_path = Path(output_dir) / "model_internal_variability.npy"
    print(f"  Internal Variability (std): {std_value:.4f}")
    if out_path.exists() and not overwrite:
        print(f"  File already exists and overwrite is False: {out_path}")
        print(f"  Skipping export of model internal variability.")
        return std_value, weights
    print(f"  Saving model internal variability to {out_path}")
    np.save(out_path, std_value)
    
    return std_value, weights


def compute_and_save_grid_variability(
    exp2member2data_combined: dict[str, dict[str, list[xr.Dataset]]],
    var: str,
    esm_var: str,
    output_dir: Path,
    overwrite: bool = False
) -> xr.Dataset:
    """
    Calculates and saves the GRID-LEVEL model internal variability based on the 'baseline' experiment.
    
    Method:
    1. Select Baseline Data (2025-2099)
    2. Detrend per-pixel (remove linear trend at each lat/lon)
    3. Compute Standard Deviation of residuals over the reference period (2025-2039)
    4. Export as NetCDF
    """
    print("Computing grid-level model internal variability...")
    
    # 1. Extract Baseline Data
    if 'baseline' not in exp2member2data_combined:
        print("\033[93mWarning: No 'baseline' experiment found. Skipping grid variability.\033[0m")
        return

    baseline_members = list(exp2member2data_combined['baseline'].values())
    
    if not baseline_members:
        print("Warning: Baseline experiment is empty.")
        return

    # Concatenate Members -> (member, time, lat, lon)
    ds_baseline = xr.concat(baseline_members, dim="member", join="outer")

    # Extract DataArray if needed
    if isinstance(ds_baseline, xr.Dataset):
        da = ds_baseline[esm_var]
    else:
        da = ds_baseline

    # 2. Slice Full Period for Detrending (2025-2099)
    # We use the full period to get a robust trend estimate, even if we calculate std on a shorter window later.
    regional = da.sel(time=slice(2025, 2099))
    
    print("  Calculating per-pixel linear trend (this may take a moment)...")
    
    # 3. Vectorized Detrending
    # polyfit computes the slope and intercept for every (lat, lon) simultaneously
    trend_coeffs = regional.polyfit(dim="time", deg=1)
    
    # Evaluate the trend line: y = mx + b
    fitted_trend = xr.polyval(regional["time"], trend_coeffs.polyfit_coefficients)
    
    # Subtract trend to get the "noise"
    detrended_residuals = regional - fitted_trend

    # 4. Compute Variability (Standard Deviation)
    # We focus on the near-term window (2025-2039) to estimate variability
    print("  Computing standard deviation (2025-2039)...")
    reference_window = detrended_residuals.sel(time=slice(2025, 2039))
    
    # Calculate std across BOTH time and ensemble members
    grid_variability = reference_window.std(dim=("time", "member"))

    # 5. Export
    # Convert back to Dataset for saving with metadata
    ds_out = grid_variability.to_dataset(name=var)
    
    # Add attributes for reproducibility
    ds_out[var].attrs = da.attrs
    ds_out[var].attrs['description'] = "Internal Variability (Std Dev of detrended residuals 2025-2039)"
    
    out_path = Path(output_dir) / "grid_level_model_internal_variability.nc"
    if out_path.exists() and not overwrite:
        print(f"  File already exists and overwrite is False: {out_path}")
        print(f"  Skipping export of grid-level model internal variability.")
        return ds_out
    print(f"  Saving grid variability to {out_path}")
    ds_out.to_netcdf(out_path)
    return ds_out


# --- Main Function --- #
def process_monthly(var, data_dir, output_dir, overwrite=False):
    """
    Process monthly Simulator data and calculate model internal variability for each variable.
    (it does not calculate model internal variability for sea ice. It is done in variability.py)

    Args:
        var: Variable to process
        data_dir: Directory containing Simulator data
        output_dir: Directory to save processed data
        overwrite: Whether to overwrite existing files
    """
    print(f"var: {var}")
    print("="*100)

    # Validation & Setup
    if var not in var2esm_var:
        raise ValueError(f"\033[91mvar must be one of {list(var2esm_var.keys())}\033[0m")

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"\033[91mData directory does not exist: {data_dir}\033[0m")

    output_dir = Path(output_dir) / var
    output_dir.mkdir(exist_ok=True, parents=True)

    esm_var = var2esm_var[var]
    print(f"--- Processing {var} (Variable: {esm_var}) ---")

    # Special Pre-processing (PRECT)
    # If PRECT, we must generate the files first so they can be "found" in the next step.
    if esm_var == "PRECT":
        prect_out_dir = data_dir / "PRECT"
        # Check if we actually need to generate them (skip if exists and not overwrite)
        # For now, assuming we always run the pairing check as it's fast
        print(">> Pairing PRECC and PRECL files...")
        prect_member2paths = defaultdict(list)
        pair_precc_precl_files(data_dir, prect_member2paths)
        # for member, file_pairs in prect_member2paths.items():
        #     for file_c, file_l in file_pairs:
        #         print(f" >>> file_c: {file_c},\n <<< file_l: {file_l}")
        
        print(f">> Exporting unified PRECT files to {prect_out_dir}...")
        process_and_export_prect_per_file(prect_member2paths, prect_out_dir, overwrite)
        print(">> PRECT generation complete.")

    # Path Gathering
    print(">> Scanning for files...")
    search_pattern = f"*.h0.{esm_var}.*.nc"
    # Search in main dir AND subdirectory (common for PRECT or organized data)
    cdf_paths = sorted(data_dir.glob(search_pattern)) + \
                sorted((data_dir / esm_var).glob(search_pattern))
    
    if not cdf_paths:
        print(f"\033[93mWarning: No files found for {esm_var}\033[0m")
        return
    
    print(f"   Found {len(cdf_paths)} files.")

    # Loading & Parsing (Member/Exp Organization)
    exp2member2data = defaultdict(lambda: defaultdict(list))
    exp2member2paths = defaultdict(lambda: defaultdict(list))
    
    for cdf_path in tqdm(cdf_paths, desc="Loading Files"):
        member, exp = identify_file(cdf_path.name)
        
        if not member:
            print(f"\033[91mSkipping {cdf_path.name}: No member ID found.\033[0m")
            continue
        if not exp:
            print(f"\033[91mSkipping {cdf_path.name}: Unknown experiment.\033[0m")
            continue

        # Use the cleaner process_file helper from previous steps
        x_processed = process_file(cdf_path, esm_var, exp)
        
        if x_processed is not None:
            exp2member2data[exp][member].append(x_processed)
            exp2member2paths[exp][member].append(cdf_path)
            

    # Combination & Aggregation
    print(">> Combining & Aggregating...")

    # Create grid for output (lat/lon)
    # Get correct latitudes from CESM2-WACCM (center of bounds)
    correct_lat = np.load(output_dir.parent / "correct_lat.npy")
    common_grid = {
        'lon': np.linspace(0, 358.75, 288),
        'lat': correct_lat
    }

    exp2member2data_combined = defaultdict(lambda: defaultdict(list))
    for exp, member2data in tqdm(exp2member2data.items(), desc="Experiments"):
        for member, data_list in member2data.items():
            # print(f"exp: {exp}")
            # print(f"member: {member}")
            # for data in data_list:
            #     print(f"data.time: {data.time}")
            # print(f"datalist size: {len(data_list)}")
            try:
                # A. Combine time steps
                ds_combined = xr.combine_by_coords(data_list, combine_attrs='drop_conflicts')
                
                # B. Validation
                validate_continuity(ds_combined)

                # C. Yearly Aggregation
                # print(f"  Getting yearly aggregate for {var} experiment: {exp} member: {member} ...")
                ds_yearly = get_yearly_aggregate(ds_combined, var, esm_var)

                # D. Convert Time Index (Datetime -> Year Integer)
                # Extract years safely from the datetime index
                years = ds_yearly.time.dt.year.values
                ds_yearly = ds_yearly.assign_coords(time=years)
                
                exp2member2data_combined[exp][member] = ds_yearly

                # E. Save (Implementation depends on your specific save needs)
                # final_path = output_dir / f"{exp}_{member}_{var}.nc"
                # ds_yearly.to_netcdf(final_path)
                
            except Exception as e:
                print(f"\033[91mError processing {exp} member {member}: {e}\033[0m")


    # output_dir is passed from process_monthly
    std_value, weights = compute_and_save_variability(exp2member2data_combined, esm_var, output_dir, overwrite)
    
    # Compute grid-level model internal variability
    grid_level_variability = compute_and_save_grid_variability(exp2member2data_combined, var, esm_var, output_dir, overwrite)

    # Calcualting regional variability
    process_geojson_variability(output_dir.parent / "geojsons/original", grid_level_variability, var, weights)

    # Stitch Historical + Baseline
    merge_historical_and_baseline(exp2member2data_combined)

    print("Regridding and creating ensemble DataArrays...")

    # Initialize Regridder ONCE
    first_exp = next(iter(exp2member2data_combined.values()))
    first_member_ds = next(iter(first_exp.values()))
    
    # We define standard coords just in case we need to force them on everyone
    standard_lat = np.linspace(-90, 90, len(first_member_ds.lat))
    standard_lon = np.linspace(0, 360, len(first_member_ds.lon))
    
    # Try to build regridder; keep track if we used the fallback
    use_coord_fix = False
    try:
        regridder = xe.Regridder(first_member_ds, common_grid, 'bilinear', periodic=True)
    except Exception:
        print("\033[93m  Standard regridder failed. Switching to robust mode (forcing coords).\033[0m")
        use_coord_fix = True
        # Create a temp copy just to build the weights
        ds_fix = first_member_ds.copy()
        ds_fix['lat'] = standard_lat
        ds_fix['lon'] = standard_lon
        regridder = xe.Regridder(ds_fix, common_grid, 'bilinear', periodic=True, ignore_degenerate=True)

    # Process All Experiments
    final_output_data = {} 

    for exp, member2data in tqdm(exp2member2data_combined.items(), desc="Regridding Experiments"):
        
        regridded_members = []

        for member, ds in member2data.items():
            
            # If we had to fix the regridder, we MUST fix the input data to match it
            if use_coord_fix:
                ds = ds.copy() # Don't modify original data in the dict
                ds['lat'] = standard_lat
                ds['lon'] = standard_lon

            # Apply regridder
            regridded = regridder(ds)
            regridded = regridded.assign_coords(member=member)
            regridded_members.append(regridded)

        if regridded_members:
            # Concatenate
            ds_ensemble = xr.concat(regridded_members, dim="member", join="outer")
            final_output_data[exp] = ds_ensemble

    print("Regridding complete.")  

    print("Finalizing and saving data...")
    
    # Common metadata to add
    metadata_coords = {
        "ssp": ["ssp245"],
        "model": ["CESM2-WACCM"]
    }

    for exp, ds_ensemble in final_output_data.items():
        
        # Save "All Members" (for specific variables)
        # We do this BEFORE averaging so we keep the member dimension
        if var in ["icefrac", "tas"]:
            out_path_members = output_dir / f"output_gauss-{exp}-all-members.nc"
            print(f"  Saving all-members data to {out_path_members.name}...")
            
            # Rename variable for consistency before saving
            ds_members_renamed = ds_ensemble.rename_vars({esm_var: var})
            
            # Save
            if overwrite or not out_path_members.exists():
                save_processed_data(ds_members_renamed, out_path_members, overwrite)
            else:
                print(f"    Skipping (file exists): {out_path_members.name}")

        # Compute Ensemble Mean
        # The user code had an if/else that did the same thing for both cases.
        # We simplify it to just one operation.
        ds_mean = ds_ensemble.mean(dim="member")

        # Add Metadata Coordinates
        ds_mean = ds_mean.expand_dims({"ssp": 1, "model": 1})
        ds_mean = ds_mean.assign_coords(
            ssp=('ssp', metadata_coords["ssp"]),
            model=('model', metadata_coords["model"]),
            lon=(ds_mean.lon % 360) # Ensure 0-360 longitude
        )

        # Rename Variable
        ds_mean = ds_mean.rename_vars({esm_var: var})

        # Special Calculation: P-E
        # This assumes 'var' is p-e and the current data is Evaporation (or similar),
        # requiring subtraction from Precipitation (pr).
        if var == "p-e":
            pr_path = output_dir.parent / "pr" / f"output_gauss-{exp}.nc"
            
            if pr_path.exists():
                print(f"  Calculating P-E using {pr_path.name}...")
                try:
                    ds_pr = xr.open_dataset(pr_path, decode_times=True)
                    # Calculate: P - E
                    # Note: Ensure units match (both should be mm/day based on earlier processing)
                    ds_mean[var] = ds_pr["pr"] - ds_mean[var]
                except Exception as e:
                    print(f"\033[91m  [ERROR] Failed to calculate P-E: {e}\033[0m")
            else:
                print(f"\033[93m  [WARNING] Cannot calculate P-E. Missing 'pr' file: {pr_path}\033[0m")

        # Save Final Averaged File
        out_path_mean = output_dir / f"output_gauss-{exp}.nc"
        print(f"  Saving ensemble mean to {out_path_mean.name}")
        
        if overwrite or not out_path_mean.exists():
            save_processed_data(ds_mean, out_path_mean, overwrite)
        else:
            print(f"    Skipping (file exists): {out_path_mean.name}")

    print(f"--- Processing {var} Complete ---")


if __name__ == "__main__":
    fire.Fire(process_monthly)
