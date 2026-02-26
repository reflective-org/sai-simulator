import fire
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import time

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from backend.utils import open_xarray_datasets

def day_to_month(var, data_dir, ignore_existing=False):
    # Start profiling
    overall_start_time = time.time()

    var2gauss_var = {
        "tas": "TREFHT",
        "pr": "PRECT",
    }
    input_dir = Path(data_dir)

    if var not in var2gauss_var:
        raise ValueError(f"var must be one of {list(var2gauss_var.keys())}")

    gauss_var = var2gauss_var[var]

    # Set output directory as subdirectory with the var name
    output_dir = input_dir / gauss_var
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. DEFINE ONLY THE STEMS
    # The key is the scenario name, the value is the base filename prefix.
    scenarios = {
        "daily_historical": "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM",
        "daily_baseline":   "b.e21.BWSSP245.f09_g17.release-cesm2.1.3.WACCM-MA-1deg",
        "daily_1.5":        "b.e21.BWSSP245.f09_g17.release-cesm2.1.3.WACCM-MA-1deg.SSP245-MA-GAUSS-DEFAULT",
        "daily_1.0":        "b.e21.BWSSP245.f09_g17.release-cesm2.1.3.WACCM-MA-1deg.SSP245-MA-GAUSS-LOWER-0.5",
        "daily_0.5":        "b.e21.BWSSP245.f09_g17.release-cesm2.1.3.WACCM-MA-1deg.SSP245-MA-GAUSS-LOWER-1.0",
    }
    
    for key, stem in scenarios.items():
        scenario_start_time = time.time()
        print(f"key: {key}")
        print(f"stem: {stem}")
        print("-"*100)
        # 2. DYNAMICALLY BUILD GLOB PATTERNS
        # Use the specific fix for 'baseline' to avoid the greedy glob issue
        if key == "daily_baseline":
            glob_pattern = f"{stem}.[0-9]*h1.{gauss_var}.*nc"
        else:
            glob_pattern = f"{stem}*h1*.{gauss_var}.*nc"
            
        paths = sorted(input_dir.glob(glob_pattern))

        if not paths:
            print(f"No data found for {key}")
            continue

        print(f"Processing {key} data")
        
        # 3. DYNAMICALLY CALCULATE INDEX
        # The member ID is always the segment immediately following the stem.
        # We can find its index by counting the dots in the stem.
        # e.g., "a.b.c" has 2 dots, so it occupies indices 0, 1, 2. The next item is index 3.
        member_index = stem.count(".") + 1
        
        member2paths = defaultdict(list)

        for path in paths:
            print(f"path: {path}")
            # Keep your specific baseline filter if needed
            if "baseline" in key and "50101" not in str(path):
                continue
                
            # Extract member using the calculated index
            try:
                member = path.name.split(".")[member_index]
                member2paths[member].append(path)
                print(f"member: {member}")
            except IndexError:
                print(f"Skipping malformed filename: {path.name}")
                continue

        # Processing logic continues below...
        members = sorted(list(member2paths.keys()))
        print("-"*100)
        print(f"--> {len(members)} members founded: {', '.join(members)}")
        print("-"*100)
        for member in tqdm(members):
            member_start_time = time.time()
            print("\n" + "="*100)
            print(f"member: {member}")
            print("="*100 + "\n")
            paths = member2paths[member]

            # 1. Check if output files already exist
            # Use glob to find existing files for this member/variable combination
            if not ignore_existing:
                # Define tasks first to know what suffixes to check for
                temp_tasks = []
                if gauss_var == "TREFHT":
                    temp_tasks = ["_above_40", "_above_35", "_below_0"]
                elif gauss_var == "PRECT":
                    temp_tasks = ["_above_10", "_above_20"]

                # Check if all expected output files exist
                all_exist = True
                for suffix in temp_tasks:
                    
                    new_var_name = gauss_var + suffix
                    # Set output directory as subdirectory with the var name
                    output_dir = input_dir / new_var_name
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Use glob to find any file matching this pattern (with any time range)
                    pattern = f"{stem}.{member}.cam.h0.{new_var_name}.*.nc"
                    existing_files = list(output_dir.glob(pattern))
                    if not existing_files:
                        all_exist = False
                        break

                if all_exist:
                    print(f"\n>>> All output files for member {member} already exist, skipping processing entirely")
                    continue

            # 2. Define the Tasks
            # A dictionary where Key = Suffix, Value = Lambda function for the math
            # This allows us to easily add/remove metrics without changing the loop structure
            tasks = {}

            if gauss_var == "TREFHT":
                # Threshold tasks (Always run)
                tasks["_above_40"] = lambda d: (d > (40 + 273.15)).resample(time="1MS").sum()
                tasks["_above_35"] = lambda d: (d > (35 + 273.15)).resample(time="1MS").sum()
                tasks["_below_0"]  = lambda d: (d < (0 + 273.15)).resample(time="1MS").sum()

                # Historical-only tasks
                # if "historical" in stem:
                #     tasks["MX"] = lambda d: d.resample(time="1MS").max()
                #     tasks["MN"] = lambda d: d.resample(time="1MS").min()

            elif gauss_var == "PRECT":
                # Convert m/s to mm/day for the threshold check: * 1000 (mm) * 86400 (day)
                tasks["_above_10"] = lambda d: (d * 86400000 > 10).resample(time="1MS").sum()
                tasks["_above_20"] = lambda d: (d * 86400000 > 20).resample(time="1MS").sum()

            # 3. Accumulators
            # Dictionary to hold lists of PROCESSED monthly chunks.
            # Structure: { "MX": [ds1, ds2...], "_above_40": [ds1, ds2...] }
            accumulators = defaultdict(list)

            # 4. Process Files Individually
            for path in tqdm(paths, desc="Processing files"):
                file_start_time = time.time()
                try:
                    ds_raw = open_xarray_datasets(path)
                    ds_raw = ds_raw.drop_dims(['lev', 'ilev'], errors='ignore')

                    # Extract variable and add member
                    x = ds_raw[gauss_var]
                    x['member'] = member

                    # --- Time Fixing Logic
                    try:
                        # Try standard decoding
                        _ = x.time[0].dt
                    except:
                        # Fallback to filename parsing
                        t_str = path.name.split(".")[-2] 
                        t1, t2 = t_str.split("-")[-2], t_str.split("-")[-1]
                        days = xr.date_range(
                            start=t1+"01", end=t2+"31", freq="D", calendar="noleap", use_cftime=True
                        )
                        x = x.isel(time=slice(len(days)))
                        x['time'] = days
                    
                    # --- TREFHT Specific Cleanup ---
                    if gauss_var == "TREFHT":
                        # Remove time steps where the global mean is 0 (missing data)
                        x = x.where(x.mean(dim=('lat', 'lon')) != 0, drop=True)

                    # --- Execute Tasks ---
                    # Instead of appending the HUGE daily 'x' to a list, we 
                    # run the math NOW and append the tiny monthly result.
                    for suffix, func in tasks.items():
                        chunk_start_time = time.time()
                        monthly_chunk = func(x)
                        monthly_chunk.name = gauss_var + suffix
                        # Reduce dtype to float32 if it is a float64 DataArray
                        if monthly_chunk.dtype == "float64":
                            print(f"Casting {path.name} to float32")
                            monthly_chunk = monthly_chunk.astype("float32")
                        accumulators[suffix].append(monthly_chunk)
                        chunk_end_time = time.time()
                        # print(f"\n    Task {suffix} on {path.name} took {(chunk_end_time - chunk_start_time):.2f}s\n")
                
                except Exception as e:
                    print(f"\nError processing {path}: {e}\n")
                    continue
                file_end_time = time.time()
                # print(f"\n  Processing file {path.name} took {(file_end_time - file_start_time):.2f}s\n")


            # 5. Concatenate and Save
            # Now we have lists of small monthly chunks. We concat and save once.
            concat_save_start_time = time.time()
            # Determine the full time range for the filename from the first/last processed chunks
            # (We assume all metrics cover the same timeframe, so we just check one)
            reference_key = next(iter(accumulators)) # e.g., "MX" or "_above_40"
            
            # Quick concat to get time bounds
            ref_ds = xr.concat(accumulators[reference_key], dim='time').sortby('time')
            if "historical" in stem:
                ref_ds = ref_ds.sel(time=slice(None, '2014'))
                
            final_time1 = ref_ds.time[0].dt.strftime("%Y%m").values
            final_time2 = ref_ds.time[-1].dt.strftime("%Y%m").values

            for suffix, chunks in accumulators.items():
                suffix_save_start_time = time.time()
                # New variable name
                new_var_name = gauss_var + suffix
                output_dir = input_dir / new_var_name
                output_dir.mkdir(parents=True, exist_ok=True)
                out_path = output_dir / f"{stem}.{member}.cam.h0.{new_var_name}.{final_time1}_{final_time2}.nc"

                if out_path.exists() and not ignore_existing:
                    print(f"\nPath {out_path} exists, skipping")
                    continue

                print(f"\nConcatenating and saving {new_var_name}...")
                
                # Concatenate the list of monthly chunks
                ds_final = xr.concat(chunks, dim='time')
                ds_final = ds_final.sortby('time')
                
                # Apply historical cut if needed
                if "historical" in stem:
                    ds_final = ds_final.sel(time=slice(None, '2014'))
                    
                ds_final = ds_final.rename(new_var_name)
                
                ds_final.to_netcdf(out_path)
                suffix_save_end_time = time.time()
                print(f"\nSaved to {out_path} (took {(suffix_save_end_time-suffix_save_start_time):.2f}s)")

            concat_save_end_time = time.time()
            print(f"\n>>> Concatenate and save step for member {member} took {(concat_save_end_time - concat_save_start_time):.2f}s")
            member_end_time = time.time()
            print(f"\n=== Total for member {member}: {(member_end_time-member_start_time):.2f}s ===\n")

        scenario_end_time = time.time()
        print(f"\n##### Scenario {key} finished in {(scenario_end_time - scenario_start_time):.2f}s #####")

    overall_end_time = time.time()
    print(f"\n" + "="*50)
    print(f"Total script run time: {(overall_end_time - overall_start_time):.2f} seconds")
    print(f"="*50 + "\n")

if __name__ == "__main__":
    fire.Fire(day_to_month)
