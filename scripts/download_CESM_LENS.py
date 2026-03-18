import xarray as xr
import sys
import time
import os
import fire

def download_cesm_lens(
    var: str,
    frequency: str,
    output_dir: str,
    members: str,
    overwrite: bool = False
):
    """
    Download CESM2 LENS members from S3 and save to NetCDF.

    Args:
        var (str): Variable name (e.g., TREFHTMN, TREFHTMAX)
        frequency (str): 'daily' or 'monthly'
        output_dir (str): Output directory to save NetCDF files.
        members (list of str or comma-separated str): Member numbers to download (e.g., ['001', '002', '003'] or "001,002,003")
        overwrite (bool): Overwrite existing NetCDF files if they exist.
    """
    start_time = time.time()

    # Accept both comma-separated strings and lists for members
    if isinstance(members, str):
        # Allow both space- and comma-separated values
        if "," in members:
            member_numbers = [m.strip() for m in members.split(",")]
        else:
            member_numbers = members.split()
    else:
        member_numbers = members

    # Set frequency prefix (h1 for daily, h0 for monthly)
    freq_prefix = 'h1' if frequency == 'daily' else 'h0'

    # S3 URL
    zarr_url = f's3://ncar-cesm2-lens/atm/{frequency}/cesm2LE-historical-cmip6-{var}.zarr'
    print(f"Connecting to: {zarr_url}")

    # Open dataset
    try:
        ds = xr.open_zarr(zarr_url, consolidated=True, storage_options={'anon': True})
    except Exception as e:
        print(f"Error opening Zarr store: {e}")
        print(f"Check if the variable name is correct and exists in the 'atm/{frequency}' bucket path (s3://ncar-cesm2-lens/atm/{frequency}/).")
        sys.exit(1)

    # Prepare member IDs (e.g., 'r1i1281p1f1') from member numbers (001, 002, etc.)
    members_to_save = {}
    for member_num in member_numbers:
        member_num = member_num.strip()
        if not member_num:
            continue
        member_int = int(member_num.lstrip('0') or '0')
        member_id = f'r{member_int}i1281p1f1'
        # Always use 3-digit suffix for filename
        members_to_save[member_id] = member_num.zfill(3)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through members and save to NetCDF
    for member_id, member_suffix in members_to_save.items():
        filename = f"cesm2LE-historical-cmip6.{member_suffix}.{freq_prefix}.{var}.185001-201412.nc"
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path) and not overwrite:
            print(f"File {filename} already exists. Skipping... (use --overwrite to force)")
            continue

        # Select the member from xarray dataset
        ds_single = ds.sel(member_id=member_id)
        # Drop member_id from dataset before saving (only a scalar coordinate now)
        ds_single = ds_single.drop_vars('member_id', errors='ignore')

        print(f"Downloading and saving {filename}{' (overwriting)' if overwrite and os.path.exists(file_path) else ''}...")
        ds_single.to_netcdf(file_path)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Processing for {var} completed in {duration:.2f} seconds.\n")

if __name__ == "__main__":
    # Use the fire library for easy CLI
    fire.Fire(download_cesm_lens)
