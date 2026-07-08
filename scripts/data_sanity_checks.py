"""Download-time sanity checks for the simulator's raw data directory.

Verifies that the locally downloaded data tree matches an expected S3 manifest, contains the
required ensemble members for each variable, has no duplicate basenames, and that every
``*.nc`` file is at least header-readable. Designed to be invoked from ``setup.sh`` (or
manually as ``python scripts/data_sanity_checks.py --data_dir ...``) before any of the
processing/fitting scripts run, so missing or corrupted files are caught up front.
"""
import argparse
import re
import subprocess
import sys
from collections.abc import Callable
from collections import Counter
from pathlib import Path

import xarray as xr

RED = "\033[91m"
RESET = "\033[0m"


def print_cli_error(message: str) -> None:
    """Print colored error text for interactive CLI use."""
    print(f"{RED}ERROR:{RESET} {message}", file=sys.stderr)


def list_relative_files(root_dir: Path) -> list[str]:
    """Return all file paths under root as sorted, POSIX-style relative paths."""
    return sorted(
        str(path.relative_to(root_dir)).replace("\\", "/")
        for path in root_dir.rglob("*")
        if path.is_file()
    )


def list_netcdf_paths_relative(root_dir: Path) -> list[str]:
    """Return relative paths of all ``*.nc`` files under ``root_dir`` (download / sanity scope)."""
    root_dir = root_dir.resolve()
    return sorted(
        str(path.relative_to(root_dir)).replace("\\", "/")
        for path in root_dir.rglob("*.nc")
        if path.is_file()
    )


def build_lens_expected_filenames() -> list[str]:
    """Build expected CESM-LENS filenames downloaded by setup."""
    members = ("001", "002", "003")
    variables = ("TREFHTMN", "TREFHTMX")
    time_ranges = ("185001-201412",)
    frequency_prefix = "h0"

    expected = []
    for member in members:
        for variable in variables:
            for time_range in time_ranges:
                expected.append(
                    f"cesm2LE-historical-cmip6.{member}.{frequency_prefix}.{variable}.{time_range}.nc"
                )
    return sorted(expected)


def parse_aws_s3_ls_recursive_output(output: str) -> list[str]:
    """Parse `aws s3 ls --recursive` output into relative object paths."""
    files = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=3)
        if len(parts) == 4:
            files.append(parts[3].replace("\\", "/"))
    return sorted(files)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """
    Parse an ``s3://`` URI into bucket name and object key prefix.

    The prefix is normalized (no leading/trailing slashes). For
    ``s3://my-bucket/{SIMULATOR_VERSION}/`` the prefix is ``{SIMULATOR_VERSION}``.
    """
    raw = uri.strip()
    match = re.match(r"s3://([^/]+)/?(.*)", raw, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid S3 URI (expected s3://bucket/prefix): {uri!r}")
    bucket = match.group(1)
    prefix = match.group(2).strip("/")
    return bucket, prefix


def strip_s3_key_prefix_for_local_compare(key: str, object_prefix: str) -> str | None:
    """
    Strip the S3 URI prefix from a listing key so it matches local layout after ``aws s3 sync``.

    ``aws s3 sync s3://bucket/{SIMULATOR_VERSION}/ dest/`` stores objects as
    ``dest/<key without {SIMULATOR_VERSION}/>``.
    Listing returns full keys like ``{SIMULATOR_VERSION}/path/file.nc``; this returns
    ``path/file.nc``.

    Returns ``None`` for pure prefix / directory-marker objects with no file path left.
    """
    normalized = key.replace("\\", "/").lstrip("/")
    if not object_prefix:
        return normalized or None
    prefix = object_prefix.strip("/")
    if not normalized:
        return None
    if normalized == prefix:
        return None
    if normalized.startswith(prefix + "/"):
        stripped = normalized[len(prefix) + 1 :].lstrip("/")
        return stripped or None
    # Key outside listed prefix (should not happen for recursive ls under URI)
    return normalized


def list_bucket_files(bucket_uri: str) -> list[str]:
    """List all object keys under an S3 URI using AWS CLI, paths relative to sync destination root."""
    _, object_prefix = parse_s3_uri(bucket_uri)
    result = subprocess.run(
        ["aws", "s3", "ls", "--recursive", bucket_uri, "--no-sign-request"],
        check=True,
        capture_output=True,
        text=True,
    )
    keys = parse_aws_s3_ls_recursive_output(result.stdout)
    relative: list[str] = []
    for key in keys:
        stripped = strip_s3_key_prefix_for_local_compare(key, object_prefix)
        if stripped:
            relative.append(stripped)
    return sorted(relative)


def find_duplicate_basenames(relative_paths: list[str]) -> dict[str, int]:
    """
    Return basename -> count for basenames appearing more than once.

    Intended for NetCDF download trees: pass only paths you care about (e.g.
    from :func:`list_netcdf_paths_relative`) so unrelated files under the same
    root are not counted.
    """
    counts = Counter(Path(path).name for path in relative_paths)
    return {name: count for name, count in counts.items() if count > 1}


def validate_expected_files(expected_paths: list[str], actual_paths: list[str]) -> list[str]:
    """Return expected file paths that are missing from the local download."""
    actual_set = set(actual_paths)
    return sorted(path for path in expected_paths if path not in actual_set)


def parse_required_variables(raw: str | None) -> list[str]:
    """Parse a comma-separated variable list into normalized entries."""
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def extract_member_and_variable(filename: str) -> tuple[str, str] | None:
    """Extract ensemble member and variable token from a simulator filename."""
    member_match = re.search(r"\.(\d{3})\.", filename)
    variable_match = re.search(r"\.h[01]\.([A-Za-z0-9_]+)\.", filename)
    if not member_match or not variable_match:
        return None
    return member_match.group(1), variable_match.group(1)


def build_variable_to_members_map(relative_paths: list[str]) -> dict[str, set[str]]:
    """Build variable -> set(member_id) from downloaded filenames."""
    var_to_members: dict[str, set[str]] = {}
    for path in relative_paths:
        parsed = extract_member_and_variable(Path(path).name)
        if not parsed:
            continue
        member, variable = parsed
        var_to_members.setdefault(variable, set()).add(member)
    return var_to_members


def validate_required_variables_and_members(
    relative_paths: list[str], required_variables: list[str], min_ensemble_members: int
) -> None:
    """Validate required variables exist and each has minimum ensemble coverage."""
    if not required_variables:
        return

    var_to_members = build_variable_to_members_map(relative_paths)
    missing_variables = sorted(variable for variable in required_variables if variable not in var_to_members)
    if missing_variables:
        raise ValueError(
            "Missing required variables in downloaded data: " + ", ".join(missing_variables)
        )

    insufficient = sorted(
        (
            variable,
            len(var_to_members[variable]),
            sorted(var_to_members[variable]),
        )
        for variable in required_variables
        if len(var_to_members[variable]) < min_ensemble_members
    )
    if insufficient:
        details = "; ".join(
            f"{variable} has {count} member(s): {', '.join(members)}"
            for variable, count, members in insufficient
        )
        raise ValueError(
            f"Variables with fewer than {min_ensemble_members} ensemble members: {details}"
        )


def validate_netcdf_file_integrity(file_path: Path) -> None:
    """
    Validate that a NetCDF file is readable and not obviously corrupted.

    This intentionally performs a lightweight integrity check: open the dataset
    and access metadata (dims/data_vars) without loading full payload arrays.
    It catches missing/unreadable/truncated headers and many obvious failures,
    but may not detect payload-level corruption when headers remain intact.
    That tradeoff keeps download-time sanity checks fast for large datasets.
    """
    with xr.open_dataset(file_path, decode_times=False) as ds:
        _ = tuple(ds.dims.items())
        _ = tuple(ds.data_vars.keys())


def find_corrupted_netcdf_files(
    data_dir: Path,
    validator: Callable[[Path], None] = validate_netcdf_file_integrity,
) -> list[str]:
    """Return relative paths of unreadable/corrupted NetCDF files."""
    corrupted: list[str] = []
    for file_path in sorted(data_dir.rglob("*.nc")):
        try:
            validator(file_path)
        except Exception:
            corrupted.append(str(file_path.relative_to(data_dir)).replace("\\", "/"))
    return corrupted


def run_download_sanity_checks(
    data_dir: Path,
    expected_bucket_uri: str | None = None,
    include_lens_files: bool = False,
    required_variables: list[str] | None = None,
    min_ensemble_members: int = 3,
) -> None:
    """Run all download-time sanity checks and raise on any violations."""
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    actual_files = list_relative_files(data_dir)
    netcdf_paths = list_netcdf_paths_relative(data_dir)
    duplicates = find_duplicate_basenames(netcdf_paths)
    if duplicates:
        duplicate_summary = ", ".join(
            f"{name} ({count} copies)" for name, count in sorted(duplicates.items())
        )
        raise ValueError(
            f"Duplicate NetCDF basenames under {data_dir}: {duplicate_summary}"
        )

    expected_files: list[str] = []

    if expected_bucket_uri:
        expected_files.extend(list_bucket_files(expected_bucket_uri))

    if include_lens_files:
        expected_files.extend(build_lens_expected_filenames())

    if expected_files:
        missing_files = validate_expected_files(sorted(set(expected_files)), actual_files)
        if missing_files:
            sample = ", ".join(missing_files[:10])
            more = "" if len(missing_files) <= 10 else f" ... (+{len(missing_files) - 10} more)"
            raise ValueError(
                f"Missing {len(missing_files)} expected files in {data_dir}: {sample}{more}"
            )

    validate_required_variables_and_members(
        netcdf_paths,
        required_variables=required_variables or [],
        min_ensemble_members=min_ensemble_members,
    )

    corrupted_files = find_corrupted_netcdf_files(data_dir)
    if corrupted_files:
        sample = ", ".join(corrupted_files[:10])
        more = "" if len(corrupted_files) <= 10 else f" ... (+{len(corrupted_files) - 10} more)"
        raise ValueError(
            f"Detected {len(corrupted_files)} corrupted/unreadable NetCDF files in {data_dir}: "
            f"{sample}{more}"
        )


def main() -> None:
    """CLI entry point: parse args, run :func:`run_download_sanity_checks`, exit non-zero on failure.

    On a sanity-check failure (``ValueError`` or ``FileNotFoundError``) the error is rendered
    in red via :func:`print_cli_error` and the process exits with status ``2`` so callers
    (e.g. ``setup.sh``) can distinguish "checks failed" from generic Python errors.
    """
    parser = argparse.ArgumentParser(description="Sanity checks for downloaded raw data files.")
    parser.add_argument("--data_dir", required=True, help="Local raw data directory.")
    parser.add_argument(
        "--expected_bucket_uri",
        default=None,
        help="Optional S3 URI to verify expected files against.",
    )
    parser.add_argument(
        "--include_lens_files",
        action="store_true",
        help="Include expected CESM LENS files in expected set.",
    )
    parser.add_argument(
        "--required_variables",
        default=None,
        help="Comma-separated list of required variable names (e.g. TREFHT,PRECT,ICEFRAC).",
    )
    parser.add_argument(
        "--min_ensemble_members",
        type=int,
        default=3,
        help="Minimum number of ensemble members required per required variable.",
    )
    args = parser.parse_args()

    try:
        run_download_sanity_checks(
            data_dir=Path(args.data_dir),
            expected_bucket_uri=args.expected_bucket_uri,
            include_lens_files=args.include_lens_files,
            required_variables=parse_required_variables(args.required_variables),
            min_ensemble_members=args.min_ensemble_members,
        )
        print("Download sanity checks passed.")
    except (ValueError, FileNotFoundError) as exc:
        print_cli_error(str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
