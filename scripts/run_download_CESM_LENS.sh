#!/bin/bash

# Usage: ./run_download_CESM_LENS.sh TREFHTMN daily /output/dir 001 002 003 [--overwrite]

set -e

print_usage() {
    echo "Usage: $0 VARIABLE FREQUENCY OUTPUT_DIR MEMBER1 [MEMBER2 ...] [--overwrite]"
    echo "  VARIABLE     Variable name (e.g., TREFHTMN, TREFHTMAX)"
    echo "  FREQUENCY    daily|monthly"
    echo "  OUTPUT_DIR   Output directory to save NetCDF files (absolute or relative path)"
    echo "  MEMBER       At least one member number (e.g., 001)"
    echo "  --overwrite  (optional) Overwrite existing files"
    echo ""
    echo "Example: $0 TREFHTMN daily /path/to/my/raw_dir 001 002 003 --overwrite"
}

# At least 4 arguments needed: var, freq, output_dir, at least 1 member
if [ $# -lt 4 ]; then
    print_usage
    exit 1
fi

VARIABLE="$1"
FREQUENCY="$2"
OUTPUT_DIR="$3"
shift 3

if [[ "$FREQUENCY" != "daily" && "$FREQUENCY" != "monthly" ]]; then
    echo "Error: Frequency must be 'daily' or 'monthly'"
    exit 1
fi

OVERWRITE_FLAG=""
MEMBERS=()
for arg in "$@"; do
    if [[ "$arg" == "--overwrite" ]]; then
        OVERWRITE_FLAG="--overwrite"
    else
        MEMBERS+=("$arg")
    fi
done

if [ "${#MEMBERS[@]}" -eq 0 ]; then
    echo "Error: At least one member number is required (e.g., 001)"
    exit 1
fi

echo "Running extraction for input variable: $VARIABLE"
echo "Frequency: $FREQUENCY"
echo "Members to download: ${MEMBERS[*]}"
echo "Output directory: $OUTPUT_DIR"
if [[ -n "$OVERWRITE_FLAG" ]]; then
    echo "Overwrite is enabled."
fi

MEMBERS_ARG=$(IFS=, ; echo "${MEMBERS[*]}")

CMD=(python scripts/download_CESM_LENS.py --var "$VARIABLE" --frequency "$FREQUENCY" --output_dir "$OUTPUT_DIR" --members "$MEMBERS_ARG")
if [[ -n "$OVERWRITE_FLAG" ]]; then
    CMD+=("$OVERWRITE_FLAG")
fi

"${CMD[@]}"
