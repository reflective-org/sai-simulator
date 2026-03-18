#!/bin/bash

# ==============================================================================
# CONFIGURATION & SETUP
# ==============================================================================

# We remove 'set -e' so the script continues even if Python fails.
# We keep 'set -u' (error on undefined vars) and 'pipefail' (error on pipe failures)
set -uo pipefail

# Define Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Define the list of variables to process
VARIABLES=(
    "tas"
    "pr"
    "tasmax"
    "tasmin"
    "tas_above_35"
    "tas_above_40"
    "tas_below_0"
    "pr_above_10"
    "pr_above_20"
    "p-e"
    "icefrac"
)

SCRIPT_PATH="scripts/process_monthly_gauss.py"

# Initialize arrays to track status
SUCCESSFUL_VARS=()
FAILED_VARS=()

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

log_info() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

usage() {
    echo -e "Usage: $0 <data_dir> <output_dir> [--overwrite]"
    exit 1
}

# ==============================================================================
# ARGUMENT PARSING & VALIDATION
# ==============================================================================

if [ "$#" -lt 2 ]; then
    usage
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
OVERWRITE_FLAG=""

if [[ "${3:-}" == "--overwrite" ]]; then
    OVERWRITE_FLAG="--overwrite"
    log_warn "Overwrite mode ENABLED. Existing files will be replaced."
elif [[ -n "${3:-}" ]]; then
    log_error "Unknown argument: $3"
    usage
fi

if [ ! -d "$DATA_DIR" ]; then
    log_error "Input directory does not exist: $DATA_DIR"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    log_error "Python script not found at: $SCRIPT_PATH"
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    log_info "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# ==============================================================================
# MAIN EXECUTION LOOP
# ==============================================================================

log_info "Starting batch processing..."

for var in "${VARIABLES[@]}"; do
    echo "-----------------------------------------------------------------"
    log_info "Processing variable: ${GREEN}$var${NC}"
    
    # Run Python script. The 'if' statement captures the exit code.
    if python "$SCRIPT_PATH" \
        --var "$var" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        $OVERWRITE_FLAG; then
        
        # If exit code is 0 (Success)
        SUCCESSFUL_VARS+=("$var")
    else
        # If exit code is non-zero (Failure)
        log_error "Failed to process $var. continuing to next..."
        FAILED_VARS+=("$var")
    fi
done

# ==============================================================================
# SUMMARY REPORT
# ==============================================================================

echo -e "================================================================="
echo -e "                        PROCESSING SUMMARY                       "
echo -e "=================================================================\n"

# 1. Print Successful Variables
if [ ${#SUCCESSFUL_VARS[@]} -gt 0 ]; then
    echo -e "${GREEN}Process Succeeded for (${#SUCCESSFUL_VARS[@]}):${NC}"
    for var in "${SUCCESSFUL_VARS[@]}"; do
        echo -e "  ✅ $var"
    done
else
    echo -e "${YELLOW}No variables were processed successfully.${NC}"
fi

echo "" # Empty line

# 2. Print Failed Variables
if [ ${#FAILED_VARS[@]} -gt 0 ]; then
    echo -e "${RED}Process Failed for (${#FAILED_VARS[@]}):${NC}"
    for var in "${FAILED_VARS[@]}"; do
        echo -e "  ❌ $var"
    done
    
    # Exit with error code 1 so external tools know the job had failures
    echo -e "\n${RED}Batch processing completed with errors.${NC}"
    exit 1
else
    echo -e "${GREEN}All variables processed successfully!${NC}"
    exit 0
fi