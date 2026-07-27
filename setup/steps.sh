# Steps B–G: data download, processing, model fitting, variability,
# caching, and the optional Docker-based API pregeneration step.

# B. Data Download
download_data() {
    print_header "Step B: Data Download"

    print_info "This will download ~140GB of data and takes about 30 minutes."

    print_info "Installing AWS CLI..."
    pip install awscli
    print_success "AWS CLI installed."

    print_info "Creating data directory..."
    mkdir -p "$RAW_DATA_DIR"
    print_success "Directory created: $RAW_DATA_DIR"

    print_info "Downloading data from S3 bucket (this may take a while)..."
    local start_time end_time duration minutes seconds
    start_time=$(date +%s)
    aws s3 sync "$DATA_BUCKET_URI" "$RAW_DATA_DIR/" --no-sign-request
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    print_success "Data downloaded from S3."
    print_info "Download completed in ${minutes} minutes and ${seconds} seconds."

    local OVERWRITE_ARGS=()
    if [ "$USER_OVERWRITE_LENS" = "yes" ]; then
        OVERWRITE_ARGS+=("--overwrite")
        print_info "Will overwrite existing LENS files"
    else
        print_info "Will skip existing LENS files"
    fi

    print_info "Downloading LENS NCAR data for TREFHTMX..."
    if [ -f "scripts/run_download_CESM_LENS.sh" ]; then
        bash scripts/run_download_CESM_LENS.sh TREFHTMX monthly "$RAW_DATA_DIR" 001 002 003 "${OVERWRITE_ARGS[@]}"
        print_success "TREFHTMX data downloaded."
    else
        print_error "scripts/run_download_CESM_LENS.sh not found!"
    fi

    print_info "Downloading LENS NCAR data for TREFHTMN..."
    if [ -f "scripts/run_download_CESM_LENS.sh" ]; then
        bash scripts/run_download_CESM_LENS.sh TREFHTMN monthly "$RAW_DATA_DIR" 001 002 003 "${OVERWRITE_ARGS[@]}"
        print_success "TREFHTMN data downloaded."
    else
        print_error "scripts/run_download_CESM_LENS.sh not found!"
    fi

    print_info "Running sanity checks on downloaded files..."
    if [ -f "scripts/data_sanity_checks.py" ]; then
        # --exclude_dirs skips subdirs that downstream processing writes under $RAW_DATA_DIR.
        # The PRECT collision is the load-bearing one: process_monthly_gauss.py's PRECC+PRECL
        # pairing step (scripts/process_monthly_gauss.py:718) writes derived PRECT files with
        # the same basenames as the S3-native PRECT files at the top level. TREFHT/PRECT bare
        # dirs and TREFHT_*/PRECT_* threshold-count subdirs are process_daily_gauss.py outputs
        # (scripts/process_daily_gauss.py). Without this exclusion, a rerun of setup.sh after
        # any processing has run trips "Duplicate NetCDF basenames".
        python scripts/data_sanity_checks.py \
            --data_dir "$RAW_DATA_DIR" \
            --expected_bucket_uri "$DATA_BUCKET_URI" \
            --include_lens_files \
            --required_variables "TREFHT,PRECT,PRECC,PRECL,QFLX,ICEFRAC,TREFHTMN,TREFHTMX" \
            --min_ensemble_members 3 \
            --exclude_dirs "TREFHT,PRECT,TREFHT_above_40,TREFHT_above_35,TREFHT_below_0,PRECT_above_10,PRECT_above_20"
        print_success "Download sanity checks passed."
    else
        print_error "scripts/data_sanity_checks.py not found!"
        exit 1
    fi

    print_success "Data download complete!"
}

# C. Data Processing
process_data() {
    print_header "Step C: Data Processing"

    local DAILY_OVERWRITE_FLAG=""
    if [ "$USER_OVERWRITE_DAILY" = "yes" ]; then
        DAILY_OVERWRITE_FLAG="--ignore-existing"
        print_info "Will overwrite existing daily processed files"
    else
        print_info "Will skip existing daily processed files"
    fi

    print_info "Processing daily data to create monthly values..."
    if [ -f "scripts/process_all_daily.sh" ]; then
        bash scripts/process_all_daily.sh "$RAW_DATA_DIR" $DAILY_OVERWRITE_FLAG
        print_success "Daily data processed."
    else
        print_error "scripts/process_all_daily.sh not found!"
    fi

    local MONTHLY_OVERWRITE_FLAG=""
    if [ "$USER_OVERWRITE_MONTHLY" = "yes" ]; then
        MONTHLY_OVERWRITE_FLAG="--overwrite"
        print_info "Will overwrite existing monthly processed files"
    else
        print_info "Will skip existing monthly processed files"
    fi

    print_info "Processing monthly data to create annual values..."
    mkdir -p "$PROCESSED_DATA_DIR"
    if [ -f "scripts/process_all_monthly.sh" ]; then
        bash scripts/process_all_monthly.sh "$RAW_DATA_DIR" "$PROCESSED_DATA_DIR" $MONTHLY_OVERWRITE_FLAG
        print_success "Monthly data processed."
    else
        print_error "scripts/process_all_monthly.sh not found!"
    fi

    print_success "Data processing complete!"
}

# D. Fit Regression Models
fit_models() {
    print_header "Step D: Fit Regression Models"

    print_info "Fitting regression models for all variables..."
    mkdir -p "$MODELS_DIR"

    if [ -f "scripts/fit_all.sh" ]; then
        bash scripts/fit_all.sh "$PROCESSED_DATA_DIR" "$MODELS_DIR"
        print_success "Regression models fitted."
    else
        print_error "scripts/fit_all.sh not found!"
    fi

    print_success "Model fitting complete!"
}

# E. Variability Calculation
calculate_variability() {
    print_header "Step E: Variability Calculation"

    print_info "Calculating model internal variability for sea ice..."

    if [ -f "scripts/variability.py" ]; then
        python scripts/variability.py icefrac "$PROCESSED_DATA_DIR" "$MODELS_DIR"
        print_success "Variability calculated."
    else
        print_error "scripts/variability.py not found!"
    fi

    print_success "Variability calculation complete!"
}

# F. Cache the Data
cache_data() {
    print_header "Step F: Cache the Data"

    print_info "Caching data for efficient loading by the frontend..."
    mkdir -p "$CACHE_DIR"

    if [ -f "scripts/cache.py" ]; then
        python scripts/cache.py --data_dir "$PROCESSED_DATA_DIR" --model_dir "$MODELS_DIR" --output_dir "$CACHE_DIR"
        print_success "Data cached."
    else
        print_error "scripts/cache.py not found!"
    fi

    print_success "Data caching complete!"
}

# G. Optional API cache pregeneration via a prebuilt Docker image. Skipped
# unless RUN_API_PREGENERATION_DOCKER=yes (typically via AUTO_PROFILE=ec2).
run_api_pregeneration_in_docker() {
    if [ "$RUN_API_PREGENERATION_DOCKER" != "yes" ]; then
        return 0
    fi

    print_header "Step G: API Cache Pregeneration (Docker)"

    if [ -z "$PREGENERATION_DOCKER_IMAGE_URI" ]; then
        print_error "RUN_API_PREGENERATION_DOCKER=yes but PREGENERATION_DOCKER_IMAGE_URI is not set."
        print_info "Set PREGENERATION_DOCKER_IMAGE_URI to your ECR image URI and re-run."
        return 1
    fi

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed."
        print_info "Install Docker and re-run setup, or disable RUN_API_PREGENERATION_DOCKER."
        return 1
    fi

    print_info "Pulling pregeneration image: $PREGENERATION_DOCKER_IMAGE_URI"
    if ! docker pull "$PREGENERATION_DOCKER_IMAGE_URI"; then
        print_error "Failed to pull pregeneration image."
        return 1
    fi

    local docker_args=(--rm --name sai-api-pregeneration)
    docker_args+=(-e "CELERY_WORKERS=$PREGENERATION_CELERY_WORKERS")

    if [ -n "$PREGENERATION_ENV_FILE" ] && [ -f "$PREGENERATION_ENV_FILE" ]; then
        print_info "Using env file: $PREGENERATION_ENV_FILE"
        docker_args+=(--env-file "$PREGENERATION_ENV_FILE")
    fi

    print_info "Running API cache pregeneration container..."
    if ! docker run "${docker_args[@]}" "$PREGENERATION_DOCKER_IMAGE_URI"; then
        print_error "Pregeneration container run failed."
        return 1
    fi
    print_success "API cache pregeneration run completed."
    return 0
}
