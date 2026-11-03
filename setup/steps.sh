# Steps B–G: data download, processing, model fitting, variability,
# caching, and the optional Docker-based API pregeneration step.

# B. Data Access
# The consumer reads the per-experiment training data lazily from S3-hosted
# Zarr stores at ${ZARR_STORES_URI}. There is nothing to download up front.
# This step just verifies the stores are reachable and have the expected
# schema before the processing step starts using them.
download_data() {
    print_header "Step B: Data Access"

    print_info "Verifying access to the Zarr training-data stores..."
    if [ -f "scripts/data_sanity_checks.py" ]; then
        python scripts/data_sanity_checks.py
        print_success "Sanity checks passed."
    else
        print_error "scripts/data_sanity_checks.py not found!"
        exit 1
    fi

    print_success "Data access verified."
}

# C. Data Processing
process_data() {
    print_header "Step C: Data Processing"

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

    # scripts/variability.py is the only writer of each model's internal-variability artifacts
    # (grid_level_model_internal_variability_<model>.nc, grid_level_lag1_autocorr_<model>.nc,
    # grid_level_reference_variability_<model>.nc, the global .npy and the region columns), which
    # the significance test and the band read for EVERY variable. icefrac also needs the sea-ice
    # sigmoids fitted in Step D. Same variable list as scripts/process_all_monthly.sh.
    print_info "Calculating each model's internal variability for every variable..."

    if [ -f "scripts/variability.py" ]; then
        for var in tas pr tasmax tasmin tas_above_35 tas_above_40 tas_below_0 pr_above_10 pr_above_20 p-e icefrac; do
            print_info "  $var"
            if ! python scripts/variability.py "$var" "$PROCESSED_DATA_DIR" "$MODELS_DIR"; then
                print_error "Variability failed for $var."
                return 1
            fi
        done
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

# F2. InVERT variability caches (line-plot envelope). Built here, never inside a
# dashboard request. The InVERT ensembles are read from the public R2 Zarr store
# (backend.invert_variability.INVERT_STORE, ~15 GB read once for all four variables;
# up-to-date caches are not rebuilt). netCDFs under $PROCESSED_DATA_DIR/invert/, if
# present, are used instead. A variable whose source cannot be read is reported as
# skipped, and the dashboard falls back to the legacy band for it.
build_invert_caches() {
    print_header "Step F2: InVERT Variability Caches"

    python -m backend.invert_variability --data-dir "$PROCESSED_DATA_DIR" --cache-dir "$CACHE_DIR"
    print_success "InVERT caches step finished (see the per-variable status above)."
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
