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
