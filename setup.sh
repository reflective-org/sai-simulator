#!/usr/bin/env bash

# SAI Simulator Setup Script
# Compatible with bash and zsh
# This script automates the installation and setup process for the SAI Simulator.
#
# Modular layout — see ./setup/ for the implementation:
#   lib.sh    UI helpers, predicates, python-version + shell-config helpers
#   config.sh global state, env-var defaults, AUTO_PROFILE, collect_all_choices
#   esmf.sh   ESMF build + ESMPy install
#   env.sh    Step A: conda / venv / existing-Python setup
#   steps.sh  Steps B–G: download, process, fit, variability, cache, pregeneration

set -e

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup"

source "$SETUP_DIR/lib.sh"
source "$SETUP_DIR/config.sh"
source "$SETUP_DIR/esmf.sh"
source "$SETUP_DIR/env.sh"
source "$SETUP_DIR/steps.sh"

main() {
    print_header "SAI Simulator Setup Script"

    # Apply optional profile presets before any prompting/decision logic.
    apply_auto_profile

    # Note: Python version check is done per-environment type
    # - For venv: system Python 3.11+ is required
    # - For conda: Python version is managed by conda

    print_info "This script will guide you through the complete setup process."
    print_info "Required disk space: ~140GB"
    echo ""

    if ! is_non_interactive && ! prompt_yes_no "Do you want to continue?"; then
        print_info "Setup cancelled."
        exit 0
    fi

    collect_all_choices

    # Step A: Environment Setup
    print_header "Step A: Environment Setup"
    if is_truthy "$SKIP_ENV_SETUP"; then
        print_warning "SKIP_ENV_SETUP is enabled. Reusing current Python environment."
        setup_existing_python_env
    elif [ "$USER_ENV_TYPE" = "conda" ]; then
        setup_conda
    else
        setup_venv
    fi

    # Steps B–F: each is a no-op when the user opted out during config.
    if [ "$USER_DOWNLOAD_DATA" = "yes" ]; then
        download_data
    else
        print_warning "Skipping data download."
    fi

    if [ "$USER_PROCESS_DATA" = "yes" ]; then
        process_data
    else
        print_warning "Skipping data processing."
    fi

    if [ "$USER_FIT_MODELS" = "yes" ]; then
        fit_models
    else
        print_warning "Skipping model fitting."
    fi

    if [ "$USER_CALC_VARIABILITY" = "yes" ]; then
        calculate_variability
    else
        print_warning "Skipping variability calculation."
    fi

    if [ "$USER_CACHE_DATA" = "yes" ]; then
        cache_data
    else
        print_warning "Skipping data caching."
    fi

    # Step G is opt-in via env vars (typically AUTO_PROFILE=ec2 + image URI).
    if ! run_api_pregeneration_in_docker; then
        print_error "Pregeneration failed. Setup incomplete."
        exit 1
    fi

    print_header "Setup Complete!"
    print_success "All requested steps have been completed successfully."
    echo ""
    print_info "Data directories used:"
    echo "  Raw data:       $RAW_DATA_DIR"
    echo "  Processed data: $PROCESSED_DATA_DIR"
    echo "  Models:         $MODELS_DIR"
    echo "  Cache:          $CACHE_DIR"
    echo ""
    print_info "To launch the simulator, run:"
    echo "  python launch_gradio.py --use_local_cache"
    echo ""
    print_info "For custom injection, run:"
    echo "  python variable_launch_gradio.py --use_local_cache"
    echo ""
    print_info "Note: Use Firefox browser (not Safari) for the web UI."
}

main
