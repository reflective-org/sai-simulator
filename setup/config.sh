# Global config: env-var defaults, data-directory globals, and the
# user-choice-collection flow that populates the rest of the setup.

# Data directory globals (set by configure_data_directories*).
RAW_DATA_DIR=""
PROCESSED_DATA_DIR=""
MODELS_DIR=""
CACHE_DIR=""

# User-choice globals (set by collect_all_choices).
USER_ENV_TYPE=""           # "conda" or "venv"
USER_INSTALL_ESMF=""       # "yes" or "no"
USER_ESMF_VERSION=""       # e.g., "release/8.9.0"
USER_DOWNLOAD_DATA=""      # "yes" or "no"
USER_PROCESS_DATA=""       # "yes" or "no"
USER_FIT_MODELS=""         # "yes" or "no"
USER_CALC_VARIABILITY=""   # "yes" or "no"
USER_CACHE_DATA=""         # "yes" or "no"
USER_OVERWRITE_LENS=""     # "yes" or "no"
USER_OVERWRITE_DAILY=""    # "yes" or "no"
USER_OVERWRITE_MONTHLY=""  # "yes" or "no"

# Optional automation hook for API cache pregeneration on EC2.
# Enable by exporting RUN_API_PREGENERATION_DOCKER=yes before running this script.
RUN_API_PREGENERATION_DOCKER="${RUN_API_PREGENERATION_DOCKER:-no}"
PREGENERATION_DOCKER_IMAGE_URI="${PREGENERATION_DOCKER_IMAGE_URI:-}"
PREGENERATION_ENV_FILE="${PREGENERATION_ENV_FILE:-}"
PREGENERATION_CELERY_WORKERS="${PREGENERATION_CELERY_WORKERS:-2}"

# Run fully unattended by setting NON_INTERACTIVE=yes.
# Optional one-flag preset:
#   AUTO_PROFILE=ec2
#     - enables NON_INTERACTIVE
#     - uses EC2-safe defaults (skips heavy local data/model build steps)
#     - enables Docker pregeneration step if image URI is provided
# Optional overrides in this mode:
#   NON_INTERACTIVE_DEFAULT_ANSWER=yes|no (fallback for unexpected prompts)
#   AUTO_ENV_TYPE=conda|venv
#   AUTO_INSTALL_ESMF=yes|no
#   AUTO_ESMF_VERSION=release/8.9.0
#   AUTO_DOWNLOAD_DATA=yes|no
#   AUTO_PROCESS_DATA=yes|no
#   AUTO_FIT_MODELS=yes|no
#   AUTO_CALC_VARIABILITY=yes|no
#   AUTO_CACHE_DATA=yes|no
#   AUTO_OVERWRITE_LENS=yes|no
#   AUTO_OVERWRITE_DAILY=yes|no
#   AUTO_OVERWRITE_MONTHLY=yes|no
#   AUTO_RAW_DATA_DIR=...
#   AUTO_PROCESSED_DATA_DIR=...
#   AUTO_MODELS_DIR=...
#   AUTO_CACHE_DIR=...
NON_INTERACTIVE="${NON_INTERACTIVE:-no}"
AUTO_PROFILE="${AUTO_PROFILE:-}"
# Skip conda/venv creation in Step A, but still run Step A installs against the
# current Python environment (requirements + optional ESMF).
SKIP_ENV_SETUP="${SKIP_ENV_SETUP:-no}"
# Skip requirements installation when reusing existing env (e.g. runtime container step
# where dependencies were already installed at image build time).
SKIP_REQUIREMENTS_INSTALL="${SKIP_REQUIREMENTS_INSTALL:-no}"

apply_auto_profile() {
    case "$AUTO_PROFILE" in
        ec2|EC2)
            print_info "Applying AUTO_PROFILE=ec2 preset..."
            NON_INTERACTIVE="yes"
            NON_INTERACTIVE_DEFAULT_ANSWER="${NON_INTERACTIVE_DEFAULT_ANSWER:-yes}"

            # Conservative EC2 preset: skip heavyweight simulator build steps
            # unless explicitly overridden by AUTO_* env vars.
            AUTO_ENV_TYPE="${AUTO_ENV_TYPE:-conda}"
            AUTO_INSTALL_ESMF="${AUTO_INSTALL_ESMF:-no}"
            AUTO_DOWNLOAD_DATA="${AUTO_DOWNLOAD_DATA:-no}"
            AUTO_PROCESS_DATA="${AUTO_PROCESS_DATA:-no}"
            AUTO_FIT_MODELS="${AUTO_FIT_MODELS:-no}"
            AUTO_CALC_VARIABILITY="${AUTO_CALC_VARIABILITY:-no}"
            AUTO_CACHE_DATA="${AUTO_CACHE_DATA:-no}"

            # Enable API pregeneration hook only when image is provided.
            if [ -n "${PREGENERATION_DOCKER_IMAGE_URI:-}" ]; then
                RUN_API_PREGENERATION_DOCKER="yes"
            else
                RUN_API_PREGENERATION_DOCKER="${RUN_API_PREGENERATION_DOCKER:-no}"
                print_warning "AUTO_PROFILE=ec2 set but PREGENERATION_DOCKER_IMAGE_URI is empty."
                print_warning "Docker pregeneration hook will be skipped unless image URI is provided."
            fi
            ;;
        "")
            # No profile selected.
            ;;
        *)
            print_warning "Unknown AUTO_PROFILE='$AUTO_PROFILE'. Ignoring preset."
            ;;
    esac
}

configure_data_directories() {
    print_header "Data Directory Configuration"

    print_info "You need approximately 140GB of disk space for data storage."
    echo ""

    read -p "Enter the directory for raw data [default: data/raw]: " user_raw_dir
    RAW_DATA_DIR="${user_raw_dir:-data/raw}"
    print_success "Raw data will be stored in: $RAW_DATA_DIR"

    read -p "Enter the directory for processed data [default: data/processed]: " user_processed_dir
    PROCESSED_DATA_DIR="${user_processed_dir:-data/processed}"
    print_success "Processed data will be stored in: $PROCESSED_DATA_DIR"

    read -p "Enter the directory for models [default: data/models]: " user_models_dir
    MODELS_DIR="${user_models_dir:-data/models}"
    print_success "Models will be stored in: $MODELS_DIR"

    read -p "Enter the directory for cache [default: data/cache]: " user_cache_dir
    CACHE_DIR="${user_cache_dir:-data/cache}"
    print_success "Cache will be stored in: $CACHE_DIR"

    echo ""
    print_info "Summary of data directories:"
    echo "  Raw data:       $RAW_DATA_DIR"
    echo "  Processed data: $PROCESSED_DATA_DIR"
    echo "  Models:         $MODELS_DIR"
    echo "  Cache:          $CACHE_DIR"
    echo ""
}

configure_data_directories_non_interactive() {
    print_header "Data Directory Configuration (Non-Interactive)"

    RAW_DATA_DIR="${AUTO_RAW_DATA_DIR:-data/raw}"
    PROCESSED_DATA_DIR="${AUTO_PROCESSED_DATA_DIR:-data/processed}"
    MODELS_DIR="${AUTO_MODELS_DIR:-data/models}"
    CACHE_DIR="${AUTO_CACHE_DIR:-data/cache}"

    print_success "Raw data will be stored in: $RAW_DATA_DIR"
    print_success "Processed data will be stored in: $PROCESSED_DATA_DIR"
    print_success "Models will be stored in: $MODELS_DIR"
    print_success "Cache will be stored in: $CACHE_DIR"
    echo ""
}

# Emit the final summary of resolved user choices and directories.
_print_config_summary() {
    print_header "Configuration Summary"
    echo "  Environment type:      $USER_ENV_TYPE"
    echo "  Install ESMF:          $USER_INSTALL_ESMF"
    if [ "$USER_INSTALL_ESMF" = "yes" ]; then
        echo "    ESMF version:        $USER_ESMF_VERSION"
    fi
    echo "  Download data:         $USER_DOWNLOAD_DATA"
    echo "  Process data:          $USER_PROCESS_DATA"
    echo "  Fit models:            $USER_FIT_MODELS"
    echo "  Calculate variability: $USER_CALC_VARIABILITY"
    echo "  Cache data:            $USER_CACHE_DATA"
    echo ""
    echo "  Data directories:"
    echo "    Raw data:            $RAW_DATA_DIR"
    echo "    Processed data:      $PROCESSED_DATA_DIR"
    echo "    Models:              $MODELS_DIR"
    echo "    Cache:               $CACHE_DIR"
    echo ""
}

# Interactive questionnaire that populates every USER_* / *_DATA_DIR global.
# In non-interactive mode, sources defaults from AUTO_* env vars.
collect_all_choices() {
    print_header "Setup Configuration"

    if is_non_interactive; then
        print_info "NON_INTERACTIVE mode detected. Applying automatic configuration."
        echo ""

        USER_ENV_TYPE="${AUTO_ENV_TYPE:-conda}"
        USER_INSTALL_ESMF="${AUTO_INSTALL_ESMF:-yes}"
        USER_ESMF_VERSION="${AUTO_ESMF_VERSION:-release/8.9.0}"
        USER_DOWNLOAD_DATA="${AUTO_DOWNLOAD_DATA:-yes}"
        USER_OVERWRITE_LENS="${AUTO_OVERWRITE_LENS:-no}"
        USER_PROCESS_DATA="${AUTO_PROCESS_DATA:-yes}"
        USER_OVERWRITE_DAILY="${AUTO_OVERWRITE_DAILY:-no}"
        USER_OVERWRITE_MONTHLY="${AUTO_OVERWRITE_MONTHLY:-no}"
        USER_FIT_MODELS="${AUTO_FIT_MODELS:-yes}"
        USER_CALC_VARIABILITY="${AUTO_CALC_VARIABILITY:-yes}"
        USER_CACHE_DATA="${AUTO_CACHE_DATA:-yes}"

        configure_data_directories_non_interactive

        _print_config_summary
        print_success "Auto-configuration accepted (NON_INTERACTIVE)."
        return 0
    fi

    echo ""
    if prompt_yes_no "Is this your first time running the setup?"; then
        # First-time setup: use all defaults, skip individual questions.
        print_success "First-time setup selected. Using default configuration."
        echo ""

        USER_ENV_TYPE="conda"
        USER_INSTALL_ESMF="yes"
        USER_ESMF_VERSION="release/8.9.0"
        USER_DOWNLOAD_DATA="yes"
        USER_OVERWRITE_LENS="no"
        USER_PROCESS_DATA="yes"
        USER_OVERWRITE_DAILY="no"
        USER_OVERWRITE_MONTHLY="no"
        USER_FIT_MODELS="yes"
        USER_CALC_VARIABILITY="yes"
        USER_CACHE_DATA="yes"

        configure_data_directories
    else
        print_info "Please answer the following questions to configure the setup."
        echo ""

        # 1. Environment type
        echo "Choose your preferred environment setup:"
        echo "1) Conda"
        echo "2) venv (Standard Python virtual environment)"
        while true; do
            read -p "Enter your choice (1 or 2): " choice
            case $choice in
                1) USER_ENV_TYPE="conda"; break;;
                2) USER_ENV_TYPE="venv"; break;;
                *) print_error "Invalid choice. Please enter 1 or 2.";;
            esac
        done
        print_success "Environment type: $USER_ENV_TYPE"
        echo ""

        # 2. Data directories
        configure_data_directories

        # 3. ESMF installation
        print_info "ESMF library needs to be compiled for regridding operations."
        if prompt_yes_no "Do you want to compile and install ESMF?"; then
            USER_INSTALL_ESMF="yes"
            print_info "The default ESMF version is release/8.9.0"
            read -p "Enter ESMF version to use [default: release/8.9.0]: " esmf_ver
            USER_ESMF_VERSION="${esmf_ver:-release/8.9.0}"
            print_success "ESMF version: $USER_ESMF_VERSION"
        else
            USER_INSTALL_ESMF="no"
            print_warning "ESMF installation will be skipped."
        fi
        echo ""

        # 4. Data download
        print_info "Data download requires ~140GB of disk space."
        if prompt_yes_no "Do you want to download the data?"; then
            USER_DOWNLOAD_DATA="yes"
            if prompt_yes_no "Do you want to overwrite existing LENS NCAR files if they exist?"; then
                USER_OVERWRITE_LENS="yes"
            else
                USER_OVERWRITE_LENS="no"
            fi
        else
            USER_DOWNLOAD_DATA="no"
        fi
        echo ""

        # 5. Data processing
        if prompt_yes_no "Do you want to process the data (daily and monthly)?"; then
            USER_PROCESS_DATA="yes"
            if prompt_yes_no "Do you want to overwrite existing daily processed files?"; then
                USER_OVERWRITE_DAILY="yes"
            else
                USER_OVERWRITE_DAILY="no"
            fi
            if prompt_yes_no "Do you want to overwrite existing monthly processed files?"; then
                USER_OVERWRITE_MONTHLY="yes"
            else
                USER_OVERWRITE_MONTHLY="no"
            fi
        else
            USER_PROCESS_DATA="no"
        fi
        echo ""

        # 6. Fit models
        if prompt_yes_no "Do you want to fit the regression models?"; then
            USER_FIT_MODELS="yes"
        else
            USER_FIT_MODELS="no"
        fi
        echo ""

        # 7. Variability calculation
        if prompt_yes_no "Do you want to calculate variability for ice fraction?"; then
            USER_CALC_VARIABILITY="yes"
        else
            USER_CALC_VARIABILITY="no"
        fi
        echo ""

        # 8. Cache data
        if prompt_yes_no "Do you want to cache the data?"; then
            USER_CACHE_DATA="yes"
        else
            USER_CACHE_DATA="no"
        fi
        echo ""
    fi

    _print_config_summary

    if ! prompt_yes_no "Do you want to proceed with this configuration?"; then
        print_info "Setup cancelled."
        exit 0
    fi
}
