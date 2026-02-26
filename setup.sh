#!/usr/bin/env bash

# SAI Simulator Setup Script
# Compatible with bash and zsh
# This script automates the installation and setup process for the SAI Simulator

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global variables for data directories
RAW_DATA_DIR=""
PROCESSED_DATA_DIR=""
MODELS_DIR=""
CACHE_DIR=""

# Global variables for user choices (collected up front)
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

# Helper functions
print_header() {
    echo -e "\n${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

prompt_yes_no() {
    while true; do
        read -p "$1 (y/n): " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes (y) or no (n).";;
        esac
    done
}

check_python_version() {
    # This function checks the system Python version for venv setup
    # For conda, Python version is managed by conda itself
    print_info "Checking Python version..."

    # Check if python is available
    if ! command -v python &> /dev/null; then
        print_error "Python 3 is not installed."
        print_error "Please install Python 3.11 or higher."
        print_info "Visit: https://www.python.org/downloads/"
        exit 1
    fi

    # Get Python version
    PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_MAJOR=$(python -c 'import sys; print(sys.version_info[0])')
    PYTHON_MINOR=$(python -c 'import sys; print(sys.version_info[1])')

    print_info "Detected Python version: $PYTHON_VERSION"

    # Check if version is 3.11 or higher
    if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]; }; then
        print_error "Python version $PYTHON_VERSION is not supported."
        print_error "This project requires Python 3.11 or higher."
        print_info "Please upgrade your Python installation."
        print_info "Visit: https://www.python.org/downloads/"
        exit 1
    fi

    print_success "Python version $PYTHON_VERSION is compatible!"
}

check_python_version_in_env() {
    # Check Python version in the current environment (after activation)
    local PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    local PYTHON_MAJOR=$(python -c 'import sys; print(sys.version_info[0])')
    local PYTHON_MINOR=$(python -c 'import sys; print(sys.version_info[1])')

    if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]; }; then
        return 1  # Version not compatible
    fi
    return 0  # Version compatible
}

configure_data_directories() {
    print_header "Data Directory Configuration"

    print_info "You need approximately 140GB of disk space for data storage."
    echo ""

    # Ask for raw data directory
    read -p "Enter the directory for raw data [default: data/raw]: " user_raw_dir
    RAW_DATA_DIR="${user_raw_dir:-data/raw}"
    print_success "Raw data will be stored in: $RAW_DATA_DIR"

    # Ask for processed data directory
    read -p "Enter the directory for processed data [default: data/processed]: " user_processed_dir
    PROCESSED_DATA_DIR="${user_processed_dir:-data/processed}"
    print_success "Processed data will be stored in: $PROCESSED_DATA_DIR"

    # Ask for models directory
    read -p "Enter the directory for models [default: data/models]: " user_models_dir
    MODELS_DIR="${user_models_dir:-data/models}"
    print_success "Models will be stored in: $MODELS_DIR"

    # Ask for cache directory
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

collect_all_choices() {
    print_header "Setup Configuration"

    # Ask if this is a first-time setup
    echo ""
    if prompt_yes_no "Is this your first time running the setup?"; then
        # First-time setup: use all defaults, skip individual questions
        print_success "First-time setup selected. Using default configuration."
        echo ""

        # Set all defaults for first-time setup
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

        # Still need to configure data directories
        configure_data_directories

    else
        # Not first-time: ask all questions individually
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
            # Ask for version
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
            # Ask about LENS overwrite
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
            # Ask about overwrite flags
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

    # Summary
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

    if ! prompt_yes_no "Do you want to proceed with this configuration?"; then
        print_info "Setup cancelled."
        exit 0
    fi
}

# Main setup function
main() {
    print_header "SAI Simulator Setup Script"

    # Note: Python version check is done per-environment type
    # - For venv: system Python 3.11+ is required
    # - For conda: Python version is managed by conda

    print_info "This script will guide you through the complete setup process."
    print_info "Required disk space: ~140GB"
    echo ""

    if ! prompt_yes_no "Do you want to continue?"; then
        print_info "Setup cancelled."
        exit 0
    fi

    # Collect all user choices up front
    collect_all_choices

    # Step A: Environment Setup
    print_header "Step A: Environment Setup"
    if [ "$USER_ENV_TYPE" = "conda" ]; then
        setup_conda
    else
        setup_venv
    fi

    # Step B: Data Download
    if [ "$USER_DOWNLOAD_DATA" = "yes" ]; then
        download_data
    else
        print_warning "Skipping data download."
    fi

    # Step C: Data Processing
    if [ "$USER_PROCESS_DATA" = "yes" ]; then
        process_data
    else
        print_warning "Skipping data processing."
    fi

    # Step D: Fit Regression Models
    if [ "$USER_FIT_MODELS" = "yes" ]; then
        fit_models
    else
        print_warning "Skipping model fitting."
    fi

    # Step E: Variability Calculation
    if [ "$USER_CALC_VARIABILITY" = "yes" ]; then
        calculate_variability
    else
        print_warning "Skipping variability calculation."
    fi

    # Step F: Cache the Data
    if [ "$USER_CACHE_DATA" = "yes" ]; then
        cache_data
    else
        print_warning "Skipping data caching."
    fi

    # Final summary
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

# A. Environment Setup
setup_conda() {
    print_info "Setting up Conda environment..."

    # Check if conda is installed
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed. Please install Conda first."
        print_info "Visit: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi

    # Get conda base path for activation
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"

    # Check if a conda environment is currently activated
    CURRENT_ENV="${CONDA_DEFAULT_ENV:-}"

    if [ -n "$CURRENT_ENV" ] && [ "$CURRENT_ENV" != "base" ]; then
        print_info "Currently active conda environment: $CURRENT_ENV"

        if [ "$CURRENT_ENV" = "sai-simulator" ]; then
            # Already in sai-simulator environment - check Python version
            print_info "You are already in the 'sai-simulator' environment."

            PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
            print_info "Python version in current environment: $PYTHON_VERSION"

            if check_python_version_in_env; then
                print_success "Python version is compatible!"
                echo ""
                if prompt_yes_no "Do you want to continue with the existing 'sai-simulator' environment?"; then
                    # Skip to ESMF installation
                    print_info "Continuing with existing environment."
                    offer_esmf_installation
                    print_success "Environment setup complete!"
                    print_info "To activate this environment in the future, run: conda activate sai-simulator"
                    return 0
                else
                    print_info "Will recreate the environment..."
                    conda deactivate
                    conda env remove -n sai-simulator -y
                    print_success "Existing environment removed."
                fi
            else
                print_warning "Python version $PYTHON_VERSION is not compatible (requires 3.11+)."
                echo ""
                if prompt_yes_no "Do you want to recreate the environment with Python 3.12?"; then
                    conda deactivate
                    conda env remove -n sai-simulator -y
                    print_success "Existing environment removed."
                else
                    print_error "Cannot continue with incompatible Python version."
                    exit 1
                fi
            fi
        else
            # A different conda environment is active
            print_warning "A different conda environment '$CURRENT_ENV' is currently active."
            echo ""
            echo "Options:"
            echo "  1) Deactivate '$CURRENT_ENV' and create/use 'sai-simulator'"
            echo "  2) Exit and let me manually manage environments"
            echo ""

            while true; do
                read -p "Enter your choice (1 or 2): " env_choice
                case $env_choice in
                    1)
                        print_info "Deactivating '$CURRENT_ENV'..."
                        conda deactivate
                        print_success "Environment deactivated."
                        break
                        ;;
                    2)
                        print_info "Exiting. Please manually activate the desired environment and re-run setup."
                        print_info "To use sai-simulator: conda activate sai-simulator"
                        exit 0
                        ;;
                    *)
                        print_error "Invalid choice. Please enter 1 or 2."
                        ;;
                esac
            done
        fi
    fi

    # Check if sai-simulator environment already exists
    if conda env list | grep -q "^sai-simulator "; then
        print_info "Conda environment 'sai-simulator' already exists."

        # Activate and check Python version
        conda activate sai-simulator
        PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_info "Python version in 'sai-simulator': $PYTHON_VERSION"

        if check_python_version_in_env; then
            print_success "Python version is compatible!"
            echo ""
            echo "Options:"
            echo "  1) Use existing 'sai-simulator' environment (recommended)"
            echo "  2) Delete and recreate the environment"
            echo ""

            while true; do
                read -p "Enter your choice (1 or 2): " existing_choice
                case $existing_choice in
                    1)
                        print_info "Using existing 'sai-simulator' environment."
                        offer_esmf_installation
                        print_success "Environment setup complete!"
                        print_info "To activate this environment in the future, run: conda activate sai-simulator"
                        return 0
                        ;;
                    2)
                        print_info "Removing existing conda environment..."
                        conda deactivate
                        conda env remove -n sai-simulator -y
                        print_success "Existing environment removed."
                        break
                        ;;
                    *)
                        print_error "Invalid choice. Please enter 1 or 2."
                        ;;
                esac
            done
        else
            print_warning "Python version $PYTHON_VERSION is not compatible (requires 3.11+)."
            echo ""
            if prompt_yes_no "Do you want to recreate the environment with Python 3.12?"; then
                conda deactivate
                conda env remove -n sai-simulator -y
                print_success "Existing environment removed."
            else
                print_error "Cannot continue with incompatible Python version."
                exit 1
            fi
        fi
    fi

    # Create conda environment with Python 3.12
    print_info "Creating conda environment 'sai-simulator' with Python 3.12..."
    conda create -n sai-simulator python=3.12 -c conda-forge -y

    print_success "Conda environment created."

    # Activate environment and install requirements
    print_info "Installing Python packages from requirements.txt..."

    conda activate sai-simulator

    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Requirements installed."
    else
        print_error "requirements.txt not found!"
        exit 1
    fi

    # Offer ESMF installation
    offer_esmf_installation

    print_success "Environment setup complete!"
    print_info "To activate this environment in the future, run: conda activate sai-simulator"
}

offer_esmf_installation() {
    # Install ESMF based on user's earlier choice
    if [ "$USER_INSTALL_ESMF" = "yes" ]; then
        print_info "Installing ESMF (version: $USER_ESMF_VERSION)..."
        install_esmf
    else
        print_warning "Skipping ESMF installation as per your configuration."
        print_info "If you need ESMF later, manual installation requires:"
        echo "  - gfortran: brew install gcc"
        echo "  - Xcode Command Line Tools: xcode-select --install"
    fi
}

setup_venv() {
    print_info "Setting up Python virtual environment..."

    # For venv, we need to check the system Python version first
    check_python_version

    # Check if .venv already exists
    if [ -d ".venv" ]; then
        print_warning "Virtual environment '.venv' already exists."

        # Activate and check Python version
        source .venv/bin/activate
        PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_info "Python version in existing .venv: $PYTHON_VERSION"

        if check_python_version_in_env; then
            print_success "Python version is compatible!"
            echo ""
            echo "Options:"
            echo "  1) Use existing virtual environment (recommended)"
            echo "  2) Delete and recreate the virtual environment"
            echo ""

            while true; do
                read -p "Enter your choice (1 or 2): " venv_choice
                case $venv_choice in
                    1)
                        print_info "Using existing virtual environment."
                        offer_esmf_installation
                        print_success "Environment setup complete!"
                        print_info "To activate this environment in the future, run: source .venv/bin/activate"
                        return 0
                        ;;
                    2)
                        print_info "Removing existing virtual environment..."
                        deactivate 2>/dev/null || true
                        rm -rf .venv
                        print_success "Existing virtual environment removed."
                        break
                        ;;
                    *)
                        print_error "Invalid choice. Please enter 1 or 2."
                        ;;
                esac
            done
        else
            print_warning "Python version $PYTHON_VERSION is not compatible (requires 3.11+)."
            echo ""
            if prompt_yes_no "Do you want to recreate the virtual environment with system Python?"; then
                deactivate 2>/dev/null || true
                rm -rf .venv
                print_success "Existing virtual environment removed."
            else
                print_error "Cannot continue with incompatible Python version."
                exit 1
            fi
        fi
    fi

    # Create virtual environment
    print_info "Creating virtual environment in .venv..."
    python -m venv .venv

    # Activate virtual environment
    print_info "Activating virtual environment..."
    source .venv/bin/activate

    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip

    # Install requirements
    if [ -f "requirements.txt" ]; then
        print_info "Installing Python packages from requirements.txt..."
        pip install -r requirements.txt
        print_success "Requirements installed."
    else
        print_error "requirements.txt not found!"
        exit 1
    fi

    # Offer ESMF installation
    offer_esmf_installation

    print_success "Environment setup complete!"
    print_info "To activate this environment in the future, run: source .venv/bin/activate"
}

configure_esmf_env() {
    # Configure ESMFMKFILE in shell configuration
    local esmf_mk_path="$1"

    # Determine which shell config file to use
    if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_CONFIG="$HOME/.bash_profile"
    else
        SHELL_CONFIG="$HOME/.bashrc"
    fi

    export ESMFMKFILE="$esmf_mk_path"

    # Add ESMFMKFILE to shell config if not already present
    if ! grep -q "ESMFMKFILE" "$SHELL_CONFIG" 2>/dev/null; then
        echo "" >> "$SHELL_CONFIG"
        echo "# ESMF Configuration (added by sai-simulator setup)" >> "$SHELL_CONFIG"
        echo "export ESMFMKFILE=\"$ESMFMKFILE\"" >> "$SHELL_CONFIG"
        print_success "Added ESMFMKFILE to $SHELL_CONFIG"
    else
        print_info "ESMFMKFILE already exists in $SHELL_CONFIG"
    fi
}

install_esmpy() {
    # Install ESMPy from the esmf directory
    local ORIGINAL_DIR=$(pwd)

    print_info "Installing ESMPy..."
    cd esmf/src/addon/esmpy
    if python -m pip install .; then
        print_success "ESMPy installed successfully!"
        cd "$ORIGINAL_DIR"

        # Add esmf directory to .gitignore if it exists
        if [ -f ".gitignore" ]; then
            if ! grep -qE "^esmf/?$" .gitignore 2>/dev/null; then
                echo "esmf/" >> .gitignore
                print_info "Added esmf/ to .gitignore"
            fi
        fi

        print_success "ESMF and ESMPy installation complete!"
        print_warning "Please restart your terminal or run: source $SHELL_CONFIG"
        return 0
    else
        print_error "ESMPy installation failed!"
        cd "$ORIGINAL_DIR"
        return 1
    fi
}

install_esmf() {
    print_info "Installing ESMF library..."

    # Store original directory
    ORIGINAL_DIR=$(pwd)

    # Check if esmf directory already exists
    if [ -d "esmf" ]; then
        print_info "Found existing esmf directory."

        # Try to find existing compiled libraries
        EXISTING_ESMF_MK=$(find esmf/lib -name "esmf.mk" 2>/dev/null | head -1)
        ESMF_IS_COMPILED=false

        if [ -n "$EXISTING_ESMF_MK" ]; then
            LIB_DIR=$(dirname "$EXISTING_ESMF_MK")

            # Check for required libraries
            if [[ "$OSTYPE" == "darwin"* ]]; then
                REQUIRED_FILES=("libesmf_fullylinked.dylib" "libesmf.dylib" "esmf.mk")
            else
                REQUIRED_FILES=("libesmf_fullylinked.so" "libesmf.so" "esmf.mk")
            fi

            all_exist=true
            for file in "${REQUIRED_FILES[@]}"; do
                if [ ! -f "$LIB_DIR/$file" ]; then
                    all_exist=false
                    break
                fi
            done

            if [ "$all_exist" = true ]; then
                ESMF_IS_COMPILED=true
                print_success "Found compiled ESMF installation at: $LIB_DIR"
            else
                print_warning "ESMF repository exists but is not fully compiled."
            fi
        else
            print_warning "ESMF repository exists but has no compiled libraries."
        fi

        # Ask user what to do with existing repo
        echo ""
        echo "Options:"
        if [ "$ESMF_IS_COMPILED" = true ]; then
            echo "  1) Use existing compiled ESMF (recommended)"
            echo "  2) Recompile existing ESMF with version $USER_ESMF_VERSION"
            echo "  3) Delete existing and perform fresh installation"
        else
            echo "  1) Compile the existing ESMF repository"
            echo "  2) Delete existing and perform fresh installation"
        fi
        echo ""

        while true; do
            if [ "$ESMF_IS_COMPILED" = true ]; then
                read -p "Enter your choice (1, 2, or 3): " esmf_choice
                case $esmf_choice in
                    1)
                        # Use existing compiled ESMF
                        export ESMFMKFILE="$EXISTING_ESMF_MK"
                        print_success "Using existing ESMF installation."
                        configure_esmf_env "$EXISTING_ESMF_MK"
                        install_esmpy
                        cd "$ORIGINAL_DIR"
                        return $?
                        ;;
                    2)
                        # Recompile with specified version
                        print_info "Will recompile ESMF with version $USER_ESMF_VERSION..."
                        break
                        ;;
                    3)
                        # Delete and fresh install
                        print_info "Removing existing esmf directory..."
                        rm -rf esmf
                        print_success "Existing esmf directory removed."
                        break
                        ;;
                    *)
                        print_error "Invalid choice. Please enter 1, 2, or 3."
                        ;;
                esac
            else
                read -p "Enter your choice (1 or 2): " esmf_choice
                case $esmf_choice in
                    1)
                        # Compile existing
                        print_info "Will compile existing ESMF repository..."
                        break
                        ;;
                    2)
                        # Delete and fresh install
                        print_info "Removing existing esmf directory..."
                        rm -rf esmf
                        print_success "Existing esmf directory removed."
                        break
                        ;;
                    *)
                        print_error "Invalid choice. Please enter 1 or 2."
                        ;;
                esac
            fi
        done
    fi

    # Check for required compilers
    print_info "Checking for required compilers..."
    if ! command -v gfortran &> /dev/null; then
        print_error "gfortran not found. Please install it first:"
        echo "  brew install gcc"
        cd "$ORIGINAL_DIR"
        return 1
    fi

    if ! command -v clang &> /dev/null || ! command -v clang++ &> /dev/null; then
        print_error "clang/clang++ not found. Please install Xcode Command Line Tools:"
        echo "  xcode-select --install"
        cd "$ORIGINAL_DIR"
        return 1
    fi

    print_success "All required compilers found."

    # Clone ESMF repository if it doesn't exist
    if [ ! -d "esmf" ]; then
        print_info "Cloning ESMF repository..."
        print_info "ESMF will be compiled in: $(pwd)/esmf"
        print_info "Size after compilation: ~500-600 MB"
        echo ""
        git clone https://github.com/esmf-org/esmf.git
    fi

    cd esmf
    git fetch

    # Checkout the specified version
    print_info "Checking out $USER_ESMF_VERSION..."
    git checkout "$USER_ESMF_VERSION"

    # Clean previous build artifacts
    print_info "Cleaning previous build artifacts..."
    make distclean 2>/dev/null || true

    # Set up ESMF build environment
    print_info "Setting up ESMF build environment..."
    export ESMF_DIR=$(pwd)
    export ESMF_BOPT=O
    export ESMF_CXX=/usr/bin/clang++
    export ESMF_C=/usr/bin/clang
    export ESMF_F90=gfortran
    export ESMF_COMPILER=gfortranclang
    export ESMF_SHARED_LIB_BUILD=ON
    export ESMF_CXXLINKLIBS="-lc++"
    export ESMF_F90LINKLIBS="-lc++"
    export ESMF_COMM=mpiuni

    print_info "Build configuration:"
    echo "  ESMF_DIR: $ESMF_DIR"
    echo "  ESMF_COMPILER: $ESMF_COMPILER"
    echo "  ESMF_COMM: $ESMF_COMM"
    echo "  ESMF_SHARED_LIB_BUILD: $ESMF_SHARED_LIB_BUILD"
    echo ""

    # Build ESMF
    print_info "Building ESMF (this may take 10-20 minutes)..."
    if ! make; then
        print_error "ESMF build failed!"
        cd "$ORIGINAL_DIR"
        return 1
    fi

    print_success "ESMF build completed successfully!"

    # Determine lib directory structure
    print_info "Locating ESMF libraries..."

    # Find the actual lib directory (libO for optimized builds)
    LIB_SUBDIR="lib${ESMF_BOPT}"

    # Find the OS-specific directory
    OS_DIR=$(find "$ESMF_DIR/lib/$LIB_SUBDIR" -maxdepth 1 -type d ! -name "$LIB_SUBDIR" | head -1)

    if [ -z "$OS_DIR" ]; then
        print_error "Could not find ESMF library directory!"
        cd "$ORIGINAL_DIR"
        return 1
    fi

    OS_SYSTEM=$(basename "$OS_DIR")
    LIB_PATH="$ESMF_DIR/lib/$LIB_SUBDIR/$OS_SYSTEM"

    print_info "Library path: $LIB_PATH"

    # Verify required files exist
    print_info "Verifying ESMF installation..."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        REQUIRED_FILES=("libesmf_fullylinked.dylib" "libesmf.dylib" "esmf.mk")
    else
        REQUIRED_FILES=("libesmf_fullylinked.so" "libesmf.so" "esmf.mk")
    fi

    all_found=true
    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "$LIB_PATH/$file" ]; then
            print_success "Found: $file"
        else
            print_error "Missing: $file"
            all_found=false
        fi
    done

    if [ "$all_found" = false ]; then
        print_error "ESMF installation verification failed!"
        cd "$ORIGINAL_DIR"
        return 1
    fi

    # Set ESMFMKFILE environment variable
    export ESMFMKFILE="$LIB_PATH/esmf.mk"
    print_success "ESMFMKFILE set to: $ESMFMKFILE"

    # Add to shell configuration
    print_info "Adding ESMFMKFILE to shell configuration..."

    # Determine which shell config file to use
    if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_CONFIG="$HOME/.bash_profile"
    else
        SHELL_CONFIG="$HOME/.bashrc"
    fi

    # Add ESMFMKFILE to shell config if not already present
    if ! grep -q "ESMFMKFILE" "$SHELL_CONFIG" 2>/dev/null; then
        echo "" >> "$SHELL_CONFIG"
        echo "# ESMF Configuration (added by sai-simulator setup)" >> "$SHELL_CONFIG"
        echo "export ESMFMKFILE=\"$ESMFMKFILE\"" >> "$SHELL_CONFIG"
        print_success "Added ESMFMKFILE to $SHELL_CONFIG"
    else
        print_info "ESMFMKFILE already exists in $SHELL_CONFIG"
    fi

    # Install ESMPy
    print_info "Installing ESMPy..."
    cd src/addon/esmpy
    if python -m pip install .; then
        print_success "ESMPy installed successfully!"
    else
        print_error "ESMPy installation failed!"
        cd "$ORIGINAL_DIR"
        return 1
    fi

    # Return to original directory
    cd "$ORIGINAL_DIR"

    # Add esmf directory to .gitignore if it exists
    if [ -f ".gitignore" ]; then
        # Check if esmf or esmf/ already exists as an uncommented line
        if ! grep -qE "^esmf/?$" .gitignore 2>/dev/null; then
            echo "esmf/" >> .gitignore
            print_info "Added esmf/ to .gitignore"
        else
            print_info "esmf is already in .gitignore"
        fi
    fi

    print_success "ESMF and ESMPy installation complete!"
    print_info "ESMF installed at: $(pwd)/esmf (~500-600 MB)"
    print_warning "Please restart your terminal or run: source $SHELL_CONFIG"
}

# B. Data Download
download_data() {
    print_header "Step B: Data Download"

    print_info "This will download ~140GB of data and takes about 30 minutes."

    # Install AWS CLI
    print_info "Installing AWS CLI..."
    pip install awscli
    print_success "AWS CLI installed."

    # Create data directory
    print_info "Creating data directory..."
    mkdir -p "$RAW_DATA_DIR"
    print_success "Directory created: $RAW_DATA_DIR"

    # Download data from S3
    print_info "Downloading data from S3 bucket (this may take a while)..."
    start_time=$(date +%s)
    aws s3 sync s3://reflective-simulator-bucket/v1.3.0/ "$RAW_DATA_DIR/" --no-sign-request
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    print_success "Data downloaded from S3."
    print_info "Download completed in ${minutes} minutes and ${seconds} seconds."

    # Set overwrite flag based on earlier choice
    OVERWRITE_FLAG=""
    if [ "$USER_OVERWRITE_LENS" = "yes" ]; then
        OVERWRITE_FLAG="--overwrite"
        print_info "Will overwrite existing LENS files"
    else
        print_info "Will skip existing LENS files"
    fi

    # Download LENS NCAR data
    print_info "Downloading LENS NCAR data for TREFHTMX..."
    if [ -f "scripts/run_download_CESM_LENS.sh" ]; then
        sh scripts/run_download_CESM_LENS.sh TREFHTMX monthly "$RAW_DATA_DIR" 001 002 003 $OVERWRITE_FLAG
        print_success "TREFHTMX data downloaded."
    else
        print_error "scripts/run_download_CESM_LENS.sh not found!"
    fi

    print_info "Downloading LENS NCAR data for TREFHTMN..."
    if [ -f "scripts/run_download_CESM_LENS.sh" ]; then
        sh scripts/run_download_CESM_LENS.sh TREFHTMN monthly "$RAW_DATA_DIR" 001 002 003 $OVERWRITE_FLAG
        print_success "TREFHTMN data downloaded."
    else
        print_error "scripts/run_download_CESM_LENS.sh not found!"
    fi

    print_success "Data download complete!"
}

# C. Data Processing
process_data() {
    print_header "Step C: Data Processing"

    # Set daily overwrite flag based on earlier choice
    DAILY_OVERWRITE_FLAG=""
    if [ "$USER_OVERWRITE_DAILY" = "yes" ]; then
        DAILY_OVERWRITE_FLAG="--ignore-existing"
        print_info "Will overwrite existing daily processed files"
    else
        print_info "Will skip existing daily processed files"
    fi

    # Process daily data
    print_info "Processing daily data to create monthly values..."
    if [ -f "scripts/process_all_daily.sh" ]; then
        sh scripts/process_all_daily.sh "$RAW_DATA_DIR" $DAILY_OVERWRITE_FLAG
        print_success "Daily data processed."
    else
        print_error "scripts/process_all_daily.sh not found!"
    fi

    # Set monthly overwrite flag based on earlier choice
    MONTHLY_OVERWRITE_FLAG=""
    if [ "$USER_OVERWRITE_MONTHLY" = "yes" ]; then
        MONTHLY_OVERWRITE_FLAG="--overwrite"
        print_info "Will overwrite existing monthly processed files"
    else
        print_info "Will skip existing monthly processed files"
    fi

    # Process monthly data
    print_info "Processing monthly data to create annual values..."
    mkdir -p "$PROCESSED_DATA_DIR"
    if [ -f "scripts/process_all_monthly.sh" ]; then
        sh scripts/process_all_monthly.sh "$RAW_DATA_DIR" "$PROCESSED_DATA_DIR" $MONTHLY_OVERWRITE_FLAG
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
        sh scripts/fit_all.sh "$PROCESSED_DATA_DIR" "$MODELS_DIR"
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

# Run main function
main
