# Step A: Python environment setup. Three variants — conda, venv, and "use the
# already-active interpreter" — share the same post-install ESMF hand-off.

# Trigger ESMF install if the user opted in earlier; otherwise just inform them.
offer_esmf_installation() {
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

setup_conda() {
    print_info "Setting up Conda environment..."

    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed. Please install Conda first."
        print_info "Visit: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi

    local CONDA_BASE
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"

    local CURRENT_ENV="${CONDA_DEFAULT_ENV:-}"

    if [ -n "$CURRENT_ENV" ] && [ "$CURRENT_ENV" != "base" ]; then
        print_info "Currently active conda environment: $CURRENT_ENV"

        if [ "$CURRENT_ENV" = "sai-simulator" ]; then
            # Already in sai-simulator: check Python version, then either continue or recreate.
            print_info "You are already in the 'sai-simulator' environment."

            PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
            print_info "Python version in current environment: $PYTHON_VERSION"

            if check_python_version_in_env; then
                print_success "Python version is compatible!"
                echo ""
                if is_non_interactive || prompt_yes_no "Do you want to continue with the existing 'sai-simulator' environment?"; then
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
                if is_non_interactive || prompt_yes_no "Do you want to recreate the environment with Python 3.12?"; then
                    conda deactivate
                    conda env remove -n sai-simulator -y
                    print_success "Existing environment removed."
                else
                    print_error "Cannot continue with incompatible Python version."
                    exit 1
                fi
            fi
        else
            # A different conda environment is active — deactivate or bail.
            print_warning "A different conda environment '$CURRENT_ENV' is currently active."
            echo ""
            echo "Options:"
            echo "  1) Deactivate '$CURRENT_ENV' and create/use 'sai-simulator'"
            echo "  2) Exit and let me manually manage environments"
            echo ""

            if is_non_interactive; then
                print_info "NON_INTERACTIVE: deactivating '$CURRENT_ENV' and continuing."
                conda deactivate
                print_success "Environment deactivated."
            else
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
    fi

    # If sai-simulator already exists on disk, offer to reuse or recreate.
    if conda env list | grep -q "^sai-simulator "; then
        print_info "Conda environment 'sai-simulator' already exists."

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

            if is_non_interactive; then
                print_info "NON_INTERACTIVE: using existing 'sai-simulator' environment."
                offer_esmf_installation
                print_success "Environment setup complete!"
                print_info "To activate this environment in the future, run: conda activate sai-simulator"
                return 0
            fi

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
            if is_non_interactive || prompt_yes_no "Do you want to recreate the environment with Python 3.12?"; then
                conda deactivate
                conda env remove -n sai-simulator -y
                print_success "Existing environment removed."
            else
                print_error "Cannot continue with incompatible Python version."
                exit 1
            fi
        fi
    fi

    print_info "Creating conda environment 'sai-simulator' with Python 3.12..."
    conda create -n sai-simulator python=3.12 -c conda-forge -y
    print_success "Conda environment created."

    print_info "Installing Python packages from requirements.txt..."
    conda activate sai-simulator

    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Requirements installed."
    else
        print_error "requirements.txt not found!"
        exit 1
    fi

    offer_esmf_installation

    print_success "Environment setup complete!"
    print_info "To activate this environment in the future, run: conda activate sai-simulator"
}

setup_venv() {
    print_info "Setting up Python virtual environment..."

    # For venv we need the system Python to satisfy the version requirement.
    check_python_version

    if [ -d ".venv" ]; then
        print_warning "Virtual environment '.venv' already exists."

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

            if is_non_interactive; then
                print_info "NON_INTERACTIVE: using existing virtual environment."
                offer_esmf_installation
                print_success "Environment setup complete!"
                print_info "To activate this environment in the future, run: source .venv/bin/activate"
                return 0
            fi

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
            if is_non_interactive || prompt_yes_no "Do you want to recreate the virtual environment with system Python?"; then
                deactivate 2>/dev/null || true
                rm -rf .venv
                print_success "Existing virtual environment removed."
            else
                print_error "Cannot continue with incompatible Python version."
                exit 1
            fi
        fi
    fi

    print_info "Creating virtual environment in .venv..."
    python -m venv .venv

    print_info "Activating virtual environment..."
    source .venv/bin/activate

    print_info "Upgrading pip..."
    pip install --upgrade pip

    if [ -f "requirements.txt" ]; then
        print_info "Installing Python packages from requirements.txt..."
        pip install -r requirements.txt
        print_success "Requirements installed."
    else
        print_error "requirements.txt not found!"
        exit 1
    fi

    offer_esmf_installation

    print_success "Environment setup complete!"
    print_info "To activate this environment in the future, run: source .venv/bin/activate"
}

# Reuse whichever Python interpreter is already active. Skips env creation
# but still installs requirements (unless SKIP_REQUIREMENTS_INSTALL) and ESMF.
setup_existing_python_env() {
    print_info "Using existing Python environment (no conda/venv creation)."

    if ! command -v python &> /dev/null; then
        print_error "Python is not available in the current environment."
        exit 1
    fi

    PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_info "Detected Python version: $PYTHON_VERSION"

    if ! check_python_version_in_env; then
        print_error "Python version $PYTHON_VERSION is not compatible (requires 3.11+)."
        exit 1
    fi

    if is_truthy "$SKIP_REQUIREMENTS_INSTALL"; then
        print_warning "Skipping requirements install in existing environment (SKIP_REQUIREMENTS_INSTALL=$SKIP_REQUIREMENTS_INSTALL)."
    else
        if [ -f "requirements.txt" ]; then
            print_info "Installing Python packages from requirements.txt..."
            python -m pip install -r requirements.txt
            print_success "Requirements installed."
        else
            print_error "requirements.txt not found!"
            exit 1
        fi
    fi

    # Keep ESMF behavior consistent with the normal setup flow.
    offer_esmf_installation

    print_success "Environment setup complete in existing Python environment."
}
