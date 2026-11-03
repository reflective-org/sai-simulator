# ESMF/ESMPy install: configure shell env, build from source, and install ESMPy.

# Set ESMFMKFILE in the current shell and persist it to the user's shell rc file.
configure_esmf_env() {
    local esmf_mk_path="$1"
    detect_shell_config

    export ESMFMKFILE="$esmf_mk_path"

    if ! grep -q "ESMFMKFILE" "$SHELL_CONFIG" 2>/dev/null; then
        echo "" >> "$SHELL_CONFIG"
        echo "# ESMF Configuration (added by sai-simulator setup)" >> "$SHELL_CONFIG"
        echo "export ESMFMKFILE=\"$ESMFMKFILE\"" >> "$SHELL_CONFIG"
        print_success "Added ESMFMKFILE to $SHELL_CONFIG"
    else
        print_info "ESMFMKFILE already exists in $SHELL_CONFIG"
    fi
}

# Install ESMPy from the cloned esmf source tree. Caller must be in the project
# root (where the esmf/ directory lives) before invoking.
install_esmpy() {
    local ORIGINAL_DIR
    ORIGINAL_DIR=$(pwd)

    print_info "Installing ESMPy..."
    cd esmf/src/addon/esmpy
    if python -m pip install .; then
        print_success "ESMPy installed successfully!"
        cd "$ORIGINAL_DIR"

        if [ -f ".gitignore" ]; then
            if ! grep -qE "^esmf/?$" .gitignore 2>/dev/null; then
                echo "esmf/" >> .gitignore
                print_info "Added esmf/ to .gitignore"
            else
                print_info "esmf is already in .gitignore"
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

# Confirm gfortran and the platform C/C++ toolchain are available.
_check_esmf_compilers() {
    print_info "Checking for required compilers..."

    if ! command -v gfortran &> /dev/null; then
        print_error "gfortran not found."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "  brew install gcc"
        else
            echo "  sudo dnf install -y gcc-gfortran gcc gcc-c++"
        fi
        return 1
    fi

    if [[ "$OSTYPE" == "darwin"* ]]; then
        if ! command -v clang &> /dev/null || ! command -v clang++ &> /dev/null; then
            print_error "clang/clang++ not found. Please install Xcode Command Line Tools:"
            echo "  xcode-select --install"
            return 1
        fi
    else
        if ! command -v gcc &> /dev/null || ! command -v g++ &> /dev/null; then
            print_error "gcc/g++ not found."
            echo "  sudo {package-manager} install -y gcc gcc-c++ OR install -y build-essentials"
            return 1
        fi
    fi

    print_success "All required compilers found."
}

# Export the ESMF_* env vars consumed by the upstream Makefile. Caller must be
# in the esmf/ source root so $(pwd) resolves correctly for ESMF_DIR.
_set_esmf_build_env() {
    print_info "Setting up ESMF build environment..."

    export ESMF_DIR
    ESMF_DIR=$(pwd)
    export ESMF_BOPT=O
    export ESMF_F90=gfortran
    export ESMF_SHARED_LIB_BUILD=ON
    export ESMF_COMM=mpiuni

    if [[ "$OSTYPE" == "darwin"* ]]; then
        export ESMF_CXX=/usr/bin/clang++
        export ESMF_C=/usr/bin/clang
        export ESMF_COMPILER=gfortranclang
        export ESMF_CXXLINKLIBS="-lc++"
        export ESMF_F90LINKLIBS="-lc++"
    else
        export ESMF_CXX=/usr/bin/g++
        export ESMF_C=/usr/bin/gcc
        export ESMF_COMPILER=gfortran
        unset ESMF_CXXLINKLIBS
        unset ESMF_F90LINKLIBS

        # Strip macOS-only linker flags that leak from inherited environments.
        local flag_var flag_val
        for flag_var in LDFLAGS CFLAGS CXXFLAGS FFLAGS FCFLAGS CPPFLAGS; do
            flag_val="${!flag_var}"
            if [ -n "$flag_val" ]; then
                flag_val="$(printf '%s' "$flag_val" | sed 's/-Wl,-ld_classic//g; s/-ld_classic//g')"
                flag_val="$(printf '%s' "$flag_val" | xargs)"
                export "$flag_var=$flag_val"
            fi
        done
    fi

    print_info "Build configuration:"
    echo "  ESMF_DIR: $ESMF_DIR"
    echo "  ESMF_COMPILER: $ESMF_COMPILER"
    echo "  ESMF_COMM: $ESMF_COMM"
    echo "  ESMF_SHARED_LIB_BUILD: $ESMF_SHARED_LIB_BUILD"
    echo ""
}

# Verify the expected shared libs and esmf.mk exist under the given lib path.
# Returns 0 if all present, 1 otherwise. Emits per-file status to stdout.
_verify_esmf_artifacts() {
    local lib_path="$1"
    local required_files
    local all_found=true
    local file

    if [[ "$OSTYPE" == "darwin"* ]]; then
        required_files=("libesmf_fullylinked.dylib" "libesmf.dylib" "esmf.mk")
    else
        required_files=("libesmf_fullylinked.so" "libesmf.so" "esmf.mk")
    fi

    for file in "${required_files[@]}"; do
        if [ -f "$lib_path/$file" ]; then
            print_success "Found: $file"
        else
            print_error "Missing: $file"
            all_found=false
        fi
    done

    [ "$all_found" = true ]
}

# Quietly check whether $1 contains the required ESMF artifacts (no output).
_esmf_artifacts_present() {
    _verify_esmf_artifacts "$1" > /dev/null 2>&1
}

install_esmf() {
    print_info "Installing ESMF library..."

    local ORIGINAL_DIR
    ORIGINAL_DIR=$(pwd)

    # If a prior esmf/ checkout exists, let the user reuse it, recompile, or wipe.
    if [ -d "esmf" ]; then
        print_info "Found existing esmf directory."

        local EXISTING_ESMF_MK
        local ESMF_IS_COMPILED=false
        EXISTING_ESMF_MK=$(find esmf/lib -name "esmf.mk" 2>/dev/null | head -1)

        if [ -n "$EXISTING_ESMF_MK" ]; then
            local LIB_DIR
            LIB_DIR=$(dirname "$EXISTING_ESMF_MK")

            if _esmf_artifacts_present "$LIB_DIR"; then
                ESMF_IS_COMPILED=true
                print_success "Found compiled ESMF installation at: $LIB_DIR"
            else
                print_warning "ESMF repository exists but is not fully compiled."
            fi
        else
            print_warning "ESMF repository exists but has no compiled libraries."
        fi

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
                if is_non_interactive; then
                    # Auto-select safest fast path for unattended runs.
                    export ESMFMKFILE="$EXISTING_ESMF_MK"
                    print_success "NON_INTERACTIVE: using existing compiled ESMF installation."
                    configure_esmf_env "$EXISTING_ESMF_MK"
                    install_esmpy
                    local rc=$?
                    cd "$ORIGINAL_DIR"
                    return $rc
                fi
                read -p "Enter your choice (1, 2, or 3): " esmf_choice
                case $esmf_choice in
                    1)
                        export ESMFMKFILE="$EXISTING_ESMF_MK"
                        print_success "Using existing ESMF installation."
                        configure_esmf_env "$EXISTING_ESMF_MK"
                        install_esmpy
                        local rc=$?
                        cd "$ORIGINAL_DIR"
                        return $rc
                        ;;
                    2)
                        print_info "Will recompile ESMF with version $USER_ESMF_VERSION..."
                        break
                        ;;
                    3)
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
                if is_non_interactive; then
                    print_info "NON_INTERACTIVE: compiling existing ESMF repository."
                    break
                fi
                read -p "Enter your choice (1 or 2): " esmf_choice
                case $esmf_choice in
                    1)
                        print_info "Will compile existing ESMF repository..."
                        break
                        ;;
                    2)
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

    if ! _check_esmf_compilers; then
        cd "$ORIGINAL_DIR"
        return 1
    fi

    # Clone ESMF repository if it doesn't exist.
    if [ ! -d "esmf" ]; then
        print_info "Cloning ESMF repository..."
        print_info "ESMF will be compiled in: $(pwd)/esmf"
        print_info "Size after compilation: ~500-600 MB"
        echo ""
        git clone https://github.com/esmf-org/esmf.git
    fi

    cd esmf
    git fetch

    print_info "Checking out $USER_ESMF_VERSION..."
    git checkout "$USER_ESMF_VERSION"

    print_info "Cleaning previous build artifacts..."
    make distclean 2>/dev/null || true

    _set_esmf_build_env

    print_info "Building ESMF (this may take 10-20 minutes)..."
    if ! make; then
        print_error "ESMF build failed!"
        cd "$ORIGINAL_DIR"
        return 1
    fi

    print_success "ESMF build completed successfully!"

    print_info "Locating ESMF libraries..."
    local LIB_SUBDIR="lib${ESMF_BOPT}"
    local OS_DIR
    OS_DIR=$(find "$ESMF_DIR/lib/$LIB_SUBDIR" -maxdepth 1 -type d ! -name "$LIB_SUBDIR" | head -1)

    if [ -z "$OS_DIR" ]; then
        print_error "Could not find ESMF library directory!"
        cd "$ORIGINAL_DIR"
        return 1
    fi

    local OS_SYSTEM
    OS_SYSTEM=$(basename "$OS_DIR")
    local LIB_PATH="$ESMF_DIR/lib/$LIB_SUBDIR/$OS_SYSTEM"
    print_info "Library path: $LIB_PATH"

    print_info "Verifying ESMF installation..."
    if ! _verify_esmf_artifacts "$LIB_PATH"; then
        print_error "ESMF installation verification failed!"
        cd "$ORIGINAL_DIR"
        return 1
    fi

    configure_esmf_env "$LIB_PATH/esmf.mk"
    print_success "ESMFMKFILE set to: $ESMFMKFILE"

    cd "$ORIGINAL_DIR"

    if ! install_esmpy; then
        return 1
    fi

    print_info "ESMF installed at: $(pwd)/esmf (~500-600 MB)"
}
