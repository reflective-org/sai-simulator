# UI primitives, predicates, and small utilities shared across setup modules.
# Sourced by setup.sh; relies on the outer `set -e` semantics and on color/state
# globals defined here.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}\n"
}

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error()   { echo -e "${RED}✗ $1${NC}"; }
print_info()    { echo -e "${BLUE}ℹ $1${NC}"; }

is_non_interactive() {
    [[ "$NON_INTERACTIVE" =~ ^([Yy][Ee][Ss]|[Tt][Rr][Uu][Ee]|1)$ ]]
}

is_truthy() {
    local value="${1:-}"
    [[ "$value" =~ ^([Yy][Ee][Ss]|[Tt][Rr][Uu][Ee]|1)$ ]]
}

prompt_yes_no() {
    if is_non_interactive; then
        local default_answer="${NON_INTERACTIVE_DEFAULT_ANSWER:-yes}"
        if [[ "$default_answer" =~ ^([Yy]|yes|YES)$ ]]; then
            print_info "$1 -> auto-answer: yes (NON_INTERACTIVE)"
            return 0
        else
            print_info "$1 -> auto-answer: no (NON_INTERACTIVE)"
            return 1
        fi
    fi

    while true; do
        read -p "$1 (y/n): " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes (y) or no (n).";;
        esac
    done
}

# Validate the system Python (used before venv creation). Aborts on failure.
check_python_version() {
    print_info "Checking Python version..."

    if ! command -v python &> /dev/null; then
        print_error "Python 3 is not installed."
        print_error "Please install Python 3.11 or higher."
        print_info "Visit: https://www.python.org/downloads/"
        exit 1
    fi

    PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_MAJOR=$(python -c 'import sys; print(sys.version_info[0])')
    PYTHON_MINOR=$(python -c 'import sys; print(sys.version_info[1])')

    print_info "Detected Python version: $PYTHON_VERSION"

    if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]; }; then
        print_error "Python version $PYTHON_VERSION is not supported."
        print_error "This project requires Python 3.11 or higher."
        print_info "Please upgrade your Python installation."
        print_info "Visit: https://www.python.org/downloads/"
        exit 1
    fi

    print_success "Python version $PYTHON_VERSION is compatible!"
}

# Non-fatal version check for an already-activated environment.
check_python_version_in_env() {
    local PYTHON_VERSION
    local PYTHON_MAJOR
    local PYTHON_MINOR
    PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_MAJOR=$(python -c 'import sys; print(sys.version_info[0])')
    PYTHON_MINOR=$(python -c 'import sys; print(sys.version_info[1])')

    if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]; }; then
        return 1
    fi
    return 0
}

# Resolve the user's shell rc file and assign it to the global $SHELL_CONFIG.
# Prefers zsh when available since the script is documented as bash/zsh-compatible.
detect_shell_config() {
    if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_CONFIG="$HOME/.bash_profile"
    else
        SHELL_CONFIG="$HOME/.bashrc"
    fi
}
