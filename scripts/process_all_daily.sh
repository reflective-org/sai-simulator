# Argument validation check
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <data_dir> [--ignore-existing]"
    exit 1
fi

DATA_DIR="$1"
IGNORE_EXISTING_FLAG=""

if [[ "${2:-}" == "--ignore-existing" ]]; then
    IGNORE_EXISTING_FLAG="--ignore_existing"
fi

python scripts/process_daily_gauss.py --var tas --data_dir "$DATA_DIR" $IGNORE_EXISTING_FLAG
python scripts/process_daily_gauss.py --var pr --data_dir "$DATA_DIR" $IGNORE_EXISTING_FLAG
