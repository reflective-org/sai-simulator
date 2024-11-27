# Argument validation check
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <data_dir>"
    exit 1
fi

python scripts/process_daily_gauss.py --var tas --data_dir $1
python scripts/process_daily_gauss.py --var pr --data_dir $1
