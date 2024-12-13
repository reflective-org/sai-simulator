# Argument validation check
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <data_dir> <output_dir>"
    exit 1
fi


python scripts/process_monthly_gauss.py  --var tas --data_dir $1 --output_dir $2
python scripts/process_monthly_gauss.py  --var pr --data_dir $1 --output_dir $2
python scripts/process_monthly_gauss.py  --var tasmax --data_dir $1 --output_dir $2
python scripts/process_monthly_gauss.py  --var tasmin --data_dir $1 --output_dir $2
python scripts/process_monthly_gauss.py  --var tas_above_35 --data_dir $1 --output_dir $2
python scripts/process_monthly_gauss.py  --var tas_above_40 --data_dir $1 --output_dir $2
python scripts/process_monthly_gauss.py  --var tas_below_0 --data_dir $1 --output_dir $2
python scripts/process_monthly_gauss.py  --var pr_above_10 --data_dir $1 --output_dir $2
python scripts/process_monthly_gauss.py  --var pr_above_20 --data_dir $1 --output_dir $2
python scripts/process_monthly_gauss.py  --var p-e --data_dir $1 --output_dir $2
