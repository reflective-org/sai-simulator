# Argument validation check
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <data_dir> <output_dir>"
    exit 1
fi

# Fit all the global T -> map regressors
python scripts/fit_map.py --var tas --data_dir $1 --output_dir $2
python scripts/fit_map.py --var pr --data_dir $1 --output_dir $2
python scripts/fit_map.py --var tasmax --data_dir $1 --output_dir $2
python scripts/fit_map.py --var tasmin --data_dir $1 --output_dir $2
python scripts/fit_map.py --var tas_above_35 --data_dir $1 --output_dir $2
python scripts/fit_map.py --var tas_above_40 --data_dir $1 --output_dir $2
python scripts/fit_map.py --var tas_below_0 --data_dir $1 --output_dir $2
python scripts/fit_map.py --var pr_above_10 --data_dir $1 --output_dir $2
python scripts/fit_map.py --var pr_above_20 --data_dir $1 --output_dir $2
python scripts/fit_map.py --var p-e --data_dir $1 --output_dir $2

# Fit all the delta global T -> delta map regressors
python scripts/fit_delta.py --var tas --data_dir $1 --output_dir $2
python scripts/fit_delta.py --var pr --data_dir $1 --output_dir $2
python scripts/fit_delta.py --var tasmax --data_dir $1 --output_dir $2
python scripts/fit_delta.py --var tasmin --data_dir $1 --output_dir $2
python scripts/fit_delta.py --var tas_above_35 --data_dir $1 --output_dir $2
python scripts/fit_delta.py --var tas_above_40 --data_dir $1 --output_dir $2
python scripts/fit_delta.py --var tas_below_0 --data_dir $1 --output_dir $2
python scripts/fit_delta.py --var pr_above_10 --data_dir $1 --output_dir $2
python scripts/fit_delta.py --var pr_above_20 --data_dir $1 --output_dir $2
python scripts/fit_delta.py --var p-e --data_dir $1 --output_dir $2
python scripts/fit_delta.py --var so2 --data_dir $1 --output_dir $2