# Argument validation check
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <data_dir> <output_dir>"
    exit 1
fi

# Fit all the global T -> map regressors
echo "fit regional tas"
python scripts/fit_map.py --var tas --data_dir $1 --output_dir $2
echo "fit regional pr"
python scripts/fit_map.py --var pr --data_dir $1 --output_dir $2
echo "fit regional tasmax"
python scripts/fit_map.py --var tasmax --data_dir $1 --output_dir $2
echo "fit regional tasmin"
python scripts/fit_map.py --var tasmin --data_dir $1 --output_dir $2
echo "fit regional tas_above_35"
python scripts/fit_map.py --var tas_above_35 --data_dir $1 --output_dir $2
echo "fit regional tas_above_40"
python scripts/fit_map.py --var tas_above_40 --data_dir $1 --output_dir $2
echo "fit regional tas_below_0"
python scripts/fit_map.py --var tas_below_0 --data_dir $1 --output_dir $2
echo "fit regional pr_above_10"
python scripts/fit_map.py --var pr_above_10 --data_dir $1 --output_dir $2
echo "fit regional pr_above_20"
python scripts/fit_map.py --var pr_above_20 --data_dir $1 --output_dir $2
echo "fit regional p-e"
python scripts/fit_map.py --var p-e --data_dir $1 --output_dir $2
echo "fit regional icefrac"
python scripts/fit_map.py --var icefrac --data_dir $1 --output_dir $2

# Fit all the delta global T -> delta map regressors
echo "fit delta tas"
python scripts/fit_delta.py --var tas --data_dir $1 --output_dir $2
echo "fit delta pr"
python scripts/fit_delta.py --var pr --data_dir $1 --output_dir $2
echo "fit delta tasmax"
python scripts/fit_delta.py --var tasmax --data_dir $1 --output_dir $2
echo "fit delta tasmin"
python scripts/fit_delta.py --var tasmin --data_dir $1 --output_dir $2
echo "fit delta tas_above_35"
python scripts/fit_delta.py --var tas_above_35 --data_dir $1 --output_dir $2
echo "fit delta tas_above_40"
python scripts/fit_delta.py --var tas_above_40 --data_dir $1 --output_dir $2
echo "fit delta tas_below_0"
python scripts/fit_delta.py --var tas_below_0 --data_dir $1 --output_dir $2
echo "fit delta pr_above_10"
python scripts/fit_delta.py --var pr_above_10 --data_dir $1 --output_dir $2
echo "fit delta pr_above_20"
python scripts/fit_delta.py --var pr_above_20 --data_dir $1 --output_dir $2
echo "fit delta p-e"
python scripts/fit_delta.py --var p-e --data_dir $1 --output_dir $2
echo "fit delta so2"
python scripts/fit_delta.py --var so2 --data_dir $1 --output_dir $2
echo "fit delta icefrac"
python scripts/fit_delta.py --var icefrac --data_dir $1 --output_dir $2
