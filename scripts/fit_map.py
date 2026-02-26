import fire
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from joblib import dump, load
from collections import defaultdict
from sklearn.linear_model import LinearRegression


def fit_map(var, data_dir, output_dir, num_bootstrap_replicates=100, ignore_existing=False):
    data_dir = Path(data_dir)
    input_fair_path = data_dir / "input_fair.nc"
    data_dir = data_dir / var
    output_path = data_dir / f"output_gauss-baseline.nc"
    if not output_path.exists():
        raise ValueError(f"output_path {output_path} does not exist. Need to run process_monthly_gauss.py first.")

    input_fair = xr.open_dataset(input_fair_path)
    output = xr.open_dataset(output_path)

    # get the overlapping time range between input_fair and GCM model output
    fair_time_min, fair_time_max = input_fair.time.min().values, input_fair.time.max().values
    print(f"input_fair time range: {fair_time_min} to {fair_time_max}")
    output_time_min, output_time_max = output.time.min().values, output.time.max().values
    print(f"output time range: {output_time_min} to {output_time_max}")
    # overlapping time range
    time_min, time_max = max(fair_time_min, output_time_min), min(fair_time_max, output_time_max)
    
    if time_min > time_max:
        raise ValueError(f"No overlapping times between input_fair ({fair_time_min} to {fair_time_max}) and output ({output_time_min} to {output_time_max})")
    
    if time_max < fair_time_max or time_max < output_time_max:
        print(f"Note: input_fair has times from {fair_time_min} to {fair_time_max}, output has times from {output_time_min} to {output_time_max}")
        print(f"Using overlapping time range: {time_min} to {time_max}")
    
    # take only the overlapping time range
    output = output.sel(time=slice(time_min, time_max))
    input_fair = input_fair.sel(time=slice(time_min, time_max))

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    model_dir = output_dir / var
    model_dir.mkdir(exist_ok=True)

    np.random.seed(42)
    # Train a FaIR global tas -> ScenarioMIP regional tas model
    #   a. input_fair has variable tas and dimensions (ssp, time)
    #   b. output_scenario_mip has variable tas and dimensions (model, ssp, time, lat, lon)
    # Find overlapping ssp scenarios
    fair_ssps = set(input_fair.ssp.values)
    scenariomip_ssps = set(output.ssp.values)
    scenariomip_ssps = scenariomip_ssps & fair_ssps
    # Train a linear regression model for each model and bootstrap replicate
    scenariomip_models = set(output.model.values)
    model2bootstrapped_fair_emulators = defaultdict(list)
    for model in scenariomip_models:
        print(f"Training models for {model}")
        for i in tqdm(range(num_bootstrap_replicates), total=num_bootstrap_replicates):
            Xs, ys = [], []
            model_path = model_dir / f"fair_to_smip_{model}_{i}.joblib"
            if model_path.exists() and not ignore_existing:
                reg = load(model_path)
                model2bootstrapped_fair_emulators[model].append(reg)
                continue
            for ssp in scenariomip_ssps:
                fair_model_data = input_fair.sel(ssp=ssp)
                scenariomip_model_data = output.sel(model=model, ssp=ssp)

                fair_model_data, scenariomip_model_data = xr.align(
                    fair_model_data, scenariomip_model_data, join="inner"
                )
                
                X = fair_model_data.tas.values # (time)
                X = X[:, np.newaxis] # (time, 1)
                y = scenariomip_model_data[var].values # (time, lat, lon)
                if np.isnan(y).sum() > 0:
                    continue
                # Sample X and y with replacement
                bootstrap_inds = np.random.choice(X.shape[0], X.shape[0])
                X = X[bootstrap_inds]
                y = y[bootstrap_inds]
                # y -> (time, lat, lon) -> (time, lat * lon)
                y = y.reshape(y.shape[0], -1)
                Xs.append(X)
                ys.append(y)
            X = np.concatenate(Xs, axis=0) # (time * ssp, 1)
            y = np.concatenate(ys, axis=0) # (time * ssp, lat * lon)
            # For every climate model, train a linear regression that inputs global fair tas and outputs regional smip tas
            reg = LinearRegression().fit(X, y)
            model2bootstrapped_fair_emulators[model].append(reg)
            dump(reg, model_path)


if __name__ == "__main__":
    fire.Fire(fit_map)
