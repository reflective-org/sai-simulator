"""Validate tasmin / tasmax on the custom-injection (CIDER) path.

1. End to end: runs ``get_outputs`` for tas, tasmin, tasmax, p-e and icefrac with an equal
   injection at all latitudes (3 Tg/yr each from 2040), exactly as the CIDER dashboard calls it,
   checks every output is finite, and plots the global-mean series with and without SAI.
2. CIDER vs uniform path: both deltas are divided by their own global-mean tas delta over
   2080-2100, so the two paths are compared per K of global cooling.

``get_outputs`` writes cache files, so the cache dir is cloned into a temporary folder first.

Usage:
    python scripts/validate_cider_tasmin_tasmax.py --out PR57_pipeline \\
        --data-dir data/processed_pr55 --model-dir data/models_pr55_3model \\
        --cache-dir data/cache_pr55_3model
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from cartopy.util import add_cyclic_point

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.delta import get_regional_delta_without_rampup  # noqa: E402
from backend.get_outputs import get_outputs, get_temp_diff  # noqa: E402
from backend.variable import CESM_PARAMS_FILE, get_variable_regional_delta  # noqa: E402

SSP = 'SSP2-4.5'
UNIFORM_TARGET = 1.0  # uniform dashboard default; the per-K comparison does not depend on it
LATE = (2080, 2100)
VARS = ['tas', 'tasmin', 'tasmax', 'p-e', 'icefrac']
COLORS = {'CIDER': '#2a78d6', 'uniform': '#eb6834'}


def equal_injection() -> np.ndarray:
    """(7 lats, 66 yrs): 3 Tg/yr at every latitude from 2040."""
    injection = np.zeros((7, 66))
    injection[:, 5:] = 3.0
    return injection


def global_mean(da):
    return da.weighted(np.cos(np.deg2rad(da.lat))).mean(('lat', 'lon'))


def data_dir_with_params(data_dir: Path, tmp: Path) -> Path:
    """``data_dir`` plus the CESM parameter file, if ``data_dir`` lacks it."""
    if (data_dir / CESM_PARAMS_FILE).exists():
        return data_dir
    d = tmp / 'data'
    d.mkdir()
    for entry in data_dir.iterdir():
        if entry.name != CESM_PARAMS_FILE:  # e.g. a broken link of that name
            (d / entry.name).symlink_to(entry.resolve())
    (d / CESM_PARAMS_FILE).symlink_to(REPO / 'data' / 'processed' / CESM_PARAMS_FILE)
    return d


def check_end_to_end(data_dir: Path, model_dir: Path, cache_dir: Path, out: Path) -> list[str]:
    injection = equal_injection()
    outputs = get_outputs(SSP, None, None, None, 2091, 2100, None, None,
                          data_dir, model_dir, cache_dir, var=VARS, variable_injection=injection)
    lines = ['## End-to-end get_outputs (decade 2091-2100)', '',
             '| var | finite fraction, regional_mean | finite fraction, regional_delta_mean | global-mean delta 2091-2100 |',
             '|---|---|---|---|']
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=True)
    for var in VARS:
        o = outputs[var]
        delta = o['regional_delta_mean']
        finite_mean = float(np.isfinite(o['regional_mean']).mean())
        finite_delta = float(np.isfinite(delta).mean())
        lines.append(f'| {var} | {finite_mean:.3f} | {finite_delta:.3f} | {float(global_mean(delta)):+.4f} |')
        if var in ['tas', 'tasmin', 'tasmax']:
            ax = axes[['tas', 'tasmin', 'tasmax'].index(var)]
            m = o['mean_over_time']
            ax.plot(m['historical_model'].time, m['historical_model'], color='0.3', lw=1.5, label='historical model')
            ax.plot(m['no_sai'].time, m['no_sai'], color='#eb6834', lw=2, label=f'no SAI ({SSP})')
            ax.plot(m['with_sai'].time, m['with_sai'], color='#2a78d6', lw=2, label='with SAI (CIDER)')
            ax.set_title(f'{var}: global mean, delta from pre-industrial')
            ax.set_xlabel('year')
    axes[0].set_ylabel('K')
    axes[0].legend(frameon=False)
    fig.suptitle('get_outputs on the CIDER path, equal injection at all latitudes (3 Tg/yr each from 2040)')
    fig.tight_layout()
    fig.savefig(out / 'end_to_end_mean_over_time.png', dpi=130)
    plt.close(fig)
    return lines + ['']


def compare_with_uniform(data_dir: Path, model_dir: Path, cache_dir: Path, out: Path) -> list[str]:
    injection = equal_injection()
    temp_diff = get_temp_diff(data_dir, model_dir, cache_dir, SSP, UNIFORM_TARGET)
    late = {}
    for path in ['CIDER', 'uniform']:
        for var in ['tas', 'tasmin', 'tasmax']:
            if path == 'CIDER':
                d = get_variable_regional_delta(var, data_dir, cache_dir, injection)
            else:
                d = get_regional_delta_without_rampup(var, data_dir, model_dir, cache_dir, SSP, UNIFORM_TARGET, temp_diff)
            late[path, var] = d.sel(time=slice(*LATE)).mean('time')
    per_k = {}
    for path in ['CIDER', 'uniform']:
        gm_tas = abs(float(global_mean(late[path, 'tas'])))
        for var in ['tasmin', 'tasmax']:
            f = late[path, var] / gm_tas
            if path == 'uniform':  # the uniform grid carries its own lon coords
                f = f.assign_coords(lon=late['CIDER', var].lon.values, lat=late['CIDER', var].lat.values)
            per_k[path, var] = f

    lines = ['## CIDER vs uniform path, per K of global-mean tas cooling (2080-2100)', '',
             '| var | CIDER global mean | uniform global mean | CIDER − uniform, area-weighted RMS | pattern correlation |',
             '|---|---|---|---|---|']
    fig = plt.figure(figsize=(21, 10))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.7])
    for r, var in enumerate(['tasmin', 'tasmax']):
        c, u = per_k['CIDER', var], per_k['uniform', var]
        diff = c - u
        w = np.cos(np.deg2rad(c.lat)).broadcast_like(c)
        rms = float(np.sqrt((diff ** 2 * w).sum() / w.sum()))
        ca, ua = c - float(global_mean(c)), u - float(global_mean(u))
        corr = float((ca * ua * w).sum() / np.sqrt((ca ** 2 * w).sum() * (ua ** 2 * w).sum()))
        lines.append(f'| {var} | {float(global_mean(c)):+.3f} | {float(global_mean(u)):+.3f} | {rms:.3f} | {corr:.3f} |')

        lim = float(max(abs(c).quantile(0.995), abs(u).quantile(0.995)))
        dlim = float(abs(diff).quantile(0.995))
        for j, (title, f, vmax) in enumerate([
            (f'CIDER {var}', c, lim),
            (f'uniform {var}', u, lim),
            ('CIDER − uniform', diff, dlim),
        ]):
            ax = fig.add_subplot(gs[r, j], projection=ccrs.Robinson(central_longitude=180))
            data, lon = add_cyclic_point(f.values, coord=f.lon.values)
            im = ax.pcolormesh(lon, f.lat, data, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.coastlines(lw=0.5, color='0.2')
            ax.set_title(f'{title}\nglobal mean {float(global_mean(f)):+.3f} K per K', fontsize=10)
            fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, pad=0.05, extend='both', label='K per K of global tas cooling')
        ax = fig.add_subplot(gs[r, 3])
        for path, f in [('CIDER', c), ('uniform', u)]:
            ax.plot(f.mean('lon'), f.lat, color=COLORS[path], lw=2, label=path)
        ax.axvline(0, color='0.6', lw=0.8)
        ax.set_title(f'{var}: zonal mean')
        ax.set_xlabel('K per K')
        ax.set_ylabel('latitude')
        ax.legend(frameon=False)
    fig.suptitle(f'tasmin / tasmax response per K of global-mean tas cooling, {LATE[0]}-{LATE[1]}: '
                 f'CIDER (equal injection, {CESM_PARAMS_FILE}) vs uniform path ({SSP}, target {UNIFORM_TARGET} K)',
                 fontsize=12)
    fig.savefig(out / 'cider_vs_uniform_per_K.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    return lines + ['']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data_dir = data_dir_with_params(args.data_dir.resolve(), tmp)
        cache_dir = tmp / 'cache'
        # APFS clone: instant and takes no space; fall back to a plain copy elsewhere.
        if subprocess.run(['cp', '-cR', str(args.cache_dir), str(cache_dir)]).returncode != 0:
            shutil.copytree(args.cache_dir, cache_dir)
        lines = [f'# CIDER tasmin/tasmax validation ({CESM_PARAMS_FILE})', '']
        lines += check_end_to_end(data_dir, args.model_dir, cache_dir, args.out)
        lines += compare_with_uniform(data_dir, args.model_dir, cache_dir, args.out)
    (args.out / 'numbers.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
