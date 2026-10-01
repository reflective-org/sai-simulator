"""Compare CESM_params.mat with a newer CESM parameter file on the custom-injection (CIDER) path.

CESM_params_updated.mat adds TMN/TMX and also refits AOD, T, P, Q, RH and U10. This script
shows how much the refit moves the CIDER outputs (tas, p-e, icefrac) compared with
CESM_params.mat, and sanity-checks the TMN/TMX entries.

Every delta is computed with the real ``get_variable_regional_delta``; each file is exposed to
it through a temporary data dir that links the file under the name the backend loads
(``backend.variable.CESM_PARAMS_FILE``).

The icefrac check needs the no-SAI tas baseline, so it also takes the regional-model data,
model and cache dirs used by the rest of the simulator.

The newer file defaults to CESM_params_updated.mat; pass --new to compare another one.

Usage:
    python scripts/compare_cider_params.py --out PR57_final \
        --baseline-data-dir data/processed_pr55 --model-dir data/models_pr55_3model \
        --cache-dir data/cache_pr55_3model
"""
import argparse
import sys
import tempfile
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.baseline import get_regional_map  # noqa: E402
from backend.icefrac import predict_icefrac  # noqa: E402
from backend.variable import CESM_PARAMS_FILE, AOD_from_injection_vectorized, get_variable_regional_delta  # noqa: E402

DATA_DIR = REPO / 'data' / 'processed'
OLD = DATA_DIR / 'CESM_params.mat'
NEW = DATA_DIR / 'CESM_params_updated.mat'
SSP = 'SSP2-4.5'
LATE = (2080, 2100)

OLD_COLOR, NEW_COLOR = '#2a78d6', '#eb6834'
VAR_COLORS = {'T': '#1baf7a', 'TMN': '#2a78d6', 'TMX': '#eb6834'}
ROW_LABELS = ['inj 1', 'inj 2', 'inj 3', 'inj 4', 'inj 5', 'inj 6', 'inj 7', 'CO2']

# Two test injections, each (7 lats in CESM_params.mat order, 7 periods: 2035-2040, then
# 2040-2050 … 2090-2100), in Tg/yr per latitude.
# 'ramp': the array from climate_velocity_verification/mock_calc_cider.py (from test.py). That
#   script labels its rows as decades, but its expansion (like the one below) treats rows as
#   latitudes, so injection latitudes 1, 4 and 7 get nothing.
# 'all_lats': 3 Tg/yr at every latitude from 2040, so every row of param_AOD_all is exercised.
RAMP = np.array([
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.02, 0.10, 0.17, 0.22, 0.28, 0.32, 0.36],
    [0.31, 1.83, 3.11, 4.16, 5.18, 6.10, 6.85],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.57, 3.38, 5.75, 7.69, 9.57, 11.27, 12.66],
    [0.13, 0.79, 1.34, 1.79, 2.23, 2.62, 2.95],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
])
ALL_LATS = np.full((7, 7), 3.0)
ALL_LATS[:, 0] = 0.0
INJECTIONS = {'ramp': RAMP, 'all_lats': ALL_LATS}


def build_annual_injection(decades: np.ndarray) -> np.ndarray:
    """Expand a (7 lats, 7 periods) grid to (7 lats, 66 yrs), as variable_launch_gradio does."""
    first = np.repeat(decades[:, [0]], 6, axis=1)
    rest = np.repeat(decades[:, 1:], 10, axis=1)
    return np.concatenate([first, rest], axis=1)


def data_dir_for(mat_path: Path, tmp: Path) -> Path:
    """A data dir where the file the backend loads points at ``mat_path``."""
    d = tmp / mat_path.stem
    d.mkdir()
    (d / CESM_PARAMS_FILE).symlink_to(mat_path)
    (d / 'correct_lat.npy').symlink_to(DATA_DIR / 'correct_lat.npy')
    return d


def global_mean(da: xr.DataArray) -> xr.DataArray:
    return da.weighted(np.cos(np.deg2rad(da.lat))).mean(('lat', 'lon'))


def cider_delta(var: str, data_dir: Path, injection: np.ndarray) -> xr.DataArray:
    if var == 'p-e':
        return (get_variable_regional_delta('pr', data_dir, data_dir, injection)
                - get_variable_regional_delta('e', data_dir, data_dir, injection))
    return get_variable_regional_delta(var, data_dir, data_dir, injection)


def plot_params(old: dict, new: dict, out: Path) -> str:
    """Relative change (new/old − 1) of every [tau, mu] and AOD parameter, per row."""
    names = ['T', 'P', 'Q', 'RH', 'U10']
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=False)
    table = ['| param | row | old | new | rel. change |', '|---|---|---|---|---|']
    for ax, v in zip(axes.flat, names):
        o, n = old[f'param_{v}_all'], new[f'param_{v}_all']
        rel = n / o - 1
        x = np.arange(o.shape[0])
        ax.bar(x - 0.2, 100 * rel[:, 0], 0.38, color=OLD_COLOR, label='tau')
        ax.bar(x + 0.2, 100 * rel[:, 1], 0.38, color=NEW_COLOR, label='mu')
        ax.axhline(0, color='0.4', lw=0.8)
        ax.set_xticks(x, ROW_LABELS, rotation=45)
        ax.set_title(f'param_{v}_all')
        ax.set_ylabel('new / old − 1 (%)')
        for r in range(o.shape[0]):
            for c, pn in enumerate(['tau', 'mu']):
                table.append(f'| {v} {pn} | {ROW_LABELS[r]} | {o[r, c]:.5g} | {n[r, c]:.5g} | {100 * rel[r, c]:+.2f} % |')
    ax = axes.flat[-1]
    o, n = old['param_AOD_all'], new['param_AOD_all']
    rel = n / o - 1
    x = np.arange(o.shape[0])
    for c, (pn, col) in enumerate(zip(['beta', 'alpha', 'gamma'], [OLD_COLOR, NEW_COLOR, '#1baf7a'])):
        ax.bar(x + (c - 1) * 0.27, 100 * rel[:, c], 0.25, color=col, label=pn)
        for r in range(o.shape[0]):
            table.append(f'| AOD {pn} | {ROW_LABELS[r]} | {o[r, c]:.5g} | {n[r, c]:.5g} | {100 * rel[r, c]:+.2f} % |')
    ax.axhline(0, color='0.4', lw=0.8)
    ax.set_xticks(x, ROW_LABELS[:7], rotation=45)
    ax.set_title('param_AOD_all')
    ax.set_ylabel('new / old − 1 (%)')
    ax.legend(frameon=False)
    axes.flat[0].legend(frameon=False)
    fig.suptitle(f'Parameter change, {NEW.name} vs CESM_params.mat')
    fig.tight_layout()
    fig.savefig(out / 'params_relative_change.png', dpi=130)
    plt.close(fig)
    return '\n'.join(table)


def plot_aod(old: dict, new: dict, injection: np.ndarray, tag: str, out: Path) -> dict:
    """Monthly AOD per injection latitude for one test injection, old vs new parameters."""
    monthly = np.repeat(injection, 12, axis=1) / 12
    t = 2035 + np.arange(monthly.shape[1]) / 12
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), sharex=True)
    totals = {}
    for label, params, col in [('old', old, OLD_COLOR), ('new', new, NEW_COLOR)]:
        aod = np.array([AOD_from_injection_vectorized(params['param_AOD_all'][i], monthly[i])
                        for i in range(7)])
        totals[label] = aod.sum(axis=0)
        for i in range(7):
            axes.flat[i].plot(t, aod[i], color=col, lw=2, label=label)
        axes.flat[7].plot(t, totals[label], color=col, lw=2, label=label)
    for i in range(7):
        axes.flat[i].set_title(f'{ROW_LABELS[i]} (row {i} of the injection)')
    axes.flat[7].set_title('sum over injection latitudes')
    axes.flat[7].legend(frameon=False)
    for ax in axes[:, 0]:
        ax.set_ylabel('AOD contribution')
    fig.suptitle(f'AOD from the {tag!r} injection, old vs new param_AOD_all')
    fig.tight_layout()
    fig.savefig(out / f'aod_{tag}.png', dpi=130)
    plt.close(fig)
    return totals


def plot_delta_comparison(var: str, units: str, d_old: xr.DataArray, d_new: xr.DataArray, tag: str, out: Path) -> dict:
    """Global-mean series, late-century zonal mean, and old/new/difference maps."""
    gm_old, gm_new = global_mean(d_old), global_mean(d_new)
    late_old = d_old.sel(time=slice(*LATE)).mean('time')
    late_new = d_new.sel(time=slice(*LATE)).mean('time')
    diff = late_new - late_old

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(gm_old.time, gm_old, color=OLD_COLOR, lw=2, label='CESM_params.mat')
    ax.plot(gm_new.time, gm_new, color=NEW_COLOR, lw=2, label=NEW.name)
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_title(f'{var}: global-mean delta')
    ax.set_ylabel(units)
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(late_old.mean('lon'), late_old.lat, color=OLD_COLOR, lw=2, label='old')
    ax.plot(late_new.mean('lon'), late_new.lat, color=NEW_COLOR, lw=2, label='new')
    ax.axvline(0, color='0.6', lw=0.8)
    ax.set_title(f'zonal mean, {LATE[0]}-{LATE[1]}')
    ax.set_xlabel(units)
    ax.set_ylabel('latitude')
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(gm_old.time, gm_new - gm_old, color='0.2', lw=2)
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_title('global mean, new − old')
    ax.set_ylabel(units)

    lim = float(max(abs(late_old).max(), abs(late_new).max()))
    dlim = float(abs(diff).max()) or 1.0
    for j, (field, title, vmax) in enumerate([
        (late_old, 'old', lim), (late_new, 'new', lim), (diff, 'new − old', dlim),
    ]):
        ax = fig.add_subplot(gs[1, j])
        field.plot(ax=ax, cmap='RdBu_r', vmin=-vmax, vmax=vmax, cbar_kwargs={'label': units})
        ax.set_title(f'{title}, {LATE[0]}-{LATE[1]}')
        ax.set_xlabel('')
        ax.set_ylabel('')
    fig.suptitle(f'CIDER {var} delta for the {tag!r} injection: old vs new parameter file')
    fig.tight_layout()
    fig.savefig(out / f'delta_{var}_old_vs_new_{tag}.png', dpi=130)
    plt.close(fig)

    late_gm_old, late_gm_new = float(gm_old.sel(time=slice(*LATE)).mean()), float(gm_new.sel(time=slice(*LATE)).mean())
    return {
        'late_gm_old': late_gm_old,
        'late_gm_new': late_gm_new,
        'max_abs_gm_diff': float(abs(gm_new - gm_old).max()),
        'max_abs_late_cell_diff': float(abs(diff).max()),
    }


def plot_maps(var: str, units: str, cmap: str, d_old: xr.DataArray, d_new: xr.DataArray, tag: str, out: Path) -> None:
    """Robinson maps of the late-century delta: old file, new file (shared scale) and new − old."""
    from cartopy.util import add_cyclic_point
    fields = {
        'CESM_params.mat': d_old.sel(time=slice(*LATE)).mean('time'),
        NEW.name: d_new.sel(time=slice(*LATE)).mean('time'),
    }
    fields['new − old'] = fields[NEW.name] - fields['CESM_params.mat']
    lim = float(max(abs(fields['CESM_params.mat']).quantile(0.995), abs(fields[NEW.name]).quantile(0.995)))
    dlim = float(abs(fields['new − old']).quantile(0.995)) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(20, 4.6), subplot_kw={'projection': ccrs.Robinson(central_longitude=180)})
    for ax, (title, f) in zip(axes, fields.items()):
        vmax = dlim if title == 'new − old' else lim
        data, lon = add_cyclic_point(f.values, coord=f.lon.values)
        im = ax.pcolormesh(lon, f.lat, data, transform=ccrs.PlateCarree(), cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.coastlines(lw=0.5, color='0.2')
        gm = float(global_mean(f))
        ax.set_title(f'{title}\nglobal mean {gm:+.3f} {units}', fontsize=11)
        fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, pad=0.05, extend='both', label=f'Δ{var} ({units})')
    fig.suptitle(f'CIDER {var} delta, {LATE[0]}-{LATE[1]} mean, {tag!r} injection', fontsize=13)
    fig.savefig(out / f'maps_{var}_old_vs_new_{tag}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_icefrac(tas_no_sai: xr.DataArray, d_old: xr.DataArray, d_new: xr.DataArray, model_dir: Path, tag: str, out: Path) -> dict:
    """Arctic ice fraction with SAI (Arctic mean) from the old and new tas deltas."""
    res = {}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, d, col in [('old', d_old, OLD_COLOR), ('new', d_new, NEW_COLOR)]:
        d = d.assign_coords(lat=tas_no_sai.lat, lon=tas_no_sai.lon)
        ice = predict_icefrac(tas_no_sai + d, model_dir)
        arctic = global_mean(ice.sel(lat=slice(60, 90)))  # predict_icefrac fills non-Arctic cells with 0
        res[label] = arctic
        ax.plot(arctic.time, arctic, color=col, lw=2, label=label)
    ax.set_title(f'Arctic (60-90N) mean ice fraction with SAI, {tag!r} injection')
    ax.set_ylabel('ice fraction')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / f'icefrac_old_vs_new_{tag}.png', dpi=130)
    plt.close(fig)
    return {'max_abs_diff': float(abs(res['new'] - res['old']).max())}


def plot_tmn_tmx_checks(new: dict, lat: np.ndarray, out: Path) -> dict:
    """Base maps, the TMX − TMN range, and response patterns per injection latitude."""
    lon = np.linspace(0, 360, new['pattern_T_base'].shape[0], endpoint=False)

    def field(a):
        return xr.DataArray(np.asarray(a, float).T, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon})

    bases = {v: field(new[f'pattern_{v}_base']) - 273.15 for v in ['TMN', 'T', 'TMX']}
    tmx_minus_tmn = bases['TMX'] - bases['TMN']
    w = np.cos(np.deg2rad(tmx_minus_tmn.lat))
    stats = {
        'tmx_minus_tmn_min': float(tmx_minus_tmn.min()),
        'tmx_minus_tmn_global_mean': float(tmx_minus_tmn.weighted(w).mean()),
        'frac_area_tmx_le_tmn': float((tmx_minus_tmn <= 0).astype(float).weighted(w).mean()),
        'frac_area_T_outside_TMN_TMX': float(((bases['T'] < bases['TMN']) | (bases['T'] > bases['TMX']))
                                             .astype(float).weighted(w).mean()),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5))
    for ax, v in zip(axes.flat[:3], ['TMN', 'T', 'TMX']):
        bases[v].plot(ax=ax, cmap='RdBu_r', vmin=-40, vmax=40, cbar_kwargs={'label': '°C'})
        ax.set_title(f'pattern_{v}_base')
    tmx_minus_tmn.plot(ax=axes.flat[3], cmap='viridis', vmin=0, cbar_kwargs={'label': 'K'})
    axes.flat[3].set_title(f'TMX − TMN; min {stats["tmx_minus_tmn_min"]:.2f} K')
    for ax in axes.flat:
        ax.set_xlabel('')
        ax.set_ylabel('')
    fig.suptitle(f'{NEW.name} baselines')
    fig.tight_layout()
    fig.savefig(out / 'tmn_tmx_baselines.png', dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(3, 7, figsize=(24, 8), sharex=True, sharey=True)
    for r, v in enumerate(['TMN', 'T', 'TMX']):
        pats = new[f'pattern_{v}_all']
        lim = float(np.nanmax(abs(pats[..., :7])))
        for i in range(7):
            ax = axes[r, i]
            im = ax.pcolormesh(lon, lat, np.asarray(pats[..., i], float).T, cmap='RdBu_r', vmin=-lim, vmax=lim)
            ax.set_title(f'{v} {ROW_LABELS[i]}', fontsize=9)
        fig.colorbar(im, ax=axes[r, :], shrink=0.8, label=f'pattern_{v}_all')
    fig.suptitle(f'Response patterns per injection latitude ({NEW.name})')
    fig.savefig(out / 'tmn_t_tmx_patterns.png', dpi=110)
    plt.close(fig)
    return stats


def plot_tmn_tmx_deltas(deltas: dict, tag: str, out: Path) -> dict:
    """CIDER tas / tasmin / tasmax deltas from the new file, side by side."""
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3)
    ax_ts, ax_zm = fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2])
    names = {'tasmin': 'TMN', 'tas': 'T', 'tasmax': 'TMX'}
    lates = {v: d.sel(time=slice(*LATE)).mean('time') for v, d in deltas.items()}
    lim = float(max(abs(f).max() for f in lates.values()))
    res = {}
    for j, (v, d) in enumerate(deltas.items()):
        col = VAR_COLORS[names[v]]
        gm = global_mean(d)
        res[v] = float(gm.sel(time=slice(*LATE)).mean())
        ax_ts.plot(gm.time, gm, color=col, lw=2, label=v)
        ax_zm.plot(lates[v].mean('lon'), lates[v].lat, color=col, lw=2, label=v)
        ax = fig.add_subplot(gs[1, j])
        lates[v].plot(ax=ax, cmap='RdBu_r', vmin=-lim, vmax=lim, cbar_kwargs={'label': 'K'})
        ax.set_title(f'{v}, {LATE[0]}-{LATE[1]}')
        ax.set_xlabel('')
        ax.set_ylabel('')
    ax_ts.axhline(0, color='0.6', lw=0.8)
    ax_ts.set_title('global-mean delta')
    ax_ts.set_ylabel('K')
    ax_ts.legend(frameon=False)
    ax_zm.axvline(0, color='0.6', lw=0.8)
    ax_zm.set_title(f'zonal mean, {LATE[0]}-{LATE[1]}')
    ax_zm.set_xlabel('K')
    ax_zm.legend(frameon=False)
    fig.suptitle(f'CIDER deltas from {NEW.name} for the {tag!r} injection')
    fig.tight_layout()
    fig.savefig(out / f'delta_tas_tasmin_tasmax_new_{tag}.png', dpi=130)
    plt.close(fig)
    return res


def cider_tmn_tmx_delta(key: str, new_mat: dict, injection: np.ndarray, lat: np.ndarray) -> xr.DataArray:
    """Same chain as get_variable_regional_delta, for TMN/TMX which it does not support yet."""
    from backend.variable import pattern_from_all_injections_and_CO2
    monthly = np.repeat(injection, 12, axis=1) / 12
    inj = np.concatenate([monthly.T, np.zeros((monthly.shape[1], 1))], axis=1)
    d = pattern_from_all_injections_and_CO2(inj, new_mat['param_AOD_all'],
                                            new_mat[f'param_{key}_all'], new_mat[f'pattern_{key}_all'])
    d = d.reshape(d.shape[0], d.shape[1], -1, 12).mean(-1)
    lon = np.linspace(0, 360, d.shape[0], endpoint=False)
    return xr.DataArray(d.transpose(2, 1, 0), dims=('time', 'lat', 'lon'),
                        coords={'time': np.arange(2035, 2101), 'lat': lat, 'lon': lon})


def main() -> None:
    global NEW
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, required=True, help='output folder for plots and numbers')
    parser.add_argument('--baseline-data-dir', type=Path, required=True, help='data dir for the no-SAI tas baseline')
    parser.add_argument('--model-dir', type=Path, required=True, help='model dir (tas and icefrac_arctic fits)')
    parser.add_argument('--cache-dir', type=Path, required=True, help='cache dir matching --model-dir')
    parser.add_argument('--new', type=Path, default=NEW, help='parameter file to compare with CESM_params.mat')
    args = parser.parse_args()
    NEW = args.new.resolve()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    old, new = scipy.io.loadmat(OLD), scipy.io.loadmat(NEW)
    lat = np.load(DATA_DIR / 'correct_lat.npy')
    tas_no_sai, _ = get_regional_map('tas', args.baseline_data_dir, args.model_dir, args.cache_dir, SSP)
    lines = [f'# CIDER parameter-file comparison: CESM_params.mat vs {NEW.name} '
             '(generated by scripts/compare_cider_params.py)', '']
    lines += ['## Parameters', '', plot_params(old, new, out), '']

    for tag, decades in INJECTIONS.items():
        injection = build_annual_injection(decades)
        lines += [f'## Injection {tag!r} (total {injection.sum(axis=0).max():.1f} Tg/yr at peak)', '']
        totals = plot_aod(old, new, injection, tag, out)
        late = slice((LATE[0] - 2035) * 12, None)
        lines += [f'- peak total AOD old {totals["old"].max():.4f}, new {totals["new"].max():.4f}',
                  f'- {LATE[0]}-{LATE[1]} mean total AOD old {totals["old"][late].mean():.4f}, '
                  f'new {totals["new"][late].mean():.4f}']

        with tempfile.TemporaryDirectory() as tmp:
            dirs = {'old': data_dir_for(OLD, Path(tmp)), 'new': data_dir_for(NEW, Path(tmp))}
            deltas = {v: {k: cider_delta(v, d, injection) for k, d in dirs.items()} for v in ['tas', 'p-e']}
        for v, units, cmap in [('tas', 'K', 'RdBu_r'), ('p-e', 'mm/day', 'BrBG')]:
            plot_maps(v, units, cmap, deltas[v]['old'], deltas[v]['new'], tag, out)
            s = plot_delta_comparison(v, units, deltas[v]['old'], deltas[v]['new'], tag, out)
            lines.append(f'- {v}: {LATE[0]}-{LATE[1]} global-mean delta old {s["late_gm_old"]:.4f}, '
                         f'new {s["late_gm_new"]:.4f} {units}; max |Δ global mean| {s["max_abs_gm_diff"]:.4f}; '
                         f'max |Δ| in a {LATE[0]}-{LATE[1]} cell {s["max_abs_late_cell_diff"]:.4f}')
        s = plot_icefrac(tas_no_sai, deltas['tas']['old'], deltas['tas']['new'], args.model_dir, tag, out)
        lines.append(f'- icefrac ({SSP} baseline): max |Δ Arctic-mean ice fraction| {s["max_abs_diff"]:.4f}')

        tmn_tmx = {
            'tasmin': cider_tmn_tmx_delta('TMN', new, injection, lat),
            'tas': deltas['tas']['new'],
            'tasmax': cider_tmn_tmx_delta('TMX', new, injection, lat),
        }
        s = plot_tmn_tmx_deltas(tmn_tmx, tag, out)
        lines += [f'- new file, {LATE[0]}-{LATE[1]} global-mean delta {v}: {x:.4f} K' for v, x in s.items()] + ['']

    s = plot_tmn_tmx_checks(new, lat, out)
    lines += ['## TMN/TMX baseline sanity checks (new file)', ''] + [f'- {k}: {v:.4f}' for k, v in s.items()]

    (out / 'numbers.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
