"""Summary figures comparing CESM_params.mat with CESM_params_updated.mat on the CIDER path.

Makes two figures for the equal-injection ('all_lats') scenario:
- aod_summary.png: the AOD parameters [beta, alpha, gamma] per injection latitude, and the AOD
  time series per latitude they produce.
- maps_tas_summary_all_lats.png: 2080-2100 mean tas delta for each file, and their difference.

Usage:
    python scripts/compare_cider_params_summary.py --out PR57_final
"""
import argparse
import sys
import tempfile
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from cartopy.util import add_cyclic_point

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_cider_params import (  # noqa: E402
    ALL_LATS, DATA_DIR, LATE, build_annual_injection, cider_delta, data_dir_for, global_mean,
)
from backend.variable import AOD_from_injection_vectorized  # noqa: E402

FILES = {
    'CESM_params.mat': DATA_DIR / 'CESM_params.mat',
    'CESM_params_updated.mat': DATA_DIR / 'CESM_params_updated.mat',
}
COLORS = {'CESM_params.mat': '#2a78d6', 'CESM_params_updated.mat': '#eb6834'}
# Row order of param_AOD_all, inferred from pattern_AOD_all (see CESM_params_README.md).
LATS = ['60N', '30N', '15N', '0°', '15S', '30S', '60S']


def plot_aod(mats: dict, injection: np.ndarray, out: Path) -> None:
    """AOD parameters per latitude (top) and the AOD they give for the injection (bottom)."""
    monthly = np.repeat(injection, 12, axis=1) / 12
    t = 2035 + np.arange(monthly.shape[1]) / 12
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 12, height_ratios=[1.1, 1, 1], hspace=0.45, wspace=1.2)

    x = np.arange(7)
    width = 0.8 / len(mats)
    for c, name in enumerate(['beta', 'alpha', 'gamma']):
        ax = fig.add_subplot(gs[0, 4 * c:4 * c + 4])
        for k, (label, m) in enumerate(mats.items()):
            ax.bar(x + (k - (len(mats) - 1) / 2) * width, m['param_AOD_all'][:, c], width * 0.92,
                   color=COLORS[label], label=label)
        ax.set_xticks(x, LATS)
        ax.set_title(f'param_AOD_all: {name}')
        ax.set_xlabel('injection latitude')
        if c == 0:
            fig.legend(*ax.get_legend_handles_labels(), loc='upper center', ncol=len(mats), frameon=False,
                       bbox_to_anchor=(0.5, 0.95), fontsize=12)

    series_axes = [fig.add_subplot(gs[1 + i // 4, 3 * (i % 4):3 * (i % 4) + 3]) for i in range(8)]
    for label, m in mats.items():
        aod = np.array([AOD_from_injection_vectorized(m['param_AOD_all'][i], monthly[i]) for i in range(7)])
        for i in range(7):
            series_axes[i].plot(t, aod[i], color=COLORS[label], lw=2, label=label)
        series_axes[7].plot(t, aod.sum(axis=0), color=COLORS[label], lw=2, label=label)
    for i, ax in enumerate(series_axes):
        ax.set_title(f'AOD from {LATS[i]} injection' if i < 7 else 'total AOD')
        ax.set_ylim(bottom=0)
    series_axes[0].set_ylabel('AOD')
    series_axes[4].set_ylabel('AOD')
    fig.suptitle('AOD parameters and the AOD they give for 3 Tg/yr at every latitude from 2040', fontsize=14)
    fig.savefig(out / 'aod_summary.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_maps(deltas: dict, out: Path) -> None:
    """Late-century tas delta for each file (one colour scale) and updated − old (its own scale)."""
    late = {k: d.sel(time=slice(*LATE)).mean('time') for k, d in deltas.items()}
    old, new = 'CESM_params.mat', 'CESM_params_updated.mat'
    diff = late[new] - late[old]
    lim = float(max(abs(f).quantile(0.995) for f in late.values()))
    dlim = float(abs(diff).quantile(0.995))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), subplot_kw={'projection': ccrs.Robinson(central_longitude=180)})
    for ax, (title, f, vmax) in zip(axes, [(old, late[old], lim), (new, late[new], lim), (f'{new} − {old}', diff, dlim)]):
        data, lon = add_cyclic_point(f.values, coord=f.lon.values)
        im = ax.pcolormesh(lon, f.lat, data, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.coastlines(lw=0.5, color='0.2')
        ax.set_title(f'{title}\nglobal mean {float(global_mean(f)):+.3f} K', fontsize=11)
        if ax is axes[1]:
            fig.colorbar(im, ax=axes[:2], orientation='horizontal', shrink=0.5, pad=0.05, extend='both', label='Δtas (K)')
        if ax is axes[2]:
            fig.colorbar(im, ax=axes[2], orientation='horizontal', shrink=0.8, pad=0.05, extend='both', label='difference in Δtas (K)')
    fig.suptitle(f"CIDER tas delta, {LATE[0]}-{LATE[1]} mean, equal injection at all latitudes "
                 "(3 Tg/yr each from 2040)", fontsize=13)
    fig.savefig(out / 'maps_tas_summary_all_lats.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, required=True, help='output folder for the figures')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    mats = {k: scipy.io.loadmat(p) for k, p in FILES.items()}
    injection = build_annual_injection(ALL_LATS)
    plot_aod(mats, injection, args.out)

    with tempfile.TemporaryDirectory() as tmp:
        deltas = {k: cider_delta('tas', data_dir_for(p, Path(tmp)), injection) for k, p in FILES.items()}
    plot_maps(deltas, args.out)
    for k, d in deltas.items():
        print(f'{k}: {LATE[0]}-{LATE[1]} global-mean Δtas {float(global_mean(d.sel(time=slice(*LATE)).mean("time"))):.4f} K')


if __name__ == '__main__':
    main()
