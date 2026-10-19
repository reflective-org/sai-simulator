# CESM parameter files for the custom-injection (CIDER) path

`backend/variable.py:get_variable_regional_delta` loads these files with `scipy.io.loadmat`.
They hold the parameters of the SO2 → AOD → climate-response chain and the spatial response
patterns fitted to CESM.

## Files

| File | Created | sha1 | Used by |
|---|---|---|---|
| `CESM_params.mat` | 2025-03-06 | `b825a28891edcf88f9b688c71a70a0bea5bc62ab` | not loaded; kept for comparison |
| `CESM_params_updated.mat` | 2026-09-30 | `d776093fd1d5d4dedd4195afdadc65ec0c1ec7df` | `get_variable_regional_delta` (`tas`, `tasmin`, `tasmax`, `pr`, `e`) |

`CESM_params_updated.mat` comes from the new CIDER version, by the same author and fitting
pipeline as `CESM_params.mat`. It was refit with minimum and maximum temperature (TMN, TMX) added.
The current copy (2026-09-30) replaces one from 2026-09-29
(`96c27fb9c36456580116ad948ddf974d31fb3fb2`) whose TMN and TMX 60S row (parameters and pattern
slice) was a copy of T's. Nothing else changed.
The file the backend loads is set by `CESM_PARAMS_FILE` in `backend/variable.py`, and its sha1 by
`CESM_PARAMS_SHA1` next to it. Update both together: the sha1 keys the saved custom-injection
outputs, and a unit test checks it matches the file.

## Attribution

Cite Farley et al., *A Climate Intervention Dynamical Emulator (CIDER) for scenario
space exploration*, Geoscientific Model Development 19, 1809–1831 (2026),
[doi:10.5194/gmd-19-1809-2026](https://doi.org/10.5194/gmd-19-1809-2026).
This citation applies to both the original parameters and the September 2026 refit.

## Layout

- `param_AOD_all` (7, 3): AOD impulse parameters `[beta, alpha, gamma]`, one row per injection
  latitude. The rows run 60N → 60S (row 1 = 60N, row 7 = 60S). This is inferred from where
  each slice of `pattern_AOD_all` peaks; the files do not record it.
- `param_<V>_all` (8, 2): semi-infinite-diffusion parameters `[tau, mu]`. Rows 1–7 are the 7
  injection latitudes; row 8 is CO2.
- `pattern_<V>_all` (288, 192, 8): response patterns on the CESM grid (lon, lat), with the same
  row order as `param_<V>_all`.
- `pattern_<V>_base` (288, 192): baseline field.
- `pattern_AOD_all` (288, 192, 7), `pattern_AOD_base` (288, 192).

`<V>` is one of:

| `<V>` | Variable | In `CESM_params.mat` | In `CESM_params_updated.mat` |
|---|---|---|---|
| `T` | 2 m temperature (K) | yes | yes |
| `P` | precipitation | yes | yes |
| `Q` | evaporation | yes | yes |
| `RH` | relative humidity | yes | yes |
| `U10` | 10 m wind speed | yes | yes |
| `TMN` | monthly minimum of daily minimum 2 m temperature (K) | no | yes |
| `TMX` | monthly maximum of daily maximum 2 m temperature (K) | no | yes |

TMN/TMX are the same quantities as the simulator's `tasmin`/`tasmax`.

TMN, T and TMX are emulated independently. On the baselines in the file, the temperatures they
give with SAI keep tasmin ≤ tas ≤ tasmax for 3 Tg/yr per latitude, but not from about 5 Tg/yr
per latitude (35 Tg/yr in total). There the order breaks in ocean cells where TMN and TMX are
close together, mainly off Peru and south of Greenland, by up to 2.7 K at 10 Tg/yr per latitude.
`tests/test_variable_injection.py` records this as an expected failure.

## Differences between `CESM_params_updated.mat` and `CESM_params.mat`

The new file is a refit, not just the old file with TMN/TMX appended.

- Identical: `pattern_{AOD,P,Q,RH,U10}_all`, `pattern_{AOD,P,Q,RH,U10}_base`.
- `param_AOD_all`: every row differs (max abs diff 0.042; up to 5× relative in rows 5–6).
- `param_{P,Q}_all`: only row 8 (CO2) differs.
- `param_T_all`: rows 1–7 differ by < 0.1 % relative; row 8 (CO2) differs substantially.
- `param_{RH,U10}_all`: all rows differ.
- `pattern_T_all`: slices 2–6 differ by < 3e-5 K; slice 8 (CO2) differs by up to 0.79 K.
- `pattern_T_base`: differs by up to 0.60 K.

The CO2 row and slice (row 8) are unused on the custom-injection path, which sets CO2 forcing
to zero.

`backend/old_CESM_params.mat` (2025-02-11) is an older version with AOD, T and P only. Nothing
references it.
