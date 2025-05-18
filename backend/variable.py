import scipy.io
import numpy as np
import xarray as xr
from scipy.special import erfc

from .constants import *


def impulse_firstOdiff_wGammaInj(beta, alpha, gamma, q, t):
    # Impulse response for AOD
    ydata = beta*np.exp(-(alpha+gamma*q)*t)
    return ydata


def AOD_from_injection(param_AOD, injection):
    # Calculates AOD from SO2 injection at one latitude
    # param_AOD_all: AOD parameters, for all injection latitudes
    # injection: single latitude injection timeseries (monthly)
    beta, alpha, gamma = param_AOD
    injection = np.array(injection)
    AOD_emulated = np.zeros(len(injection))

    if np.mean(injection) == 0:
        return AOD_emulated

    for k in range(len(AOD_emulated)):
        for j in range(k + 1):
            AOD_emulated[k] += (
                impulse_firstOdiff_wGammaInj(beta, alpha, gamma, injection[k - j], j +1)
                * injection[k - j]
            )

    return AOD_emulated


def AOD_from_injection_vectorized(param_AOD, injection):
    """
    Vectorized version of the AOD_from_injection routine.
    
    Parameters
    ----------
    param_AOD : array_like
    injection : array_like
        1D array of SO2 injection values (e.g., monthly).
    
    Returns
    -------
    AOD_emulated : ndarray
        The emulated AOD (same length as injection).
    """
    beta, alpha, gamma = param_AOD
    injection = np.asarray(injection, dtype=float)
    n = len(injection)

    # If injection is all zeros, quickly return zeros
    if n == 0 or np.allclose(injection, 0.0):
        return np.zeros(n)

    # Create a 2D index grid:
    #   K is the "current time" index (rows),
    #   J is the "lag" index (columns).
    # We'll sum over J for each K.
    K = np.arange(n).reshape(-1, 1)  # shape (n, 1)
    J = np.arange(n).reshape(1, -1)  # shape (1, n)

    # We only want to include terms where J <= K (i.e., lag doesn't exceed current time).
    valid_mask = (J <= K)

    # K - J indicates which "past" injection time is contributing to time K.
    # For invalid entries (where K - J < 0), we will set them to 0 contribution.
    KminusJ = K - J  # shape (n, n)

    # Extract injection[K - J] where valid, else 0.
    inj_2d = np.where(valid_mask, injection[KminusJ], 0.0)

    # Time offset is (j + 1) in the original code, so let's define that as T_2d = J + 1
    T_2d = (J + 1).astype(float)

    # The impulse response factor (without the extra multiplication by injection):
    #   impulse = beta * exp(-(alpha + gamma * q) * t)
    # here q = injection[KminusJ] and t = (j+1)
    impulse_2d = np.where(
        valid_mask,
        beta * np.exp(- (alpha + gamma * inj_2d) * T_2d),
        0.0
    )

    # Finally, multiply by injection[KminusJ] again as in the original loop:
    # AOD_emulated[k] += impulse(...) * injection[k - j]
    # We already stored injection[k - j] in inj_2d, so just multiply:
    AOD_2d = inj_2d * impulse_2d

    # Sum across each row (k) to get AOD at time k
    AOD_emulated = AOD_2d.sum(axis=1)

    return AOD_emulated


def impulse_semiInfDiff(t, impulse_p):
    # Impulse response for climate
    if t == 0:
        t = np.finfo(float).tiny
    mu = impulse_p['mu']
    tau = impulse_p['tau']
    h = mu * (1 / np.sqrt(np.pi * t / tau) - np.exp(t / tau) * erfc(np.sqrt(t / tau)))
    if np.isinf(np.exp(t / tau) * erfc(np.sqrt(t / tau))) or np.isnan(np.exp(t / tau) * erfc(np.sqrt(t / tau))):
        h = mu * (1 / np.sqrt(np.pi * t / tau) - 2 / np.sqrt(np.pi) * (np.sqrt(t / tau) + np.sqrt(np.sqrt(t / tau)**2 + 2))**(-1))
    return h


def response_from_1_forcing(params, forcing):
    # Emulated response for climate from a single forcing
    impulse_p_SAI = {'tau': params[0], 'mu': params[1]}
    emulated_response = np.zeros(len(forcing))

    for k in range(len(emulated_response)):
        for j in range(k + 1):
            emulated_response[k] += impulse_semiInfDiff(j+1, impulse_p_SAI) * forcing[k - j]

    return emulated_response


def response_from_1_forcing_vectorized(params, forcing):
    """
    Vectorized version of 'response_from_1_forcing' with 'impulse_semiInfDiff'.
    Parameters
    ----------
    params : tuple or list
        [tau, mu] corresponding to 'impulse_p_SAI':
            tau -> impulse_p_SAI['tau']
            mu  -> impulse_p_SAI['mu']
    forcing : array_like
        1D array representing the forcing time series (e.g., monthly).
    
    Returns
    -------
    emulated_response : ndarray
        The emulated climate response (same length as 'forcing').
    """
    tau, mu = params
    forcing = np.asarray(forcing, dtype=float)
    n = len(forcing)

    # If no forcing or all-zero forcing, quickly return zeros
    if n == 0 or np.allclose(forcing, 0.0):
        return np.zeros(n)

    # ----------------------------------------------------------------------
    # 1) Construct index grids: 
    #    - K = "current time" index, shape (n,1)
    #    - J = "lag" index, shape (1,n)
    # ----------------------------------------------------------------------
    K = np.arange(n).reshape(-1, 1)  # shape (n,1) for broadcasting
    J = np.arange(n).reshape(1, -1)  # shape (1,n)

    # Only terms where J <= K (meaning the lag doesn't exceed the current time)
    valid_mask = (J <= K)

    # (K - J) tells us which past forcing element to use at time K
    K_minus_J = K - J

    # Grab the correct forcing value. Zero it out where invalid.
    forcing_2d = np.where(valid_mask, forcing[K_minus_J], 0.0)

    # ----------------------------------------------------------------------
    # 2) Compute the impulse response 'impulse_semiInfDiff' in vector form.
    #
    #    For each pair (k,j), time t = (j+1).
    # ----------------------------------------------------------------------
    T_2d = (J + 1).astype(float)  # shape (1,n), then broadcasted to (n,n)

    # In principle, t never equals zero (since j+1 >= 1). 
    # We replicate the original check just in case.
    tiny = np.finfo(float).tiny
    T_2d_fixed = np.where(T_2d == 0.0, tiny, T_2d)

    # Calculate the standard impulse response:
    #    h = mu * [ 1 / sqrt(pi * t/tau) - exp(t/tau)*erfc(sqrt(t/tau)) ]
    sqrt_term = np.sqrt(T_2d_fixed / tau)
    exp_term  = np.exp(T_2d_fixed / tau)
    partial_val = exp_term * erfc(sqrt_term)

    # Main expression
    h_all = mu * (
        1.0 / np.sqrt(np.pi * T_2d_fixed / tau) 
        - partial_val
    )

    # Fallback for cases where np.exp(...) * erfc(...) is inf or nan
    fallback = mu * (
        1.0 / np.sqrt(np.pi * T_2d_fixed / tau) 
        - 2.0 / np.sqrt(np.pi) 
          * (sqrt_term + np.sqrt(sqrt_term**2 + 2.0))**(-1.0)
    )

    mask_bad = np.isinf(partial_val) | np.isnan(partial_val)
    h_all[mask_bad] = fallback[mask_bad]

    # ----------------------------------------------------------------------
    # 3) Multiply each impulse value by the associated forcing and sum
    #    over the "lag" dimension j for each time k.
    # ----------------------------------------------------------------------
    response_2d = forcing_2d * h_all
    emulated_response = response_2d.sum(axis=1)

    return emulated_response


def pattern_scale(mean_response, pattern_to_scale):
    # Scale pattern_to_scale by mean_response
    pattern_to_scale = pattern_to_scale[:,:,np.newaxis]
    pattern = pattern_to_scale * mean_response.reshape(1, 1, -1)
    return pattern


def pattern_from_1_injection(injection, param_AOD, param_climate, pattern_to_scale):
    # Emulate response pattern from 1 latitude of SAI injection
    # injection: single latitude injection timeseries (monthly)
    # param_AOD: inj->AOD parameters
    # param_climate: AOD->climate parameters
    # pattern_to_scale: climate pattern from that latitude

    # Emulate the AOD from the injection
    AOD_emulated = AOD_from_injection_vectorized(param_AOD, injection)

    # Compute the emulated response from the AOD
    emulated_response = response_from_1_forcing_vectorized(param_climate, AOD_emulated)

    # Scale the pattern with the emulated response
    response_pattern = pattern_scale(emulated_response, pattern_to_scale)
    return response_pattern


def pattern_from_all_injections(all_injection, all_param_AOD, all_param_climate, all_pattern_to_scale):
    # Emulate response pattern from all latitudes of SAI injection
    # all_injection: all injection timeseries (monthly), formatted as a matrix
    # all_param_AOD: AOD parameters, for all injection latitudes
    # all_param_climate: climate parameters, for all injection latitudes
    # all_pattern_to_scale: all climate patterns, formatted as a matrix

    injection_length, injection_count = all_injection.shape
    pattern_dim1, pattern_dim2, _ = all_pattern_to_scale.shape

    # Initialize the total pattern array
    total_pattern = np.zeros((pattern_dim1, pattern_dim2, injection_length))

    for i in range(injection_count):
        injection = all_injection[:, i]
        if np.mean(injection) != 0:
            param_AOD = all_param_AOD[i, :]
            param_climate = all_param_climate[i, :]
            pattern_to_scale = all_pattern_to_scale[:, :, i]

            # Compute the response pattern for the current injection
            response_pattern = pattern_from_1_injection(
                injection, param_AOD, param_climate, pattern_to_scale
            )

            # Add the response pattern to the total
            total_pattern += response_pattern

    return total_pattern


def pattern_from_all_injections_and_CO2(all_injection_and_CO2, all_param_AOD, all_param_climate, all_pattern_to_scale):
    # Emulate response pattern from all latitudes of SAI injection and CO2
    # all_injection_and_CO2: all injection timeseries and CO2 forcing (monthly), forcing as last column
    # all_param_AOD: AOD parameters, for all injection latitudes
    # all_param_climate: climate parameters, for all injection latitudes and CO2 (which is last)
    # all_pattern_to_scale: all climate patterns from injections and  and CO2 (which is last)

    total_pattern_inj = pattern_from_all_injections(
        all_injection_and_CO2[:, :-1], all_param_AOD, all_param_climate[:-1, :], all_pattern_to_scale
    )
    pattern_CO2 = pattern_scale(
        response_from_1_forcing_vectorized(all_param_climate[-1, :], all_injection_and_CO2[:, -1]),
        all_pattern_to_scale[:, :, -1]
    )
    total_pattern = total_pattern_inj + pattern_CO2

    return total_pattern


def get_variable_regional_delta(var, data_dir, cache_dir, variable_injection):
    # Assumes variable injection is the amount of Tg per latitude per year, of shape (7, 65)
    # Repeat the variable injection to an amount per month so it becomes (7, 65*12)
    # And spread it over each year by dividing by 12
    variable_injection = np.repeat(variable_injection, 12, axis=1) / 12
    # Assume no CO2 forcing to get the delta
    co2_forcing = np.zeros((variable_injection.shape[1], 1))

    all_injection_and_CO2 = np.concatenate([variable_injection.T, co2_forcing], axis=1)

    mat = scipy.io.loadmat(data_dir / 'CESM_params.mat')
    param_AOD_all = np.array(mat['param_AOD_all'])

    if var == "tas":
        param_all = np.array(mat['param_T_all'])
        pattern_all = np.array(mat['pattern_T_all'])
    elif var == "pr":
        param_all = np.array(mat['param_P_all'])
        pattern_all = np.array(mat['pattern_P_all'])
    elif var == "e":
        param_all = np.array(mat['param_Q_all'])
        pattern_all = np.array(mat['pattern_Q_all'])
    else:
        raise ValueError(f"{var} not supported for variable injection")

    regional_delta = pattern_from_all_injections_and_CO2(all_injection_and_CO2, param_AOD_all, param_all, pattern_all)
    # regional_delta is of shape (lon, lat, num_months)
    # Convert into annual values (window of 12 months)
    regional_delta_monthly = regional_delta.reshape(regional_delta.shape[0], regional_delta.shape[1], -1, 12)
    regional_delta_yearly = np.mean(regional_delta_monthly, axis=-1)

    # Convert the regional delta to an xarray DataArray
    correct_lat = np.load(data_dir / "correct_lat.npy")
    regional_delta = xr.DataArray(
        regional_delta_yearly.transpose(1, 0, 2),
        dims=('lat', 'lon', 'time'),
        coords={'lat': correct_lat,
                'lon': np.linspace(0, 360, NUM_LON, endpoint=False),
                'time': np.arange(2035, 2101)})

    # Transpose to (time, lat, lon)
    regional_delta = regional_delta.transpose('time', 'lat', 'lon')

    return regional_delta
