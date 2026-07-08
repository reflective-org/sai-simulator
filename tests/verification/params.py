"""Parameter grid for the verification harness.

Defines the SSP scenarios, temperature targets (0.5-2.5 °C in 0.1 steps), and SAI start years
(2035-2090 inclusive) that :func:`tests.verification.generate_test_cases.generate_test_cases`
samples from when building a case list. The Cartesian product covers ~95k cases, so the
random-sample path is the only practical way to drive the harness.
"""
PARAMS = {
    "ssp_scenario": [
        "SSP1-1.9",
        "SSP1-2.6",
        "SSP4-3.4",
        "SSP5-3.4",
        "SSP2-4.5",
        "SSP4-6.0",
        "SSP3-Baseline",
        "SSP5-Baseline",
    ],
    "temp_target": [x / 10 for x in range(5, 26)],  # [0.5, 0.6, ..., 2.4, 2.5]
    "start_year": list(range(2035, 2091)),  # [2035, 2036, ..., 2089, 2090]
}
