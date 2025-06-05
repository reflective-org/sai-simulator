import numpy as np

def stretched_sigmoid_x0(x, lam, beta, x0):
    """Vectorised stretched sigmoid (λ, β, x0 all scalars)."""
    z = -(x - x0) / lam    # z ≥ 0 for valid domain
    z = np.maximum(z, 0.0) # clip to domain  z ≥ 0
    return 1.0 - np.exp(-(z ** beta))

def stretched_sigmoid(x, lam, beta):
    """Vectorised stretched sigmoid (λ, β, x0 all scalars)."""
    z = -x/lam    # z ≥ 0 for valid domain
    z = np.maximum(z, 0.0) # clip to domain  z ≥ 0
    return 1.0 - np.exp(-(z ** beta))