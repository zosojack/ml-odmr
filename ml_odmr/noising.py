from numba import njit
import numpy as np

@njit
def add_poly_trend(
    x: np.ndarray,
    s: np.ndarray,
    degree: int = 3,
    scale: float = 0.1
) -> np.ndarray:
    """
    Add polynomial trend to the input data.

    Parameters:
    s (numpy.ndarray): Input data.
    degree (int): Degree of the polynomial trend.
    scale (float): Scale of the trend.

    Returns:
    numpy.ndarray: Data with polynomial trend.
    """
    result = s.astype(np.float64)
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min) if x_max != x_min else np.zeros_like(x, dtype=np.float64)
    
    trend = np.zeros_like(result)
    for d in range(1, degree + 1):
        coeff = np.random.standard_normal()
        trend += scale * coeff * (x_norm ** d)
        
    return result + trend

@njit
def add_outliers(
    s: np.ndarray,
    num_outliers: int = None,
    scale: float = 0.2
) -> np.ndarray:
    """
    Add outliers to the input data.

    Parameters:
    s (numpy.ndarray): Input data.
    num_outliers (int): Number of outliers to add.
    scale (float): Scale of the outliers.

    Returns:
    s: np.ndarray,
    num_outliers: int = None,
    scale: float = 1.0
) -> np.ndarray:
    """
    
    result = s.astype(np.float64)
    if num_outliers is None:
        num_outliers = np.random.randint(6, 20)
        
    max_val = np.max(np.abs(result))
    if max_val == 0:
        max_val = 1.0

    # Determina gli indici casuali unici (compatibile con Numba)
    perm = np.random.permutation(len(result))
    outliers = perm[:num_outliers]
    
    # Genera outlier sensati tra 1.0 e 1.5 volte il max_val
    for idx in outliers:
        outlier_value = max_val * (1.0 + np.random.uniform(0.0, 0.5))
        # Assegna il segno (positivo o negativo) in modo casuale
        sign = 1.0 if np.random.random() > 0.5 else -1.0
        result[idx] = sign * outlier_value
        
    return result

@njit
def add_poisson_noise(
    s: np.ndarray,
    scale: float = 1.0
)-> np.ndarray:
    """
    Add Poissonian noise to the input data. 
    Adds a value drawn from a Gaussian distribution with mean 0 and std dev s[i] to each s[i].

    Parameters:
    s (numpy.ndarray): Input data.
    scale (float): Scale of the noise.

    Returns:
    numpy.ndarray: Noisy data.
    """
    result = s.astype(np.float64) 
    max_val = np.max(np.abs(result))
    
    for i in range(len(result)):
        dev = np.maximum(result[i], max_val*0.05)
        # Passiamo i parametri solo come POSIZIONALI (senza loc=, scale=)
        result[i] += np.random.normal(0.0, dev * scale)
    return result