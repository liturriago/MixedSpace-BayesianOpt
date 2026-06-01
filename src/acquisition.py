import numpy as np
from scipy.special import erf

def norm_cdf(z):
    """
    Función de distribución acumulada (CDF) de la distribución normal estándar.
    """
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

def norm_pdf(z):
    """
    Función de densidad de probabilidad (PDF) de la distribución normal estándar.
    """
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z**2)

def expected_improvement(X, gp_obj, best_y, xi=0.01):
    """
    Calcula la Mejora Esperada (Expected Improvement, EI) para maximización.
    
    Matemática:
        Z = (mu(x) - best_y - xi) / std(x)  si std(x) > 0
        EI(x) = (mu(x) - best_y - xi) * CDF(Z) + std(x) * PDF(Z)
        
    Parámetros:
    -----------
    X : ndarray of shape (M, D)
        Puntos de consulta.
    gp_obj : GaussianProcessRegressor
        GP de la función objetivo ajustado.
    best_y : float
        El mejor valor factible observado del Dice Score hasta el momento.
    xi : float, default=0.01
        Hiperparámetro de exploración para evitar estancamiento en óptimos locales.
        
    Retorna:
    --------
    ei : ndarray of shape (M,)
    """
    mu, std = gp_obj.predict(X, return_std=True)
    
    # Evitar indeterminaciones de división por cero
    std = np.clip(std, 1e-9, None)
    
    Z = (mu - best_y - xi) / std
    ei = (mu - best_y - xi) * norm_cdf(Z) + std * norm_pdf(Z)
    
    # Si la desviación estándar es extremadamente pequeña, la mejora tiende a 0
    ei = np.where(std < 1e-8, 0.0, ei)
    return ei

def probability_of_feasibility(X, gp_constraint, T_max):
    """
    Calcula la Probabilidad de Viabilidad (Probability of Feasibility, PF).
    Determina la probabilidad de que el tiempo de ejecución sea menor o igual a T_max.
    
    Matemática:
        Z_c = (T_max - mu_T(x)) / std_T(x)
        PF(x) = P(T(x) <= T_max) = CDF(Z_c)
        
    Parámetros:
    -----------
    X : ndarray of shape (M, D)
    gp_constraint : GaussianProcessRegressor
        GP ajustado para modelar el tiempo de ejecución.
    T_max : float
        Umbral máximo permitido para el tiempo de ejecución por época.
        
    Retorna:
    --------
    pf : ndarray of shape (M,)
    """
    mu, std = gp_constraint.predict(X, return_std=True)
    
    std = np.clip(std, 1e-9, None)
    Z_c = (T_max - mu) / std
    pf = norm_cdf(Z_c)
    
    # Si no hay incertidumbre en el punto, verificar directamente si la media cumple
    pf = np.where(std < 1e-8, (mu <= T_max).astype(float), pf)
    return pf

def constrained_acquisition(X, gp_obj, gp_constraint, best_y, T_max, xi=0.01):
    """
    Calcula la adquisición combinada para optimización bayesiana constreñida.
    
    Matemática:
        alpha_c(x) = EI(x) * PF(x)   si se ha observado al menos un punto factible.
        alpha_c(x) = PF(x)           si no se ha observado ningún punto factible aún
                                     (el objetivo es encontrar viabilidad primero).
                                     
    Parámetros:
    -----------
    X : ndarray of shape (M, D)
    gp_obj : GaussianProcessRegressor
    gp_constraint : GaussianProcessRegressor
    best_y : float o None
    T_max : float
    xi : float
    """
    X = np.atleast_2d(X)
    pf = probability_of_feasibility(X, gp_constraint, T_max)
    
    if best_y is None or best_y == -np.inf:
        return pf
        
    ei = expected_improvement(X, gp_obj, best_y, xi=xi)
    return ei * pf

def optimize_acquisition(gp_obj, gp_constraint, best_y, T_max, bounds, n_restarts=10, xi=0.01):
    """
    Optimiza la función de adquisición combinada en el espacio de búsqueda relajado.
    Utiliza una estrategia híbrida:
      1. Evaluación densa de una cuadrícula aleatoria uniforme de 1500 puntos.
      2. Selección de los mejores n_restarts puntos como semillas.
      3. Ejecución del algoritmo cuasi-Newton local L-BFGS-B acotado desde esas semillas.
      
    Parámetros:
    -----------
    gp_obj : GaussianProcessRegressor
    gp_constraint : GaussianProcessRegressor
    best_y : float o None
    T_max : float
    bounds : list of tuples of length 3
        Límites para las variables en el espacio de búsqueda continuo.
        [(log10_lr_min, log10_lr_max), (alpha_min, alpha_max), (opt_min, opt_max)]
    n_restarts : int
        Número de búsquedas locales a iniciar.
    xi : float
        Parámetro de exploración.
        
    Retorna:
    --------
    best_x : ndarray of shape (D,)
        El punto óptimo sugerido para la siguiente iteración.
    best_val : float
        El valor de la función de adquisición en el óptimo.
    """
    from scipy.optimize import minimize
    
    # Definir la función objetivo a MINIMIZAR (el negativo de la adquisición)
    def target_func(x):
        val = constrained_acquisition(x.reshape(1, -1), gp_obj, gp_constraint, best_y, T_max, xi=xi)
        return -val[0]
        
    D = len(bounds)
    
    # 1. Búsqueda aleatoria inicial densa
    pts_rand = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(1500, D)
    )
    
    acq_vals = constrained_acquisition(pts_rand, gp_obj, gp_constraint, best_y, T_max, xi=xi)
    
    # 2. Selección de semillas
    best_indices = np.argsort(acq_vals)[::-1][:n_restarts]
    x_seeds = pts_rand[best_indices]
    
    best_x = None
    best_acq_val = -np.inf
    
    # 3. Optimización local L-BFGS-B
    for x_start in x_seeds:
        res = minimize(
            target_func,
            x0=x_start,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        current_acq_val = -res.fun
        if current_acq_val > best_acq_val:
            best_acq_val = current_acq_val
            best_x = res.x
            
    return best_x, best_acq_val
