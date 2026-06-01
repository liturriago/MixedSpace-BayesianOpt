import numpy as np

class GaussianProcessRegressor:
    """
    Clase que implementa un Regresor de Proceso Gaussiano (GPR) desde cero usando únicamente NumPy.
    Optimizado para estabilidad numérica con descomposición de Cholesky y soporte para
    Determinación Automática de Relevancia (ARD) con kernel Matérn 5/2.
    """
    
    def __init__(self, length_scales, signal_variance=1.0, noise_level=1e-4):
        """
        Inicializa el regresor de Proceso Gaussiano.
        
        Parámetros:
        -----------
        length_scales : array-like of shape (D,)
            Longitudes de escala de longitud (lengthscales) para cada dimensión de entrada (ARD).
        signal_variance : float
            Varianza a priori de la función o factor de escala de amplitud (\sigma_f^2).
        noise_level : float
            Nivel de ruido de observación \sigma_n^2 (nugget) añadido a la diagonal de la covarianza.
        """
        self.length_scales = np.array(length_scales, dtype=float)
        self.signal_variance = float(signal_variance)
        self.noise_level = float(noise_level)
        self.X_train = None
        self.y_train = None
        self.L = None        # Matriz triangular inferior de Cholesky
        self.alpha = None    # Vector de coeficientes del posterior (K_y^-1 @ y_train)
        self.y_mean = 0.0
        self.y_std = 1.0

    def _matern_52_kernel(self, X1, X2):
        """
        Calcula la matriz de covarianza Matérn 5/2 con ARD entre X1 y X2.
        
        Matemática:
            d(x, x') = sqrt( sum_{i=1}^D ((x_i - x'_i) / \ell_i)^2 )
            k(x, x') = \sigma_f^2 * (1 + \sqrt{5}*d + (5/3)*d^2) * exp(-\sqrt{5}*d)
            
        Parámetros:
        -----------
        X1 : ndarray of shape (N, D)
        X2 : ndarray of shape (M, D)
        
        Retorna:
        --------
        K : ndarray of shape (N, M)
        """
        # Escalar coordenadas por los lengthscales respectivos (implementando ARD)
        X1_scaled = X1 / self.length_scales
        X2_scaled = X2 / self.length_scales
        
        # Distancia euclidiana cuadrada eficiente usando álgebra lineal vectorial:
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 <a, b>
        sq_dist = (np.sum(X1_scaled**2, axis=1)[:, np.newaxis] +
                   np.sum(X2_scaled**2, axis=1)[np.newaxis, :] -
                   2.0 * np.dot(X1_scaled, X2_scaled.T))
        
        # Eliminar posibles valores negativos debido a precisión de punto flotante
        sq_dist = np.clip(sq_dist, 0.0, None)
        d = np.sqrt(sq_dist)
        
        # Evaluación de la ecuación Matérn 5/2
        sqrt5_d = np.sqrt(5.0) * d
        term = 1.0 + sqrt5_d + (5.0 / 3.0) * sq_dist
        K = self.signal_variance * term * np.exp(-sqrt5_d)
        return K

    def fit(self, X, y):
        """
        Ajusta el Proceso Gaussiano a las observaciones recopiladas.
        Calcula la descomposición de Cholesky de la covarianza de entrenamiento.
        
        Parámetros:
        -----------
        X : ndarray of shape (N, D)
            Puntos de entrada de entrenamiento.
        y : ndarray of shape (N,) o (N, 1)
            Valores observados de la función objetivo o restricción.
        """
        self.X_train = np.array(X, dtype=float)
        y = np.array(y, dtype=float).ravel()
        
        # Estandarización de y para garantizar estabilidad numérica y escalado de priors
        if len(y) > 1:
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)
            if self.y_std < 1e-8:
                self.y_std = 1.0
        else:
            self.y_mean = 0.0
            self.y_std = 1.0
            
        self.y_train = (y - self.y_mean) / self.y_std
        
        # Calcular matriz de covarianza de entrenamiento
        K = self._matern_52_kernel(self.X_train, self.X_train)
        
        # Escalar el nivel de ruido de observación al espacio estandarizado
        scaled_noise = self.noise_level / (self.y_std ** 2)
        
        # K_y = K(X, X) + \sigma_n^2 * I
        K_y = K + (scaled_noise + 1e-8) * np.eye(len(self.X_train))
        
        # Descomposición de Cholesky: K_y = L @ L.T
        try:
            self.L = np.linalg.cholesky(K_y)
        except np.linalg.LinAlgError:
            # En caso de mal condicionamiento numérico severo, aumentar el factor de regularización (jitter)
            K_y += 1e-6 * np.eye(len(self.X_train))
            self.L = np.linalg.cholesky(K_y)
            
        # Resolver para alpha: K_y @ alpha = y_train => L @ L.T @ alpha = y_train
        # 1. Resolver L @ beta = y_train  (Sustitución hacia adelante)
        beta = np.linalg.solve(self.L, self.y_train)
        # 2. Resolver L.T @ alpha = beta  (Sustitución hacia atrás)
        self.alpha = np.linalg.solve(self.L.T, beta)

    def predict(self, X_test, return_std=True):
        """
        Predice la media y desviación estándar de la distribución a posteriori en X_test.
        
        Parámetros:
        -----------
        X_test : ndarray of shape (M, D)
            Puntos en los que realizar la predicción.
        return_std : bool, default=True
            Si es True, se calcula y retorna la desviación estándar predictiva.
            
        Retorna:
        --------
        mu : ndarray of shape (M,)
            Media predictiva en el espacio original de observaciones.
        std : ndarray of shape (M,) (opcional)
            Desviación estándar predictiva en el espacio original de observaciones.
        """
        X_test = np.array(X_test, dtype=float)
        
        # Si el modelo no ha sido entrenado aún, retornar predicciones basadas puramente en el prior
        if self.X_train is None or len(self.X_train) == 0:
            mu = np.zeros(len(X_test)) * self.y_std + self.y_mean
            if return_std:
                std = np.sqrt(self.signal_variance) * np.ones(len(X_test)) * self.y_std
                return mu, std
            return mu
            
        # Covarianza cruzada entre datos de entrenamiento y de prueba: K_trans = K(X_train, X_test) de forma (N, M)
        K_trans = self._matern_52_kernel(self.X_train, X_test)
        
        # Media predictiva estandarizada: mu_norm = K_trans.T @ alpha
        mu_norm = np.dot(K_trans.T, self.alpha)
        # Des-estandarizar para volver a la escala original del objetivo
        mu = mu_norm * self.y_std + self.y_mean
        
        if return_std:
            # Resolver L @ v = K_trans para obtener v = L^-1 @ K_trans
            v = np.linalg.solve(self.L, K_trans)
            
            # Varianza predictiva diagonal estandarizada: var_norm = K(x_*, x_*) - v.T @ v
            # Para Matérn 5/2, el prior de la varianza en la diagonal es constante e igual a signal_variance
            K_test_diag = self.signal_variance * np.ones(len(X_test))
            var_norm = K_test_diag - np.sum(v**2, axis=0)
            
            # Asegurar valores no negativos
            var_norm = np.clip(var_norm, 0.0, None)
            
            # Des-estandarizar desviación estándar
            std = np.sqrt(var_norm) * self.y_std
            return mu, std
            
        return mu
