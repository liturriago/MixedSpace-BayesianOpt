# Framework de Optimización Bayesiana para Variables Mixtas con Restricciones de Caja Negra

Este proyecto implementa desde cero (from scratch) un framework de **Optimización Bayesiana (BO)** en Python utilizando únicamente **NumPy** para el álgebra lineal de los modelos de regresión probabilística y **PyTorch** para el pipeline de Deep Learning.

El objetivo del framework es maximizar el coeficiente **Validation Dice Score** de una red neuronal de segmentación semántica (**U-Net**) sobre el dataset **Oxford-IIIT Pet**, bajo una restricción de tiempo de ejecución de caja negra por época ($T(x) \leq T_{\max}$).

---

## 1. Planteamiento del Problema

La sintonización de hiperparámetros se realiza sobre un espacio de búsqueda continuo-discreto mixto de 3 dimensiones ($d = 3$):
1. **Tasa de aprendizaje** ($\log_{10}(\text{lr})$): Variable continua $x_1 \in [-5.0, -1.0]$ que mapea un rango real de $\text{lr} \in [10^{-5}, 10^{-1}]$.
2. **Parámetro $\alpha_{\text{focal}}$**: Variable continua $x_2 \in [0.1, 0.9]$ que controla el balance de clases en la función de costo *Focal Loss*, manteniendo $\gamma = 2.0$ fijo.
3. **Tipo de optimizador (`optimizer_type`)**: Variable categórica indexada en el conjunto `{0: 'Adam', 1: 'RMSprop', 2: 'SGD_Momentum'}`.

### Relajación Continua por Proyección
Para manejar la variable categórica utilizando Procesos Gaussianos continuos:
- El optimizador bayesiano explora el espacio continuo $[0.0, 2.0]$ para $x_3$.
- Al evaluar el pipeline en PyTorch, el valor se proyecta mediante un redondeo estricto:
  $$\hat{x}_3 = \text{int}(\text{clip}(\text{round}(x_3), 0, 2))$$
- La evaluación se registra sobre la coordenada continua original, permitiendo que el GP aprenda la topología de mesetas.

---

## 2. Marco Teórico Matemático

### 2.1 Procesos Gaussianos (GP)
Se ajustan dos modelos GP independientes basados en las observaciones acumuladas:
- $GP_f(x)$ para modelar el Dice Score de validación (función objetivo).
- $GP_T(x)$ para modelar el tiempo de ejecución por época (restricción).

Ambos procesos utilizan el **Kernel Matérn 5/2** con Determinación Automática de Relevancia (ARD):
$$k(x, x') = \sigma_f^2 \left(1 + \sqrt{5}d + \frac{5}{3}d^2\right) \exp(-\sqrt{5}d)$$
donde $d = \sqrt{\sum_{i=1}^3 \frac{(x_i - x_i')^2}{\ell_i^2}}$ es la distancia ponderada.

### 2.2 Estabilidad Numérica mediante Descomposición de Cholesky
Para evitar la inversión directa de la matriz de covarianza ruidosa $K_y = K(X, X) + \sigma_n^2 I$, calculamos su descomposición triangular:
$$K_y = L L^T$$
Las predicciones para un nuevo punto $x_*$ se resuelven eficientemente mediante sustitución hacia adelante y hacia atrás:
- Media predictiva: $\mu(x_*) = k_*^T \alpha$, resolviendo $L L^T \alpha = y$.
- Varianza predictiva: $\sigma^2(x_*) = k(x_*, x_*) - v^T v$, resolviendo $L v = k_*$.

### 2.3 Función de Adquisición con Restricciones
Para guiar la optimización, se utiliza la **Mejora Esperada (Expected Improvement, EI)** modulada por la **Probabilidad de Viabilidad (Probability of Feasibility, PF)**:
$$\alpha_c(x) = EI(x) \times PF(x)$$
Donde:
- $EI(x) = (\mu(x) - f(x^+) - \xi) \Phi(Z) + \sigma(x) \phi(Z)$ con $Z = \frac{\mu(x) - f(x^+) - \xi}{\sigma(x)}$ e $f(x^+)$ siendo el mejor Dice Score factible observado.
- $PF(x) = P(T(x) \leq T_{\max}) = \Phi\left(\frac{T_{\max} - \mu_T(x)}{\sigma_T(x)}\right)$.
- Si no se ha observado ningún punto factible aún, la adquisición se reduce a $\alpha_c(x) = PF(x)$ para priorizar la búsqueda de una región viable.

---

## 3. Estructura del Proyecto

El código está organizado de manera modular en las siguientes rutas:
```bash
MixedSpace-BayesianOpt/
├── src/
│   ├── gp.py          # Implementación del Proceso Gaussiano (Matérn 5/2 + Cholesky)
│   ├── acquisition.py # EI, PF, Adquisición combinada y optimización L-BFGS-B
│   ├── model.py       # Modelo SimpleUNet, Focal Loss y evaluación de época en PyTorch
│   └── data.py        # Dataloaders de Oxford-IIIT Pet con subconjuntos reducidos
├── main.ipynb         # Notebook principal
├── README.md          # Este archivo de documentación
└── plots/
    └── progreso_optimizacion.png # Gráfico de evolución temporal y rendimiento
```

---

## 4. Requisitos e Instalación

Asegúrate de contar con Python 3.8+ y las bibliotecas científicas estándar de Python instaladas:

```bash
pip install numpy torch torchvision scipy matplotlib
```