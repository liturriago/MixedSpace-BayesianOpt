# %% [markdown]
# # Optimización Bayesiana para Variables Mixtas con Restricciones de Caja Negra
#
# ## 1. Planteamiento del Problema de Optimización
#
# En el entrenamiento de modelos de Deep Learning complejos, como la arquitectura **U-Net** para la segmentación semántica de imágenes, el rendimiento del modelo depende críticamente de la configuración de sus hiperparámetros. 
#
# Definimos nuestro objetivo como el hallazgo de la configuración óptima de hiperparámetros $x^*$ que maximice una métrica de rendimiento de caja negra $f(x)$ (en nuestro caso, el **Validation Dice Score**), sujeta a restricciones físicas o de hardware de caja negra $g(x) \leq 0$ (en nuestro caso, el **Tiempo de ejecución por época** en segundos).
#
# El problema de optimización restringida se formula matemáticamente como:
#
# $$\max_{x \in \mathcal{X}} f(x)$$
# $$\text{sujeto a } g(x) \leq 0$$
#
# Donde $\mathcal{X}$ representa un espacio de búsqueda continuo-discreto mixto de 3 dimensiones ($d = 3$).
#
# ### 1.1 Definición del Espacio de Búsqueda Mixed-Domain ($\mathcal{X}$)
#
# El vector de diseño $x = [x_1, x_2, x_3]^T \in \mathcal{X}$ está compuesto por los siguientes tres hiperparámetros:
#
# 1. **$\log_{10}(\text{learning\_rate})$ ($x_1 \in [-5.0, -1.0]$):** Variable continua que mapea el tamaño del paso del optimizador en escala logarítmica para mejorar la homogeneidad del espacio respecto al Proceso Gaussiano. Corresponde a un rango real de $\text{lr} \in [10^{-5}, 10^{-1}]$.
# 2. **$\alpha_{\text{focal}}$ ($x_2 \in [0.1, 0.9]$):** Variable continua que controla el factor de balance de clases en la *Focal Loss* (mitigando el desbalance entre los píxeles del fondo, la silueta de la mascota y sus contornos), manteniendo el parámetro de escala fijo en su valor estándar $\gamma = 2.0$.
# 3. **$\text{optimizer\_type}$ ($x_3 \in [0.0, 2.0]$):** Variable categórica indexada que define el algoritmo de optimización estructural en PyTorch:
#    * $0 \rightarrow \text{'Adam'}$
#    * $1 \rightarrow \text{'RMSprop'}$
#    * $2 \rightarrow \text{'SGD\_Momentum'}$
#
# ### 1.2 El Enfoque de Caja Negra y Relajación Continua
#
# Las funciones $f(x)$ (Dice Score) y $g(x) = T(x) - T_{\max}$ (donde $T(x)$ es el tiempo medido por época y $T_{\max}$ es el umbral límite) carecen de una forma analítica cerrada, no proveen gradientes directos respecto a $x$ y su evaluación requiere el entrenamiento costoso de la red neuronal en hardware.
#
# Para resolver el problema sobre este dominio mixto utilizando el álgebra lineal continua de los Procesos Gaussianos, implementamos una **Relajación Continua por Proyección**:
#
# 1. El optimizador bayesiano explora e interactúa con el espacio tratándolo como puramente continuo dentro de los límites reales $\mathcal{X} = [-5, -1] \times [0.1, 0.9] \times [0, 2]$.
# 2. En la interfaz con el pipeline de PyTorch, el valor continuo sugerido para la tercera dimensión ($x_3$) se proyecta mediante un redondeo estricto al entero más cercano:
#    $$\hat{x}_3 = \text{int}(\text{clip}(\text{round}(x_3), 0, 2))$$
# 3. El índice discreto resultante se mapea a su respectiva categoría para instanciar el optimizador en PyTorch. La evaluación real del pipeline devuelve el Dice Score y el tiempo, los cuales se registran de vuelta en la coordenada continua original $x_3$, permitiendo al modelo aprender la topología del espacio en forma de mesetas estables.

# %% [markdown]
# ## 2. Marco Teórico Matemático
#
# ### 2.1 Proceso Gaussiano: Prior, Verosimilitud y Posterior
# Un Proceso Gaussiano (GP) es una colección de variables aleatorias, tales que cualquier subconjunto finito de ellas tiene una distribución conjunta gaussiana. Se define completamente por su función de media $m(x)$ y su función de covarianza (o kernel) $k(x, x')$:
# $$f(x) \sim \mathcal{GP}\left(m(x), k(x, x')\right)$$
#
# Por simplicidad computacional y siguiendo el estándar riguroso en optimización de hiperparámetros, asumimos una función de media a priori idénticamente nula, es decir, $m(x) = 0$.
#
# Suponiendo observaciones ruidosas $y = f(x) + \epsilon$ con ruido Gaussiano i.i.d. $\epsilon \sim \mathcal{N}(0, \sigma_n^2)$, la **verosimilitud (likelihood)** de las observaciones es:
# $$y | X, f \sim \mathcal{N}(f(X), \sigma_n^2 I)$$
#
# Al combinar el Prior con la Verosimilitud mediante el **Condicionamiento Gaussiano (Gaussian Conditioning)**, podemos derivar analíticamente la distribución a **posterior** para nuevos puntos de prueba $X_*$ (rebanando la distribución conjunta mediante los datos fijos observados):
# $$f_* | X, y, X_* \sim \mathcal{N}(\mu(X_*), \Sigma(X_*))$$
#
# Cuyos momentos estadísticos (media y covarianza posterior) se definen, bajo nuestra suposición de media cero, mediante álgebra de bloques como:
# $$\mu(X_*) = K(X_*, X) \left[K(X, X) + \sigma_n^2 I\right]^{-1} y$$
# $$\Sigma(X_*) = K(X_*, X_*) - K(X_*, X) \left[K(X, X) + \sigma_n^2 I\right]^{-1} K(X, X_*)$$
#
# ### 2.2 Estabilidad Numérica mediante Descomposición de Cholesky
# Para evitar la inestabilidad de la inversión directa de la matriz de covarianza ruidosa $K_y = K(X, X) + \sigma_n^2 I$ (la cual es simétrica y definida positiva gracias al kernel y al ruido de observación), realizamos la descomposición de Cholesky:
# $$K_y = L L^T$$
# donde $L$ es una matriz triangular inferior única. Así, resolvemos los sistemas lineales de manera eficiente y estable en NumPy mediante sustitución hacia adelante y hacia atrás:
# 1. Definimos $\alpha = K_y^{-1} y$. Lo resolvemos computando $L \beta = y$ y luego $L^T \alpha = \beta$. La media se evalúa como: $\mu(x_*) = k_*^T \alpha$.
# 2. Definimos $v = L^{-1} k_*$. Lo calculamos resolviendo $L v = k_*$. La varianza posterior se evalúa como: $\sigma^2(x_*) = k(x_*, x_*) - v^T v$.
#
# ### 2.3 Integral de la Mejora Esperada (Expected Improvement, EI)
# Para guiar la búsqueda hacia el máximo global de la función objetivo $f(x)$ (Validation Dice Score), calculamos la mejora sobre el mejor valor factible observado hasta el momento, al que definimos como $f(x^+) = \max_{i: g(x_i) \leq 0} f(x_i)$:
# $$I(x) = \max(0, f(x) - f(x^+))$$
#
# *Nota: En caso de que no se haya descubierto ningún punto factible en la fase de diseño inicial, se adopta un valor basal heurístico $f(x^+) = 0$ para forzar la exploración hacia zonas válidas y evitar que la integral colapse.*
#
# Tomando la esperanza matemática bajo el posterior Gaussiano $\mathcal{N}(\mu(x), \sigma^2(x))$, obtenemos la integral analítica de EI:
# $$EI(x) = \mathbb{E}[I(x)] = \int_{f(x^+)}^{\infty} (f - f(x^+)) \mathcal{N}(f; \mu(x), \sigma^2(x)) df$$
# $$EI(x) = (\mu(x) - f(x^+) - \xi) \Phi(Z) + \sigma(x) \phi(Z)$$
# donde $Z = \frac{\mu(x) - f(x^+) - \xi}{\sigma(x)}$, $\Phi(Z)$ es la CDF normal estándar, $\phi(Z)$ es la PDF normal estándar y $\xi$ es el factor de exploración.
#
# ### 2.4 Restricciones de Caja Negra y Probabilidad de Viabilidad (PF)
# Si el tiempo de ejecución por época $T(x)$ está acotado por $T_{max}$, modelamos $T(x)$ usando un segundo GP independiente. Bajo la hipótesis de independencia estadística entre el proceso del rendimiento (Dice Score) y el del tiempo de ejecución, la teoría de la utilidad condicional dicta que la función de adquisición restringida final se construye multiplicando la mejora esperada por la **Probabilidad de Viabilidad (Probability of Feasibility)**:
# $$PF(x) = P(T(x) \leq T_{max}) = \Phi\left(\frac{T_{max} - \mu_T(x)}{\sigma_T(x)}\right)$$
#
# Al multiplicar ambas métricas, la utilidad colapsará a cero en regiones donde el modelo estime, con alta probabilidad, que se violará la restricción de hardware:
# $$\alpha_c(x) = EI(x) \times PF(x)$$

# %%
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

# Importar los módulos desarrollados en la carpeta src/
from src.gp import GaussianProcessRegressor
from src.acquisition import constrained_acquisition, optimize_acquisition
from src.data import get_dataloaders
from src.model import evaluate_pipeline

# Fijar semilla para reproducibilidad general
np.random.seed(42)
torch.manual_seed(42)

# %% [markdown]
# ## 3. Configuración del Espacio de Búsqueda y Parámetros
# Definimos los límites para los 3 hiperparámetros:
# - `learning_rate` ($x_1$): Optimizado en escala logarítmica $\log_{10}(\text{lr}) \in [-5.0, -1.0]$ para mejorar el comportamiento del GP.
# - `alpha_focal` ($x_2$): Continuo en $[0.1, 0.9]$.
# - `optimizer_type` ($x_3$): Categórico relajado a continuo en $[0.0, 2.0]$.
#
# Además, configuramos la restricción de tiempo $T_{max}$ por época.

# %%
# Límites en el espacio continuo del GP:
# Dim 0: log10(learning_rate) -> [-5, -1] que corresponde a lr en [1e-5, 1e-1]
# Dim 1: alpha_focal -> [0.1, 0.9]
# Dim 2: optimizer_type -> [0.0, 2.0] (0: Adam, 1: RMSprop, 2: SGD_Momentum)
bounds = [(-5.0, -1.0), (0.1, 0.9), (0.0, 2.0)]

# Umbral máximo permitido de tiempo de ejecución (en segundos)
T_max = 6.0  # Ajustado de forma desafiante para el subconjunto CPU

# Configurar hiperparámetros de los GPs
# Longitudes de escala iniciales para ARD
length_scales = np.array([1.5, 0.4, 0.8]) 

# Inicializar GPs
gp_objective = GaussianProcessRegressor(
    length_scales=length_scales,
    signal_variance=1.0,
    noise_level=1e-4  # Ruido especificado por el usuario
)

gp_constraint = GaussianProcessRegressor(
    length_scales=length_scales,
    signal_variance=2.0,
    noise_level=0.1  # Nivel de ruido sintonizado proporcionalmente para mediciones de tiempo
)

# %% [markdown]
# ## 4. Preparación de Datos
# Cargamos los DataLoaders optimizados para el dataset Oxford-IIIT Pet.
# Usamos tamaños de subconjunto reducidos para asegurar ejecuciones rápidas compatibles con CPU.

# %%
print("Descargando y preparando el dataset Oxford-IIIT Pet...")
train_loader, val_loader = get_dataloaders(
    data_dir='./data',
    batch_size=16,
    train_subset_size=320,
    val_subset_size=160
)
print("DataLoaders listos.")

# %% [markdown]
# ## 5. Diseño Inicial (Initial Experimental Design)
# Generamos y evaluamos una serie de puntos iniciales aleatorios (diseño experimental) antes de iniciar el ciclo bayesiano. Esto provee datos de partida indispensables para entrenar el GP a priori.

# %%
# Generar 5 puntos iniciales distribuidos aleatoriamente en el espacio
n_init_points = 5
X_init = np.random.uniform(
    low=[b[0] for b in bounds],
    high=[b[1] for b in bounds],
    size=(n_init_points, 3)
)

X_observed = []
y_dice_observed = []
t_time_observed = []

print("=== Iniciando Diseño Experimental Inicial ===")
for i, x in enumerate(X_init):
    # Desempacar hiperparámetros
    log10_lr, alpha, opt_val = x
    lr = 10**log10_lr
    
    # Mapear categórico mediante redondeo indexado
    opt_idx = int(np.clip(np.round(opt_val), 0, 2))
    opt_names = {0: 'Adam', 1: 'RMSprop', 2: 'SGD_Momentum'}
    opt_name = opt_names[opt_idx]
    
    print(f"\nEvaluando punto inicial {i+1}/{n_init_points}:")
    print(f" -> lr: {lr:.6f} | alpha_focal: {alpha:.3f} | optimizer: {opt_name} (val: {opt_val:.3f})")
    
    # Evaluar red neuronal
    dice, run_time = evaluate_pipeline(
        learning_rate=lr,
        alpha_focal=alpha,
        optimizer_type=opt_idx,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cpu'
    )
    
    print(f" -> Resultado: Dice Score: {dice:.4f} | Tiempo: {run_time:.2f}s | Factible: {run_time <= T_max}")
    
    X_observed.append(x)
    y_dice_observed.append(dice)
    t_time_observed.append(run_time)

# Convertir listas a arrays de numpy para alimentar los modelos GP
X_obs_arr = np.array(X_observed)
y_obs_arr = np.array(y_dice_observed)
t_obs_arr = np.array(t_time_observed)

# %% [markdown]
# ## 6. Bucle de Optimización Bayesiana
# Ejecutamos las iteraciones de optimización bayesiana. En cada iteración:
# 1. Ajustamos los GPs con los datos históricos de Dice Score y tiempos de ejecución.
# 2. Optimizamos la función de adquisición combinada para sugerir el próximo punto.
# 3. Evaluamos la red con los hiperparámetros sugeridos.
# 4. Registramos y actualizamos las observaciones.

# %%
n_iterations = 7
print("\n=== Iniciando Bucle de Optimización Bayesiana ===")

for it in range(n_iterations):
    print(f"\n--- Iteración BO {it+1}/{n_iterations} ---")
    
    # 1. Ajustar Proceso Gaussiano de la Función Objetivo
    gp_objective.fit(X_obs_arr, y_obs_arr)
    
    # 2. Ajustar Proceso Gaussiano de la Restricción
    gp_constraint.fit(X_obs_arr, t_obs_arr)
    
    # 3. Encontrar el mejor valor factible actual (Dice Score)
    feasible_mask = t_obs_arr <= T_max
    if np.any(feasible_mask):
        best_y = np.max(y_obs_arr[feasible_mask])
        print(f"Mejor Dice Score factible actual: {best_y:.4f}")
    else:
        best_y = None
        print("Atención: Aún no se ha observado ningún punto factible.")
        
    # 4. Optimizar la función de adquisición combinada
    x_next, acq_val = optimize_acquisition(
        gp_obj=gp_objective,
        gp_constraint=gp_constraint,
        best_y=best_y,
        T_max=T_max,
        bounds=bounds,
        n_restarts=10,
        xi=0.01
    )
    
    # Desempacar la propuesta sugerida
    log10_lr_next, alpha_next, opt_val_next = x_next
    lr_next = 10**log10_lr_next
    opt_idx_next = int(np.clip(np.round(opt_val_next), 0, 2))
    opt_names = {0: 'Adam', 1: 'RMSprop', 2: 'SGD_Momentum'}
    opt_name_next = opt_names[opt_idx_next]
    
    print(f"Propuesta sugerida por adquisición (valor de adquisición: {acq_val:.4f}):")
    print(f" -> lr: {lr_next:.6f} | alpha_focal: {alpha_next:.3f} | optimizer: {opt_name_next} (val: {opt_val_next:.3f})")
    
    # 5. Evaluar la propuesta
    dice_next, run_time_next = evaluate_pipeline(
        learning_rate=lr_next,
        alpha_focal=alpha_next,
        optimizer_type=opt_idx_next,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cpu'
    )
    
    print(f"Resultado de la evaluación:")
    print(f" -> Dice Score: {dice_next:.4f} | Tiempo: {run_time_next:.2f}s | Factible: {run_time_next <= T_max}")
    
    # 6. Actualizar bases de datos históricas
    X_obs_arr = np.vstack([X_obs_arr, x_next])
    y_obs_arr = np.append(y_obs_arr, dice_next)
    t_obs_arr = np.append(t_obs_arr, run_time_next)

# %% [markdown]
# ## 7. Análisis y Resultados Óptimos
# Extraemos la mejor configuración que respete el umbral de tiempo máximo $T(x) \leq T_{max}$.

# %%
feasible_indices = np.where(t_obs_arr <= T_max)[0]

if len(feasible_indices) > 0:
    best_feasible_idx = feasible_indices[np.argmax(y_obs_arr[feasible_indices])]
    best_params = X_obs_arr[best_feasible_idx]
    best_dice = y_obs_arr[best_feasible_idx]
    best_time = t_obs_arr[best_feasible_idx]
    
    lr_opt = 10**best_params[0]
    alpha_opt = best_params[1]
    opt_idx_opt = int(np.clip(np.round(best_params[2]), 0, 2))
    opt_names = {0: 'Adam', 1: 'RMSprop', 2: 'SGD_Momentum'}
    
    print("\n==========================================")
    print("      OPTIMIZACIÓN COMPLETADA CON ÉXITO   ")
    print("==========================================")
    print(f"Mejor Configuración Factible Encontrada:")
    print(f" - learning_rate: {lr_opt:.6f}")
    print(f" - alpha_focal (Focal Loss): {alpha_opt:.4f}")
    print(f" - optimizer_type: {opt_names[opt_idx_opt]}")
    print(f"------------------------------------------")
    print(f" - Métrica (Validation Dice Score): {best_dice:.4f}")
    print(f" - Tiempo de ejecución por época: {best_time:.2f}s (Límite T_max: {T_max}s)")
    print("==========================================")
else:
    print("\nError: Ninguna de las configuraciones evaluadas cumplió con la restricción de tiempo.")

# %% [markdown]
# ## 8. Gráfico de Progreso de la Optimización
# Visualización del Dice Score a lo largo de las iteraciones, marcando las regiones factibles e infactibles.

# %%
plt.figure(figsize=(10, 5))
iterations = np.arange(len(y_obs_arr))

# Dibujar puntos
feasible_pts = plt.scatter(
    iterations[t_obs_arr <= T_max], 
    y_obs_arr[t_obs_arr <= T_max], 
    color='green', 
    s=100, 
    label='Configuración Factible', 
    zorder=3
)
infeasible_pts = plt.scatter(
    iterations[t_obs_arr > T_max], 
    y_obs_arr[t_obs_arr > T_max], 
    color='red', 
    s=100, 
    marker='X', 
    label='Excede Tiempo Máximo', 
    zorder=3
)

# Dibujar traza del mejor factible acumulado
best_so_far = []
current_best = -np.inf
for i in range(len(y_obs_arr)):
    if t_obs_arr[i] <= T_max:
        if y_obs_arr[i] > current_best:
            current_best = y_obs_arr[i]
    best_so_far.append(current_best if current_best != -np.inf else np.nan)
    
plt.plot(iterations, best_so_far, color='darkgreen', linestyle='--', linewidth=2, label='Mejor Dice Factible Acumulado')

plt.title('Progreso de la Optimización Bayesiana Constreñida')
plt.xlabel('Evaluaciones Totales (Diseño Inicial + Bucle BO)')
plt.ylabel('Validation Dice Score')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
os.makedirs('./plots', exist_ok=True)
plt.savefig('./plots/progreso_optimizacion.png')
print("\nGráfico de progreso guardado en './plots/progreso_optimizacion.png'.")
plt.show()
