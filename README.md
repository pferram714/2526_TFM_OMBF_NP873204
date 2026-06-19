# 2526_TFM_OMBF_NP873204
# MODELIZACIÓN MATEMÁTICA DEL GLIOBLASTOMA MULTIFORME MEDIANTE LA ECUACIÓN DE FISHER-KOLMOGOROV: TFM BIOINFORMÁTICA UAX
# Estimación de Parámetros de Glioblastoma mediante Redes Neuronales Informadas por la Física (PINN)

**Trabajo de Fin de Máster — Máster Universitario en Bioinformática, UAX**  
**Autora**: Paula Ferrándiz Ramos

---

## Descripción

Este proyecto implementa una metodología basada en **Physics-Informed Neural Networks (PINNs)** para estimar parámetros biológicos de crecimiento tumoral (glioblastoma multiforme) a partir de datos sintéticos ruidosos.

El modelo matemático subyacente es la ecuación de **Fisher-Kolmogorov**, una ecuación diferencial parcial de reacción-difusión que captura tanto la difusión de células tumorales como su crecimiento logístico:

```
du/dt = D · ∂²u/∂x² + ρ · u · (1 - u/K)
```

donde:
- `D` — coeficiente de difusión tumoral (mm²/día)
- `ρ` — tasa de proliferación celular (día⁻¹)
- `K` — capacidad de carga (= 1 en variables normalizadas)

El objetivo principal es inferir `D` y `ρ` a partir de observaciones parciales y ruidosas, y usar los parámetros estimados para predecir la progresión tumoral y evaluar escenarios de tratamiento.

---

## Estructura del proyecto

```
TFM_18062026/
├── config.json                        # Configuración centralizada del experimento
├── generador_datos.py                 # Generador de datos sintéticos (resolución PDE)
├── pinn_fk.py                         # Entrenamiento de la PINN e inferencia de parámetros
├── simulador.py                       # Simulador clínico de progresión y tratamiento
├── referencias.bib                    # Bibliografía en formato BibTeX
├── dataset.log                        # Log de generación de datos
├── syntetic_data/                     # Datos sintéticos generados
│   ├── configX_noiseYY.npz            # Datos generados guardados por configuración y nivel de ruido
│   └── *.png                          # Visualizaciones de la evolución tumoral
└── results/                           # Resultados del entrenamiento y la simulación
    ├── pinn_training.log              # Log del entrenamiento de la PINN
    ├── resultados.csv                 # Parámetros estimados y errores relativos
    ├── summary_errors.png             # Visualización gráfica de errores
    ├── *.pt                           # Pesos de los modelos entrenados (PyTorch)
    ├── *.npz                          # Historial de pérdidas por experimento
    └── simulador/                     # Resultado del simulador
        ├── simulador.log              # Log del proceso del simulador
        ├── simulador_resumen.csv      # Resumen de resultados tras la simulación
        └── *.png                      # Perfiles espaciales y curvas de tratamiento + gráfico resumen de errores
```

---

## Instalación

### Requisitos

- Python 3.9+
- PyTorch
- NumPy
- Matplotlib

### Instalación de dependencias

```bash
pip install torch numpy matplotlib
```

---

## Uso

El pipeline se ejecuta en tres pasos secuenciales:

### 1. Generar datos sintéticos

```bash
python generador_datos.py
```

Resuelve la EDP de Fisher-Kolmogorov mediante diferencias finitas explícitas para las 4 configuraciones tumorales y los 4 niveles de ruido definidos en `config.json`. Genera 16 archivos `.npz` y sus correspondientes visualizaciones en `syntetic_data/`.

### 2. Entrenar la PINN

```bash
python pinn_fk.py
```

Entrena una red neuronal para cada combinación configuración/ruido (16 experimentos en total). La red aprende simultáneamente la solución de la EDP y los parámetros biológicos desconocidos `D` y `ρ`. Los resultados se guardan en `results/resultados.csv`.

### 3. Simular progresión tumoral

```bash
python simulador.py
```

Usa los parámetros estimados por la PINN para simular la evolución tumoral a 90 días y comparar cuatro escenarios de tratamiento. Los resultados clínicos se guardan en `results/simulador/`.

---

## Configuración

El archivo `config.json` centraliza todos los parámetros del experimento:

### Configuraciones tumorales

| ID | D (mm²/día) | ρ (día⁻¹) | Descripción |
|----|-------------|-----------|-------------|
| A  | 0.05        | 0.20      | Difusión baja, crecimiento lento |
| B  | 0.10        | 0.50      | Perfil equilibrado |
| C  | 0.20        | 0.30      | Alta difusión, crecimiento moderado |
| D  | 0.08        | 1.00      | Crecimiento rápido, baja difusión |

### Niveles de ruido gaussiano

`[0.00, 0.01, 0.03, 0.05]` (desviación estándar sobre el campo normalizado `u`)

### Parámetros de simulación

| Parámetro | Valor |
|-----------|-------|
| Dominio espacial | x ∈ [0, 10] mm |
| Horizonte temporal | t ∈ [0, 10] días |
| Puntos espaciales | 200 |
| Pasos temporales | 10 000 |
| Condición inicial | Gaussiana (centro=5 mm, σ=0.5 mm, amplitud=0.8) |
| Snapshots por simulación | 5 |

### Arquitectura de la PINN

| Parámetro | Valor |
|-----------|-------|
| Capas | `[2, 64, 64, 64, 64, 1]` |
| Activación | Tanh |
| Épocas | 20 000 |
| Optimizador | Adam (lr=0.001, decay ×0.5 cada 5 000 pasos) |
| Puntos de colocación (PDE) | 3 000 |
| Puntos de observación | 200 |

### Pesos de la función de pérdida

```
L_total = w_data · L_data + w_phys · L_phys + w_ic · L_ic + w_bc · L_bc
```

| Término | Peso | Descripción |
|---------|------|-------------|
| `L_data` | 1.0 | Fidelidad a los datos observados |
| `L_phys` | 1.0 | Residuo de la PDE en puntos de colocación |
| `L_ic`   | 1.0 | Condición inicial (t = 0) |
| `L_bc`   | 0.1 | Condición de Neumann en los bordes (∂u/∂x = 0) |

---

## Outputs principales

| Archivo | Descripción |
|---------|-------------|
| `results/resultados.csv` | Parámetros estimados, errores relativos y pérdida final por experimento |
| `results/*.pt` | Pesos de los modelos PyTorch entrenados |
| `results/summary_errors.png` | Gráfico de errores de estimación por configuración y ruido |
| `results/simulador/simulador_resumen.csv` | Métricas clínicas: velocidad del frente tumoral, tiempo al radio crítico |
| `results/simulador/*_tratamiento.png` | Evolución del radio tumoral bajo distintos tratamientos |

---

## Escenarios de tratamiento simulados

| Escenario | Reducción de ρ | Descripción |
|-----------|----------------|-------------|
| Sin tratamiento | 0% | Progresión natural |
| Leve | 20% | Respuesta parcial |
| Moderado | 50% | Respuesta significativa |
| Agresivo | 80% | Supresión fuerte de la proliferación |

---

## Flujo de datos

```
config.json
    │
    ▼
generador_datos.py ──► syntetic_data/*.npz  +  *.png
    │
    ▼
pinn_fk.py ──► results/resultados.csv  +  *.pt  +  *.npz
    │
    ▼
simulador.py ──► results/simulador/*.csv  +  *.png
```

---

## Contexto científico

El glioblastoma multiforme (GBM) es el tumor cerebral primario más agresivo, con una mediana de supervivencia inferior a 15 meses. La dificultad para obtener datos densamente muestreados justifica el uso de modelos matemáticos para extrapolar la dinámica tumoral a partir de observaciones escasas.

Las **Physics-Informed Neural Networks** integran directamente las ecuaciones diferenciales como restricción en la función de pérdida, lo que permite:
- Estimar parámetros biológicos no observables directamente
- Regularizar el aprendizaje mediante conocimiento físico a priori
- Obtener estimaciones robustas incluso con datos escasos y ruidosos
Para más detalles teóricos leer la memoria presentada en formato PDF.
---

## Referencia

Si utilizas este código en trabajos académicos, por favor cita el TFM correspondiente. Las referencias bibliográficas completas se encuentran en [`referencias.bib`](referencias.bib).
