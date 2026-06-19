import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import json
import sys
import logging

"""
Fisher-Kolmogorov Synthetic Data Generator
===========================================
Esre código conforma un generador de datos sintéticos para estudiar el crecimiento tumoral
resolviendo la ecuación de reacción-difusión de Fisher-Kolmogorov en una dimensión 
usando diferencias finitas.

EDP:  du/dt = D * Laplacian(u) + rho * u * (1 - u/K)

Paula Ferrándiz Ramos
TFM — Máster Universitario en Bioinformática, UAX
"""

# Configuración de logs
def setup_logging():
    """Configura el sistema de logging para que escriba en la consola y en un archivo."""
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Manejador para archivo
    file_handler = logging.FileHandler("dataset.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Manejador para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


# Carga de configuración
def load_config(config_file="config.json"):
    """
    Carga la configuración desde un archivo JSON.
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        logging.error(f"El archivo de configuración '{config_file}' no fue encontrado.")
        return None
    except json.JSONDecodeError:
        logging.error(f"El archivo de configuración '{config_file}' no es un JSON válido.")
        return None
    
setup_logging()
config = load_config()
if not config:
      logging.critical("Saliendo del programa debido a un error de configuración.")  
      sys.exit(1)

# Cargamos configuraciones
OUTPUT_FOLDER = config.get("OUTPUT_FOLDER")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Semilla de reproducibilidad
RNG = np.random.default_rng(seed=42)

#  SOLVER

def solve_fk(
    D: float = 0.1,
    rho: float = 0.5,
    K: float = 1.0,
    L: float = 10.0,
    T: float = 10.0,
    Nx: int = 100,
    Nt: int = 5000,
    u0_center: float = 5.0,
    u0_width: float = 0.5,
    u0_amplitude: float = 0.8,
    noise_std: float = 0.0,
    n_snapshots: int = 5,
) -> dict:
    """
    Resuelve la ecuación de Fisher-Kolmogorov 1D usando diferencias finitas explícitas.

    Dominio espacial  : [0, L]  con Nx puntos de malla  →  dx = L / (Nx-1)
    |----|----|----|----|  (Nx puntos, separados dx)
    0                   L

    Dominio temporal  : [0, T]  con Nt pasos de tiempo  →  dt = T / Nt
    0 → dt → 2dt → ... → T  (Nt pasos)

    Condición de estabilidad (CFL):  D * dt / dx^2 <= 0.5

    Condición inicial: campana gaussiana centrada en u0_center (ay muchas células en el centro y pocas en los bordes)
        u(x,0) = u0_amplitude * exp(-(x - u0_center)^2 / (2 * u0_width^2))

    Condiciones de contorno: flujo nulo (Neumann)  du/dx = 0  en x=0, x=L

    La ecuación combina dos fenómenos:

    1. Difusión (D): las células se dispersan hacia zonas menos pobladas, como tinta en agua
    2. Proliferación (rho): las células se reproducen, pero frenadas por la capacidad de carga K 
    (no pueden crecer infinitamente)

    Parámetros:

    D            : coeficiente o tensor de difusión        
    rho          : ratio de proliferación o crecimiento      
    K            : capacidad máxima de carga del tejido     
    L            : longitud del dominio            
    T            : tiempo total de simulación
    Nx           : número de puntos de malla espaciales
    Nt           : número de pasos de tiempo
    u0_center    : centro de la gaussiana inicial
    u0_width     : desviación típica de la gaussiana inicial
    u0_amplitude : valor pico de la condición inicial
    noise_std    : desviación típica del ruido gaussiano aditivo (0 = sin ruido)
    n_snapshots  : número de instantes temporales a devolver (uniformemente espaciados)

    Devuelve un diccionario con las claves:
        x           : malla espacial        (Nx,)
        t_snaps     : instantes de snapshot (n_snapshots,)
        U_clean     : snapshots sin ruido   (n_snapshots, Nx)
        U_noisy     : snapshots con ruido   (n_snapshots, Nx)  [recortado a [0,K]]
        params      : dict de parámetros de la simulación
        dx, dt      : espaciados de la malla
        cfl         : número CFL (debe ser <= 0.5 para estabilidad)
    """

    # 1. Cálculo de los espaciados de la malla
    dx = L / (Nx - 1)
    dt = T / Nt
    cfl = D * dt / dx**2

    if cfl > 0.5:
        raise ValueError(
            f"Condición CFL violada: D*dt/dx^2 = {cfl:.4f} > 0.5. "
            f"Aumenta Nt o reduce D/dx."
        )
    
    # 2. Creación de la malla espacial. Creamos un array de Nx puntos igualmente espaciados entre 0 y L.
    x = np.linspace(0, L, Nx)

    # 3. Condición inicial. Calcula el valor inicial de u en cada punto del espacio. 
    # El resultado es una campana centrada en u0_center.
    u = u0_amplitude * np.exp(-((x - u0_center) ** 2) / (2 * u0_width**2))

    # 4. Preparación del almacenamiento de snapshots
    snap_steps = np.linspace(0, Nt - 1, n_snapshots, dtype=int) # pasos de tiempo en los que se hace la "foto"
    U_clean = np.zeros((n_snapshots, Nx)) # matriz vacía donde se irá guardando cada foto (sin ruido)
    snap_idx = 0 # contador de fotos

    # 5. Simulación (bucle principal):
    for step in range(Nt):
        u_pad = np.pad(u, 1, mode="edge") # añade un punto fantasma en cada extremo copiando el valor del borde,
        # esto implementa la condición de Neumann (flujo cero)
        laplacian = (u_pad[:-2] - 2 * u_pad[1:-1] + u_pad[2:]) / dx**2 
        # la fórmula del laplaciano mide la "curvatura" de u: si hay más células a la derecha que a la izquierda,
        # la difusión las empuja hacia la izquierda
 
        # término logístico (cálculo de la proliferación)
        reaction = rho * u * (1.0 - u / K)

        # Método de Euler explícito
        u = u + dt * (D * laplacian + reaction)
        u = np.clip(u, 0.0, K)                      

        # Si el paso actual coincide con uno de los momentos elegidos, guarda una copia de u en U_clean y avanza el contador.
        if step == snap_steps[snap_idx]:
            U_clean[snap_idx] = u.copy()
            snap_idx += 1
            if snap_idx >= n_snapshots:
                break

    # 6. Cálculo de tiempos: se convierten los índices de los snapshots a días reales.
    t_snaps = np.array([s * dt for s in snap_steps])

    # 7. Añadir ruido para datos más realistas
    if noise_std > 0:
        noise = RNG.normal(0, noise_std, U_clean.shape)
        U_noisy = np.clip(U_clean + noise, 0.0, K)
    else:
        U_noisy = U_clean.copy()

    return {
    "x": x,           # la malla espacial
    "t_snaps": t_snaps,  # los tiempos de cada snapshot
    "U_clean": U_clean,  # las fotos sin ruido
    "U_noisy": U_noisy,  # las fotos con ruido
    "params": {...},     # los parámetros usados
    "dx": dx, "dt": dt,  # los espaciados
    "cfl": cfl,          # el número de estabilidad
}


#  GENERACIÓN DEL DATASET: parameter configurations


# Para simular variabilidad biológica en el dataset, se definen 4 tipos de tumores distintos, cada uno con
# un comportamiento biológico diferente según sus parámetros D (difusión) y rho (crecimiento).
# A: Tumor pequeño, quieto
# B: Tumor equilibrado
# C: apariencia difusa en MRI: las células se dispersan por el tejido pero no proliferan mucho (mancha grande y difusa)
# D: apariencia nodular en MRI: las células se multiplican rápido pero no se mueven (bulto denso y localizado)
# Nota: MRI (Magnetic Resonance Imaging) es lo que en español se llama resonancia magnética.
CONFIGS = config.get("CONFIGS")

# Cada simulación se ejecutará con distintos niveles de ruido para imitar mediciones reales más o menos precisas.
# Al estar K=1, un ruido de 0.05 significa un 5% de perturbación sobre el valor máximo posible.
NOISE_LEVELS = config.get("NOISE_LEVELS") 


# VISUALIZACIÓN DE DATOS

def plot_snapshots(result: dict, config: dict, noise_std: float, save: bool = True):
    """
    Genera una figura con un gráfico por cada snapshot, mostrando cómo evoluciona el tumor a lo largo del tiempo.

    Para cada snapshot:
    1. Dibuja la curva limpia como una línea continua
    2. Si hay ruido, dibuja los puntos ruidosos encima como puntitos semitransparentes (.)
    3. Pone el título con el tiempo real en días
    """
    # Extracción de datos del diccionario resultante
    x = result["x"] # eje horizontal
    t = result["t_snaps"] # tiempos de cada foto
    U_clean = result["U_clean"] # curvas sin ruido
    U_noisy = result["U_noisy"] # curvas con ruido
    n = len(t) # longitud de tiempos

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5), sharey=True)
    cmap = plt.cm.plasma
    colors = [cmap(i / (n - 1)) for i in range(n)]

    for i, ax in enumerate(axes):
        ax.plot(x, U_clean[i], color=colors[i], lw=2, label="Exacto")
        if noise_std > 0:
            ax.plot(x, U_noisy[i], ".", color=colors[i], ms=3, alpha=0.5, label="Ruidoso")
        ax.set_title(f"t = {t[i]:.1f} d", fontsize=9)
        ax.set_xlabel("x (mm)", fontsize=8)
        ax.set_ylim(-0.05, 1.1)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("u(x,t)", fontsize=8)

    tag = f"D={config['D']}, ρ={config['rho']}, σ={noise_std}"
    fig.suptitle(f"FK  —  Config {config['label']}  ({tag})", fontsize=10, y=1.02)
    plt.tight_layout()

    if save:
        fname = f"{OUTPUT_FOLDER}/config{config['label']}_noise{int(noise_std*100):02d}.png"
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        logging.info(f"  Saved: {fname}")
    plt.close()


def plot_noise_comparison(results_by_noise: dict, config: dict, t_idx: int = 2):
    """
    Compara el mismo snapshot de la simulación con distintos niveles de ruido,
    para ver visualmente cuánto afecta el ruido a la medición.
    Produce una fila de subgráficos, uno por cada nivel de ruido, todos mostrando 
    el mismo instante de tiempo t_idx:
    En cada subgráfico:
    - La línea negra es siempre la curva limpia (U_clean) como referencia
    - Los puntos azules son los datos ruidosos (U_noisy) encima

    """
    fig, axes = plt.subplots(1, len(NOISE_LEVELS), figsize=(14, 3.5), sharey=True)
    x = list(results_by_noise.values())[0]["x"]

    for ax, (ns, res) in zip(axes, results_by_noise.items()):
        t_val = res["t_snaps"][t_idx]
        ax.plot(x, res["U_clean"][t_idx], "k-", lw=2, label="Exacto", zorder=3)
        if ns > 0:
            ax.plot(x, res["U_noisy"][t_idx], ".", color="steelblue",
                    ms=4, alpha=0.6, label=f"Ruidoso (σ={ns})")
        ax.set_title(f"σ = {ns}", fontsize=10)
        ax.set_xlabel("x (mm)", fontsize=9)
        ax.set_ylim(-0.1, 1.15)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("u(x,t)", fontsize=9)
    fig.suptitle(
        f"Efecto del nivel de ruido: Config {config['label']} en t={t_val:.1f} d",
        fontsize=11
    )
    plt.tight_layout()
    fname = f"{OUTPUT_FOLDER}/noise_comparison_config{config['label']}.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    logging.info(f"  Saved: {fname}")
    plt.close()

def plot_param_comparison(results_by_config: dict, t_idx: int = 3, noise_std: float = 0.0):
    """
    Compara las 4 configuraciones de tumor en el mismo instante de tiempo.
    Produce una fila de 4 subplots de línea, uno por configuración, todos en el mismo snapshot t_idx.
    Recibe results_by_config, un diccionario con una simulación por cada configuración.
    """
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, (cfg_label, res) in zip(axes, results_by_config.items()):
        x = res["x"]
        U = res["U_noisy"] if noise_std > 0 else res["U_clean"]
        t_val = res["t_snaps"][t_idx]
        cfg = next(c for c in CONFIGS if c["label"] == cfg_label)
        ax.plot(x, U[t_idx], color="crimson", linewidth=1.5)
        ax.set_title(f"Config {cfg_label}\nD={cfg['D']}, ρ={cfg['rho']}", fontsize=9)
        ax.set_xlabel("x (mm)", fontsize=8)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_ylabel("u (densidad tumoral)", fontsize=8)
    fig.suptitle(f"FK — Comparación de parámetros en t={t_val:.1f} d", fontsize=11)
    plt.tight_layout()
    fname = f"{OUTPUT_FOLDER}/param_comparison_t{t_idx}.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    logging.info(f"  Saved: {fname}")
    plt.close()


def main():
    logging.info("=" * 60)
    logging.info("Generador de Datos Sintéticos Fisher-Kolmogorov")
    logging.info("=" * 60)

    sim_params = config.get("SIM_PARAMS")

    # Datasets
    logging.info("Generando datasets...")
    all_results = {}
    results_noiseless = {}

    for cfg in CONFIGS:
        results_by_noise = {}
        for ns in NOISE_LEVELS:
            logging.info(f"  Config {cfg['label']}  D={cfg['D']}  rho={cfg['rho']}  noise={ns}")

            res = solve_fk(
                D=cfg["D"], rho=cfg["rho"],
                noise_std=ns,
                **sim_params
            )
            logging.info(f"    CFL = {res['cfl']:.4f}")

            plot_snapshots(res, cfg, ns)
            all_results[(cfg["label"], ns)] = res
            results_by_noise[ns] = res

            np.savez(
                f"{OUTPUT_FOLDER}/config{cfg['label']}_noise{int(ns*100):02d}.npz",
                x=res["x"],
                t_snaps=res["t_snaps"],
                U_clean=res["U_clean"],
                U_noisy=res["U_noisy"],
                D=cfg["D"], rho=cfg["rho"]
            )

            if ns == 0.0:
                results_noiseless[cfg["label"]] = res

        plot_noise_comparison(results_by_noise, cfg)

    plot_param_comparison(results_noiseless, t_idx=3)

    # Tabla resumen
    logging.info("=" * 60)
    logging.info("RESUMEN DEL DATASET")
    logging.info("=" * 60)
    logging.info(f"{'Config':<10} {'D':>6} {'rho':>6} {'Descripcion'}")
    logging.info("-" * 55)
    for cfg in CONFIGS:
        logging.info(f"  {cfg['label']:<8} {cfg['D']:>6.2f} {cfg['rho']:>6.2f}  {cfg['desc']}")
    logging.info(f"Niveles de ruido: {NOISE_LEVELS}")
    logging.info(f"Total archivos .npz: {len(CONFIGS) * len(NOISE_LEVELS)}")
    logging.info(f"Todos los archivos guardados en: {OUTPUT_FOLDER}/")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()