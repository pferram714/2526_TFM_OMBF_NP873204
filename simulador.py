import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv
import os
import json
import logging
from pathlib import Path

# Simulador tumoral — Glioblastoma Multiforme
# Usa los parámetros D y rho estimados por la PINN para simular la evolución
# futura del tumor y comparar distintos escenarios terapéuticos.
#
# Paula Ferrándiz Ramos — TFM Bioinformática UAX

def setup_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)


# Solver FK con diferencias finitas explícitas

def solve_fk_forward(D, rho, u0, x, T, Nt, K=1.0, n_snapshots=50):
    Nx = len(x)
    L  = x[-1]
    dx = L / (Nx - 1)
    dt = T / Nt

    # comprobamos la condición CFL; si no se cumple, aumentamos Nt automáticamente
    cfl = D * dt / dx**2
    if cfl > 0.5:
        Nt = int(D * T / (0.4 * dx**2)) + 1
        dt = T / Nt

    u = u0.copy()

    # calculamos en qué pasos guardar snapshot y sus tiempos correspondientes
    snap_steps = np.linspace(0, Nt - 1, n_snapshots, dtype=int)
    t_snaps = snap_steps * dt
    U = np.zeros((n_snapshots, Nx))
    snap_idx = 0

    for step in range(Nt):
        # punto fantasma en los bordes para la condición de Neumann
        u_pad = np.pad(u, 1, mode="edge")
        lap = (u_pad[:-2] - 2 * u_pad[1:-1] + u_pad[2:]) / dx**2
        reaccion = rho * u * (1.0 - u / K)
        # paso de Euler explícito y recorte para mantener u en [0, K]
        u = np.clip(u + dt * (D * lap + reaccion), 0.0, K)

        # guardamos el snapshot si toca
        if step == snap_steps[snap_idx]:
            U[snap_idx] = u.copy()
            snap_idx += 1
            if snap_idx >= n_snapshots:
                break

    return {"x": x, "t_snaps": t_snaps, "U": U, "D": D, "rho": rho}


# Métricas clínicas

def compute_metrics(x, U, t_snaps, K=1.0, threshold=0.5):
    # para cada snapshot calculamos el radio y volumen de la zona tumoral activa
    thr = threshold * K
    radios  = []
    volumes = []

    for snap in U:
        # zona activa = posiciones donde u supera el umbral del 50% de K
        activo = x[snap >= thr]
        if len(activo) >= 2:
            radios.append((activo[-1] - activo[0]) / 2.0)
            volumes.append(activo[-1] - activo[0])
        else:
            radios.append(0.0)
            volumes.append(0.0)

    radios    = np.array(radios)
    volumes   = np.array(volumes)
    # velocidad del frente como derivada numérica del radio respecto al tiempo
    velocidad = np.gradient(radios, t_snaps)

    return {
        "t":        t_snaps,
        "radius":   radios,
        "volume":   volumes,
        "velocity": velocidad,
    }


def time_to_critical(metrics, r_crit):
    # buscamos el primer instante en que el radio supera r_crit
    idx = np.where(metrics["radius"] >= r_crit)[0]
    return float(metrics["t"][idx[0]]) if len(idx) > 0 else float("nan")


# Escenarios de tratamiento (nombre, reducción de rho, color, estilo de línea)
SCENARIOS = [
    ("Sin tratamiento",     0.00, "#e63946", "-"),
    ("Tto. leve (20%)",     0.20, "#f4a261", "--"),
    ("Tto. moderado (50%)", 0.50, "#2a9d8f", "-."),
    ("Tto. agresivo (80%)", 0.80, "#264653", ":"),
]


def plot_forward_simulation(sim, label, out_dir):
    x       = sim["x"]
    t_snaps = sim["t_snaps"]
    U       = sim["U"]

    # elegimos 6 instantes repartidos a lo largo de la simulación
    idx_plot = np.linspace(0, len(t_snaps) - 1, 6, dtype=int)

    fig, axes = plt.subplots(1, 6, figsize=(18, 3.5), sharey=True)
    cmap   = plt.cm.inferno
    colors = [cmap(0.15 + 0.7 * i / 5) for i in range(6)]

    for k, (ax, i) in enumerate(zip(axes, idx_plot)):
        ax.fill_between(x, U[i], alpha=0.3, color=colors[k])
        ax.plot(x, U[i], color=colors[k], lw=2)
        ax.set_title(f"t = {t_snaps[i]:.0f} d", fontsize=9)
        ax.set_xlabel("x (mm)", fontsize=8)
        ax.set_ylim(-0.05, 1.1)
        ax.tick_params(labelsize=7)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.6)

    axes[0].set_ylabel("u(x,t)", fontsize=8)
    fig.suptitle(
        f"Simulacion tumoral — {label}\n"
        f"D = {sim['D']:.4f} mm²/dia   rho = {sim['rho']:.4f} dia⁻¹",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_forward.png", dpi=130, bbox_inches="tight")
    plt.close()
    logging.info(f"  Guardado: {label}_forward.png")


def plot_treatment_comparison(x, D_est, rho_est, u0, T_sim, Nt_sim, K, label, out_dir):
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])
    ax_spatial = fig.add_subplot(gs[0])
    ax_radius  = fig.add_subplot(gs[1])

    r_crit = 3.0
    t_crit_results = {}

    # simulamos cada escenario reduciendo rho según el porcentaje de tratamiento
    for nombre, reduccion, color, ls in SCENARIOS:
        rho_tto = rho_est * (1.0 - reduccion)
        sim = solve_fk_forward(D_est, rho_tto, u0, x, T_sim, Nt_sim,
                               K=K, n_snapshots=100)
        metricas = compute_metrics(x, sim["U"], sim["t_snaps"], K=K)
        t_crit   = time_to_critical(metricas, r_crit)
        t_crit_results[nombre] = t_crit

        # panel izquierdo: perfil espacial al final de la simulación
        ax_spatial.plot(x, sim["U"][-1], color=color, ls=ls, lw=2, label=nombre)
        # panel derecho: evolución del radio en el tiempo
        ax_radius.plot(metricas["t"], metricas["radius"],
                       color=color, ls=ls, lw=2, label=nombre)

    ax_spatial.set_xlabel("x (mm)", fontsize=10)
    ax_spatial.set_ylabel("Densidad celular u(x,T)", fontsize=10)
    ax_spatial.set_title(f"Perfil tumoral al dia {T_sim:.0f}", fontsize=10)
    ax_spatial.set_ylim(-0.05, 1.1)
    ax_spatial.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.5,
                       label="Umbral (0.5K)")
    ax_spatial.legend(fontsize=8)
    ax_spatial.grid(alpha=0.3)

    ax_radius.axhline(r_crit, color="crimson", ls="--", lw=1.2,
                      label=f"Radio critico ({r_crit} mm)")
    ax_radius.set_xlabel("Tiempo (dias)", fontsize=10)
    ax_radius.set_ylabel("Radio tumoral (mm)", fontsize=10)
    ax_radius.set_title("Evolucion del radio bajo tratamiento", fontsize=10)
    ax_radius.legend(fontsize=8)
    ax_radius.grid(alpha=0.3)

    fig.suptitle(
        f"Analisis de tratamiento — {label}\n"
        f"D = {D_est:.4f} mm²/dia   rho_base = {rho_est:.4f} dia⁻¹",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_tratamiento.png", dpi=130, bbox_inches="tight")
    plt.close()
    logging.info(f"  Guardado: {label}_tratamiento.png")

    return t_crit_results


def plot_error_vs_noise(pinn_results, configs, noise_levels, out_dir):
    # construimos un dict label->fila para búsqueda rápida
    pinn_rows  = {r["label"]: r for r in pinn_results}
    cfg_labels = [c["label"] for c in configs]
    noise_vals = [float(n) for n in noise_levels]
    noise_tags = [f"{int(n*100):02d}" for n in noise_levels]

    colors_cfg = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    markers    = ["o", "s", "^", "D"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, param_key, title in zip(
        axes,
        ["D_err_pct", "rho_err_pct"],
        [r"Error relativo en $D$ (%)", r"Error relativo en $\rho$ (%)"]
    ):
        for i, cfg in enumerate(cfg_labels):
            errs = []
            for tag in noise_tags:
                label = f"cfg{cfg}_noise{tag}"
                errs.append(float(pinn_rows[label][param_key]) if label in pinn_rows else float("nan"))
            ax.plot(noise_vals, errs, marker=markers[i], color=colors_cfg[i],
                    lw=2, ms=8, label=f"Config. {cfg}")

        ax.axhline(5,  color="green",  ls="--", lw=1, alpha=0.7, label="5% (excelente)")
        ax.axhline(15, color="orange", ls="--", lw=1, alpha=0.7, label="15% (limite)")
        ax.set_xlabel(r"Nivel de ruido $\sigma$", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_yscale("log")
        ax.set_xticks(noise_vals)
        ax.set_xticklabels([str(v) for v in noise_vals])

    fig.suptitle("Degradacion del error de estimacion con el nivel de ruido",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "tabla_resumen_error_vs_ruido.png", dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("  Guardado: tabla_resumen_error_vs_ruido.png")


def main():
    # leemos la configuración desde el JSON
    with open("config.json") as f:
        global_cfg = json.load(f)

    data_folder = global_cfg["OUTPUT_FOLDER"]
    sim_cfg     = global_cfg["SIMULADOR"]
    K = 1.0

    T_sim        = sim_cfg["T_sim"]
    Nt_sim       = sim_cfg["Nt_sim"]
    r_crit       = sim_cfg["r_crit"]
    target_noise = set(sim_cfg["target_noise"])

    out_dir = Path("results/simulador")
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(out_dir / "simulador.log"))
    logging.info("=" * 65)
    logging.info("Simulador Tumoral — Glioblastoma Multiforme")
    logging.info(f"Periodo de simulacion: {T_sim} dias")
    logging.info(f"Radio critico: {r_crit} mm")
    logging.info("=" * 65)

    # comprobamos que existe el CSV con los resultados de la PINN
    results_path = Path("results/resultados.csv")
    if not results_path.exists():
        logging.error("No se encuentra results/resultados.csv. Ejecuta pinn_fk.py primero.")
        return

    # cargamos todos los resultados de la PINN
    with open(results_path) as f:
        pinn_results = list(csv.DictReader(f))

    # generamos la figura de error vs ruido con todos los experimentos
    configs      = global_cfg["CONFIGS"]
    noise_levels = global_cfg["NOISE_LEVELS"]
    plot_error_vs_noise(pinn_results, configs, noise_levels, out_dir)

    filas_resumen = []

    for row in pinn_results:
        label     = row["label"]
        noise_tag = label.split("_")[-1]

        # saltamos los niveles de ruido que no están en target_noise
        if noise_tag not in target_noise:
            continue

        D_true   = float(row["D_true"])
        rho_true = float(row["rho_true"])
        D_est    = float(row["D_est"])
        rho_est  = float(row["rho_est"])
        logging.info("")
        logging.info("-" * 60)
        logging.info(f"Paciente: {label}")
        logging.info(f"  D_est={D_est:.5f}  rho_est={rho_est:.5f}")

        # construimos el path del .npz: "cfgA_noise03" → "configA_noise03.npz"
        parts     = label.split("_")      # ['cfgA', 'noise03']
        cfg_id    = parts[0][3:]          # 'cfgA' → 'A'
        noise_str = parts[1]              # 'noise03'
        npz_path  = f"{data_folder}/config{cfg_id}_{noise_str}.npz"
        if not os.path.exists(npz_path):
            logging.warning(f"  No encontrado: {npz_path}")
            continue

        npz = np.load(npz_path)
        x   = npz["x"]
        u0  = npz["U_clean"][0]  # primer snapshot como condición inicial

        # simulamos la evolución hacia adelante con los parámetros estimados
        logging.info(f"  Simulando {T_sim} dias hacia adelante...")
        sim = solve_fk_forward(D_est, rho_est, u0, x, T_sim, Nt_sim,
                               K=K, n_snapshots=100)
        plot_forward_simulation(sim, label, out_dir)

        # calculamos las métricas clínicas del caso sin tratamiento
        metricas = compute_metrics(x, sim["U"], sim["t_snaps"], K=K)

        # comparamos los cuatro escenarios de tratamiento
        logging.info("  Simulando escenarios de tratamiento...")
        t_crits = plot_treatment_comparison(
            x, D_est, rho_est, u0, T_sim, Nt_sim, K, label, out_dir
        )

        # calculamos la velocidad teórica del frente según Fisher: v* = 2*sqrt(D*rho)
        v_fisher = 2.0 * np.sqrt(D_est * rho_est)

        t_crit_sin = t_crits.get("Sin tratamiento",     float("nan"))
        t_crit_lev = t_crits.get("Tto. leve (20%)",     float("nan"))
        t_crit_mod = t_crits.get("Tto. moderado (50%)", float("nan"))
        t_crit_agr = t_crits.get("Tto. agresivo (80%)", float("nan"))

        logging.info(f"  Velocidad teorica del frente (Fisher): {v_fisher:.4f} mm/dia")
        logging.info(f"  Tiempo hasta r={r_crit}mm (sin tto): "
                     f"{'%.1f dias' % t_crit_sin if not np.isnan(t_crit_sin) else 'no alcanzado'}")

        # añadimos la fila al resumen
        filas_resumen.append({
            "label":          label,
            "D_est":          D_est,
            "rho_est":        rho_est,
            "D_err_pct":      row["D_err_pct"],
            "rho_err_pct":    row["rho_err_pct"],
            "v_frente":       round(v_fisher, 4),
            "t_crit_sin_tto": round(t_crit_sin, 1) if not np.isnan(t_crit_sin) else "N/A",
            "t_crit_leve":    round(t_crit_lev,  1) if not np.isnan(t_crit_lev) else "N/A",
            "t_crit_mod":     round(t_crit_mod,  1) if not np.isnan(t_crit_mod) else "N/A",
            "t_crit_agr":     round(t_crit_agr,  1) if not np.isnan(t_crit_agr) else "N/A",
        })

    # guardamos el CSV con las métricas clínicas de todos los pacientes
    if filas_resumen:
        csv_path = out_dir / "simulador_resumen.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=filas_resumen[0].keys())
            writer.writeheader()
            writer.writerows(filas_resumen)
        logging.info(f"\nResumen guardado en: {csv_path}")

    logging.info("")
    logging.info("=" * 65)
    logging.info("RESUMEN DEL SIMULADOR")
    logging.info("=" * 65)
    header = f"{'Paciente':<25} {'v_frente':>10} {'t_crit(sin)':>13} {'t_crit(50%)':>13}"
    logging.info(header)
    logging.info("-" * 63)
    for r in filas_resumen:
        logging.info(
            f"  {r['label']:<23} {r['v_frente']:>9.4f}  "
            f"{str(r['t_crit_sin_tto']):>12}  {str(r['t_crit_mod']):>12}"
        )
    logging.info("=" * 65)


if __name__ == "__main__":
    main()
