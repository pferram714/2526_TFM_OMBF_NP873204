import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import json
import logging
import csv
from pathlib import Path
from itertools import product

"""
PINN para estimar D y rho en la ecuación de Fisher-Kolmogorov
EDP: du/dt = D*d2u/dx2 + rho*u*(1 - u/K)
Paula Ferrándiz Ramos — TFM Bioinformática UAX
"""


torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)


# Red neuronal base

class MLP(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.net = nn.Sequential()
        # construimos la red capa a capa según la lista de tamaños
        for i in range(len(layers) - 1):
            self.net.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.net.append(nn.Tanh())

    def forward(self, x):
        return self.net(x)


# PINN con D y rho como parámetros entrenables

class PINN(nn.Module):

    def __init__(self, layers, D_init=0.1, rho_init=0.5, K=1.0,
                 x_scale=10.0, t_scale=10.0):
        super().__init__()
        self.net = MLP(layers)
        self.K = K
        self.x_scale = x_scale
        self.t_scale = t_scale
        # guardamos log(D) y log(rho) para que al aplicar exp() siempre sean positivos
        self.log_D   = nn.Parameter(torch.tensor(float(np.log(D_init))))
        self.log_rho = nn.Parameter(torch.tensor(float(np.log(rho_init))))

    @property
    def D(self):
        return torch.exp(self.log_D)

    @property
    def rho(self):
        return torch.exp(self.log_rho)

    def forward(self, x, t):
        # normalizamos x y t antes de pasarlos a la red
        xt = torch.cat([x / self.x_scale, t / self.t_scale], dim=1)
        return self.net(xt)

    def residual(self, x_col, t_col):
        # calculamos las derivadas de u respecto a x y t con diferenciación automática
        x_col = x_col.requires_grad_(True)
        t_col = t_col.requires_grad_(True)
        u = self.forward(x_col, t_col)
        grads = torch.autograd.grad(
            u, [x_col, t_col],
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )
        du_dx, du_dt = grads
        # segunda derivada espacial
        du_dxx = torch.autograd.grad(
            du_dx, x_col,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True
        )[0]
        reaccion = self.rho * u * (1.0 - u / self.K)
        # devolvemos el residuo de la EDP (debería ser ~0 si la red es buena)
        return du_dt - self.D * du_dxx - reaccion

    def bc_residual(self, x_bc, t_bc):
        # calculamos du/dx en los bordes para imponer condición de Neumann
        x_bc = x_bc.requires_grad_(True)
        u = self.forward(x_bc, t_bc)
        du_dx = torch.autograd.grad(
            u, x_bc,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        return du_dx


def compute_loss(model, x_obs, t_obs, u_obs, x_col, t_col,
                 x_ic, u_ic, x_bc, t_bc,
                 w_data, w_phys, w_ic, w_bc):

    # error entre la predicción de la red y los datos observados
    u_pred = model(x_obs, t_obs)
    L_data = torch.mean((u_pred - u_obs) ** 2)

    # error en el cumplimiento de la EDP en los puntos de colocación
    res = model.residual(x_col, t_col)
    L_phys = torch.mean(res ** 2)

    # error en la condición inicial (t=0)
    u_ic_pred = model(x_ic, torch.zeros_like(x_ic))
    L_ic = torch.mean((u_ic_pred - u_ic) ** 2)

    # error en la condición de contorno (bordes del dominio)
    bc_res = model.bc_residual(x_bc, t_bc)
    L_bc = torch.mean(bc_res ** 2)

    # pérdida total ponderada
    loss = w_data * L_data + w_phys * L_phys + w_ic * L_ic + w_bc * L_bc
    componentes = {
        "L_data": L_data.item(),
        "L_phys": L_phys.item(),
        "L_ic":   L_ic.item(),
        "L_bc":   L_bc.item(),
        "total":  loss.item(),
    }
    return loss, componentes


def load_npz(path, n_obs=200):
    # cargamos el archivo de datos generado por generador_datos.py
    data = np.load(path)
    x        = data["x"]
    t_snaps  = data["t_snaps"]
    U_noisy  = data["U_noisy"]
    U_clean  = data["U_clean"]
    D_true   = float(data["D"])
    rho_true = float(data["rho"])

    # aplanamos la malla espacio-tiempo para tener una lista de puntos (x_i, t_j, u_ij)
    Nx      = len(x)
    n_snaps = len(t_snaps)
    X_all   = np.repeat(x, n_snaps)
    T_all   = np.tile(t_snaps, Nx)
    U_all   = U_noisy.T.ravel()

    # seleccionamos n_obs puntos al azar como conjunto de observaciones
    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(X_all), size=min(n_obs, len(X_all)), replace=False)
    x_obs = X_all[idx].reshape(-1, 1).astype(np.float32)
    t_obs = T_all[idx].reshape(-1, 1).astype(np.float32)
    u_obs = U_all[idx].reshape(-1, 1).astype(np.float32)

    # condición inicial: todos los puntos espaciales en t=0
    x_ic = x.reshape(-1, 1).astype(np.float32)
    u_ic = U_noisy[0].reshape(-1, 1).astype(np.float32)

    return {
        "x_obs": x_obs, "t_obs": t_obs, "u_obs": u_obs,
        "x_ic": x_ic, "u_ic": u_ic,
        "x": x, "t_snaps": t_snaps, "U_noisy": U_noisy, "U_clean": U_clean,
        "D_true": D_true, "rho_true": rho_true,
        "L": float(x[-1]), "T": float(t_snaps[-1]),
    }


def make_collocation_points(L, T, n_col=2000):
    # puntos aleatorios dentro del dominio (x,t) donde imponemos la EDP
    rng   = np.random.default_rng(0)
    x_col = rng.uniform(0, L, (n_col, 1)).astype(np.float32)
    t_col = rng.uniform(0, T, (n_col, 1)).astype(np.float32)
    return x_col, t_col


def make_bc_points(L, T, n_bc=200):
    # puntos en los bordes x=0 y x=L para imponer la condición de contorno
    rng     = np.random.default_rng(1)
    t_vals  = rng.uniform(0, T, (n_bc, 1)).astype(np.float32)
    x_left  = np.zeros((n_bc // 2, 1), dtype=np.float32)
    x_right = np.full((n_bc - n_bc // 2, 1), L, dtype=np.float32)
    x_bc = np.vstack([x_left, x_right])
    t_bc = np.vstack([t_vals[:n_bc // 2], t_vals[n_bc // 2:]])
    return x_bc, t_bc


def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32, device=DEVICE)


def train(model, data, pinn_cfg, out_dir, label):

    # extraemos hiperparámetros del config
    n_epochs       = pinn_cfg["n_epochs"]
    lr             = pinn_cfg["lr"]
    n_col          = pinn_cfg["n_col"]
    n_bc           = pinn_cfg["n_bc"]
    w_data         = pinn_cfg["w_data"]
    w_phys         = pinn_cfg["w_phys"]
    w_ic           = pinn_cfg["w_ic"]
    w_bc           = pinn_cfg["w_bc"]
    log_every      = pinn_cfg.get("log_every", 500)
    lr_decay_step  = pinn_cfg.get("lr_decay_step", 5000)
    lr_decay_gamma = pinn_cfg.get("lr_decay_gamma", 0.5)

    L = data["L"]
    T = data["T"]

    # convertimos los datos a tensores de PyTorch
    x_obs = to_tensor(data["x_obs"])
    t_obs = to_tensor(data["t_obs"])
    u_obs = to_tensor(data["u_obs"])
    x_ic  = to_tensor(data["x_ic"])
    u_ic  = to_tensor(data["u_ic"])

    # generamos los puntos de colocación y de condición de contorno
    x_col_np, t_col_np = make_collocation_points(L, T, n_col)
    x_bc_np,  t_bc_np  = make_bc_points(L, T, n_bc)
    x_col = to_tensor(x_col_np)
    t_col = to_tensor(t_col_np)
    x_bc  = to_tensor(x_bc_np)
    t_bc  = to_tensor(t_bc_np)

    # optimizador y scheduler para reducir el learning rate cada cierto número de épocas
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma
    )

    # diccionario para guardar el histórico de pérdidas y parámetros estimados
    history = {k: [] for k in ["total", "L_data", "L_phys", "L_ic", "L_bc", "D", "rho"]}

    logging.info(f"[{label}] Iniciando entrenamiento — {n_epochs} épocas — {DEVICE}")
    logging.info(f"[{label}] D_true={data['D_true']:.4f}  rho_true={data['rho_true']:.4f}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        # calculamos la pérdida, propagamos hacia atrás y actualizamos pesos
        loss, comps = compute_loss(
            model,
            x_obs, t_obs, u_obs,
            x_col, t_col,
            x_ic, u_ic,
            x_bc, t_bc,
            w_data, w_phys, w_ic, w_bc,
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        # guardamos las métricas de esta época
        for k, v in comps.items():
            history[k].append(v)
        history["D"].append(model.D.item())
        history["rho"].append(model.rho.item())

        if epoch % log_every == 0 or epoch == 1:
            logging.info(
                f"[{label}] Época {epoch:6d}  loss={comps['total']:.4e}  "
                f"D={model.D.item():.5f}  rho={model.rho.item():.5f}"
            )

    # recogemos los valores finales estimados y calculamos el error
    D_est   = model.D.item()
    rho_est = model.rho.item()
    D_err   = abs(D_est   - data["D_true"])   / data["D_true"]  * 100
    rho_err = abs(rho_est - data["rho_true"]) / data["rho_true"] * 100

    logging.info(f"[{label}] -- Resultados finales --")
    logging.info(f"[{label}]  D_true={data['D_true']:.4f}  D_est={D_est:.4f}  error={D_err:.2f}%")
    logging.info(f"[{label}]  rho_true={data['rho_true']:.4f}  rho_est={rho_est:.4f}  error={rho_err:.2f}%")

    resultados = {
        "label":       label,
        "D_true":      data["D_true"],
        "rho_true":    data["rho_true"],
        "D_est":       D_est,
        "rho_est":     rho_est,
        "D_err_pct":   D_err,
        "rho_err_pct": rho_err,
        "final_loss":  history["total"][-1],
    }

    # guardamos el modelo entrenado y el histórico en disco
    torch.save(model.state_dict(), out_dir / f"{label}_model.pt")
    np.savez(out_dir / f"{label}_history.npz", **{k: np.array(v) for k, v in history.items()})

    return resultados


def plot_summary(all_results, out_dir):
    # extraemos labels y errores de todos los experimentos
    labels   = [r["label"] for r in all_results]
    D_errs   = [r["D_err_pct"] for r in all_results]
    rho_errs = [r["rho_err_pct"] for r in all_results]

    x = np.arange(len(labels))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 5))
    ax.bar(x - ancho / 2, D_errs,   ancho, label="Error D (%)",   alpha=0.8)
    ax.bar(x + ancho / 2, rho_errs, ancho, label="Error rho (%)", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Error relativo (%)")
    ax.set_title("Degradación del error de estimación con el nivel de ruido")
    ax.legend()
    ax.axhline(5,  color="green",  ls="--", lw=1, label="5%")
    ax.axhline(15, color="orange", ls="--", lw=1, label="15%")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "summary_errors.png", dpi=130, bbox_inches="tight")
    plt.close()


def main():
    # leemos la configuración completa del JSON
    with open("config.json") as f:
        cfg = json.load(f)

    pinn_cfg     = cfg["PINN"]
    configs      = cfg["CONFIGS"]
    noise_levels = cfg["NOISE_LEVELS"]
    data_folder  = cfg["OUTPUT_FOLDER"]

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    setup_logging(str(out_dir / "pinn_training.log"))
    logging.info("=" * 65)
    logging.info("PINN Fisher-Kolmogorov — Estimación de parámetros")
    logging.info(f"Dispositivo: {DEVICE}")
    logging.info("=" * 65)

    all_results = []

    # recorremos todas las combinaciones de configuración × nivel de ruido
    for cfg_info, noise in product(configs, noise_levels):
        cfg_label = cfg_info["label"]
        noise_tag = int(noise * 100)
        label     = f"cfg{cfg_label}_noise{noise_tag:02d}"

        npz_path = f"{data_folder}/config{cfg_label}_noise{noise_tag:02d}.npz"
        if not os.path.exists(npz_path):
            logging.warning(f"No encontrado: {npz_path} — saltando")
            continue

        logging.info("")
        logging.info("-" * 60)
        logging.info(f"Experimento: {label}")

        # cargamos los datos y creamos un modelo nuevo para cada experimento
        data = load_npz(npz_path, n_obs=pinn_cfg["n_obs"])

        model = PINN(
            layers   = pinn_cfg["layers"],
            D_init   = pinn_cfg["D_init"],
            rho_init = pinn_cfg["rho_init"],
            K        = pinn_cfg["K"],
            x_scale  = data["L"],
            t_scale  = data["T"],
        ).to(DEVICE)

        resultados = train(model, data, pinn_cfg, out_dir, label)
        all_results.append(resultados)

    # guardamos todos los resultados en un CSV
    if all_results:
        csv_path = out_dir / "resultados.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        logging.info(f"\nTabla de resultados guardada en: {csv_path}")

        plot_summary(all_results, out_dir)

    logging.info("")
    logging.info("=" * 65)
    logging.info("RESUMEN FINAL")
    logging.info("=" * 65)
    logging.info(f"{'Experimento':<30} {'D_err%':>8} {'rho_err%':>10}")
    logging.info("-" * 50)
    for r in all_results:
        logging.info(
            f"  {r['label']:<28} {r['D_err_pct']:>7.2f}%  {r['rho_err_pct']:>8.2f}%"
        )
    logging.info("=" * 65)


if __name__ == "__main__":
    main()
