"""
Experimento Comparativo: EvoSymbolic vs EvoSymbolicVNS
=======================================================
Compara las dos versiones de nuestra biblioteca sobre 15 datasets
sintéticos ordenados de simple a complejo.

Estructura esperada del proyecto:
    EvoSymbolic/
    ├── evosymbolic/        <- versión sin VNS
    ├── evosymbolicvns/     <- versión con VNS
    └── experiment_vns.py   <- este archivo

Métricas:
  R², RMSE, Tiempo (s), Complejidad (nº nodos), Mejoras VNS acumuladas

Salidas:
  Gráficas   : plot_vns_r2.png · plot_vns_rmse.png · plot_vns_time.png
               plot_vns_complexity.png · plot_vns_improvement.png
               plot_vns_radar.png
  CSVs       : results_vns_raw.csv · results_vns_metrics.csv
  Consola    : tablas detalladas por grupo de dificultad

Uso:
    python experiment_vns.py
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# RUTAS: añadir ambas carpetas al path
# ---------------------------------------------------------------------------

ROOT        = os.path.dirname(os.path.abspath(__file__))
PATH_BASE   = os.path.join(ROOT, "evosymbolic")
PATH_VNS    = os.path.join(ROOT, "evosymbolicvns")

if PATH_BASE not in sys.path:
    sys.path.insert(0, PATH_BASE)
if PATH_VNS not in sys.path:
    sys.path.insert(0, PATH_VNS)

# ---------------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
N_SAMPLES    = 150
TEST_SIZE    = 0.25

COLORS = {
    "EvoSymbolic"    : "#1D9E75",   # verde teal
    "EvoSymbolicVNS" : "#E0654F",   # coral
}

# Grupos de dificultad para el análisis
GROUPS = {
    "Simple"   : list(range(1,  6)),   # F01–F05
    "Media"    : list(range(6,  11)),  # F06–F10
    "Compleja" : list(range(11, 16)),  # F11–F15
}

# ---------------------------------------------------------------------------
# 1. DATASETS SINTÉTICOS (15 funciones, simple → compleja)
# ---------------------------------------------------------------------------

def make_datasets():
    np.random.seed(RANDOM_STATE)
    ds = {}

    # ── GRUPO 1: Simple (1 variable, estructura directa) ──────────────────
    X = np.random.uniform(-5, 5, (N_SAMPLES, 1))
    ds["F01: x²"] = (X, X[:,0]**2)

    X = np.random.uniform(-4, 4, (N_SAMPLES, 1))
    ds["F02: x²+sin(x)"] = (X, X[:,0]**2 + np.sin(X[:,0]))

    X = np.random.uniform(-3, 3, (N_SAMPLES, 1))
    ds["F03: x³-x"] = (X, X[:,0]**3 - X[:,0])

    X = np.random.uniform(0.1, 5, (N_SAMPLES, 1))
    ds["F04: sqrt(x)+log(x)"] = (X, np.sqrt(X[:,0]) + np.log(X[:,0]))

    X = np.random.uniform(-4, 4, (N_SAMPLES, 1))
    ds["F05: sin(x)·cos(x)"] = (X, np.sin(X[:,0]) * np.cos(X[:,0]))

    # ── GRUPO 2: Media (combinaciones, 1-2 variables) ─────────────────────
    X = np.random.uniform(-5, 5, (N_SAMPLES, 1))
    ds["F06: x·sin(x)+cos(x)"] = (X, X[:,0]*np.sin(X[:,0]) + np.cos(X[:,0]))

    X = np.random.uniform(-3, 3, (N_SAMPLES, 1))
    ds["F07: x³-x²+x-1"] = (X, X[:,0]**3 - X[:,0]**2 + X[:,0] - 1)

    X = np.random.uniform(-3, 3, (N_SAMPLES, 2))
    ds["F08: x0·x1+sin(x0)"] = (X, X[:,0]*X[:,1] + np.sin(X[:,0]))

    X = np.random.uniform(0.5, 4, (N_SAMPLES, 1))
    ds["F09: exp(x)/x"] = (X, np.exp(X[:,0]) / X[:,0])

    X = np.random.uniform(-4, 4, (N_SAMPLES, 1))
    ds["F10: x²·sin(x)"] = (X, X[:,0]**2 * np.sin(X[:,0]))

    # ── GRUPO 3: Compleja (multivar, alta no linealidad) ──────────────────
    X = np.random.uniform(-3, 3, (N_SAMPLES, 2))
    ds["F11: sin(x0)·cos(x1)+x0·x1"] = (
        X, np.sin(X[:,0])*np.cos(X[:,1]) + X[:,0]*X[:,1]
    )

    X = np.random.uniform(-3, 3, (N_SAMPLES, 2))
    ds["F12: x0²+x1²-x0·x1"] = (X, X[:,0]**2 + X[:,1]**2 - X[:,0]*X[:,1])

    X = np.random.uniform(-2, 2, (N_SAMPLES, 3))
    ds["F13: x0·x1+x2²-sin(x0)"] = (
        X, X[:,0]*X[:,1] + X[:,2]**2 - np.sin(X[:,0])
    )

    X = np.random.uniform(-3, 3, (N_SAMPLES, 1))
    ds["F14: sin(x²)+cos(x+1)"] = (
        X, np.sin(X[:,0]**2) + np.cos(X[:,0] + 1)
    )

    X = np.random.uniform(-2, 2, (N_SAMPLES, 2))
    ds["F15: (x0+x1)²·sin(x0-x1)"] = (
        X, (X[:,0]+X[:,1])**2 * np.sin(X[:,0]-X[:,1])
    )

    return ds

# ---------------------------------------------------------------------------
# 2. CONFIGURACIÓN DE CADA VERSIÓN
# ---------------------------------------------------------------------------

def build_regressor_base():
    """Importa SymbolicRegressor desde evosymbolic (sin VNS)."""
    # Guardar sys.path actual y poner evosymbolic primero
    original = sys.path.copy()
    sys.path = [PATH_BASE] + [p for p in sys.path if p != PATH_BASE and p != PATH_VNS]

    # Limpiar módulos cargados de la otra versión para evitar conflictos
    mods_to_remove = [k for k in sys.modules if k in (
        "sklearn_api", "motor_gp", "constant_optimization",
        "genetic_operators", "fitness", "expression_tree", "vns"
    )]
    for m in mods_to_remove:
        del sys.modules[m]

    from sklearn_api import SymbolicRegressor

    reg = SymbolicRegressor(
        population_size           = 200,
        generations               = 25,
        const_range               = (-5.0, 5.0),
        operators                 = ["add", "sub", "mul", "div"],
        functions                 = ["sin", "cos", "exp", "log", "sqrt"],
        parsimony_coeff           = 0.001,
        use_constant_optimization = True,
        local_search_every        = 5,
        local_search_top_k        = 5,
        random_state              = RANDOM_STATE,
        verbose                   = False,
    )

    sys.path = original
    return reg


def build_regressor_vns():
    """Importa SymbolicRegressor desde evosymbolicvns (con VNS)."""
    original = sys.path.copy()
    sys.path = [PATH_VNS] + [p for p in sys.path if p != PATH_BASE and p != PATH_VNS]

    mods_to_remove = [k for k in sys.modules if k in (
        "sklearn_api", "motor_gp", "constant_optimization",
        "genetic_operators", "fitness", "expression_tree", "vns"
    )]
    for m in mods_to_remove:
        del sys.modules[m]

    from sklearn_api import SymbolicRegressor

    reg = SymbolicRegressor(
        population_size           = 200,
        generations               = 25,
        const_range               = (-5.0, 5.0),
        operators                 = ["add", "sub", "mul", "div"],
        functions                 = ["sin", "cos", "exp", "log", "sqrt"],
        parsimony_coeff           = 0.001,
        use_vns                   = True,
        local_search_every        = 5,
        local_search_top_k        = 5,
        vns_every                 = 10,
        vns_top_k                 = 3,
        vns_k_max                 = 4,
        vns_iter                  = 5,
        vns_ls_maxiter            = 50,
        random_state              = RANDOM_STATE,
        verbose                   = False,
    )

    sys.path = original
    return reg

# ---------------------------------------------------------------------------
# 3. MOTOR DEL EXPERIMENTO
# ---------------------------------------------------------------------------

def run_experiment():
    datasets = make_datasets()
    records  = []

    builders = {
        "EvoSymbolic"    : build_regressor_base,
        "EvoSymbolicVNS" : build_regressor_vns,
    }

    # Determinar grupo de cada dataset
    ds_names = list(datasets.keys())
    def get_group(name):
        idx = ds_names.index(name) + 1
        if idx <= 5:  return "Simple"
        if idx <= 10: return "Media"
        return "Compleja"

    for ds_name, (X, y) in datasets.items():
        group = get_group(ds_name)
        print(f"\n{'='*65}")
        print(f"  [{group:8s}] {ds_name}  "
              f"({X.shape[0]} muestras, {X.shape[1]} variable/s)")
        print(f"{'='*65}")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        for lib_name, builder in builders.items():
            print(f"  [{lib_name:15s}] ", end="", flush=True)
            try:
                reg = builder()
                t0  = time.time()
                reg.fit(X_tr, y_tr)
                elapsed = time.time() - t0

                y_pred     = reg.predict(X_te)
                r2         = r2_score(y_te, y_pred)
                rmse       = float(np.sqrt(mean_squared_error(y_te, y_pred)))
                complexity = reg.get_complexity()["size"]
                expression = reg.get_expression()

                # Mejoras VNS (solo disponible en la versión VNS)
                vns_improvements = 0
                if lib_name == "EvoSymbolicVNS":
                    try:
                        vns_improvements = reg.engine_.n_vns_improved_
                    except AttributeError:
                        vns_improvements = 0

                print(f"R²={r2:.4f}  RMSE={rmse:.4f}  "
                      f"t={elapsed:.1f}s  size={complexity}  "
                      f"VNS_imp={vns_improvements}")

                records.append({
                    "dataset"         : ds_name,
                    "group"           : group,
                    "library"         : lib_name,
                    "r2"              : round(r2,      4),
                    "rmse"            : round(rmse,    4),
                    "time_s"          : round(elapsed, 2),
                    "complexity"      : complexity,
                    "vns_improvements": vns_improvements,
                    "expression"      : expression[:100],
                })

            except Exception as e:
                print(f"ERROR — {e}")
                records.append({
                    "dataset": ds_name, "group": group,
                    "library": lib_name,
                    "r2": None, "rmse": None, "time_s": None,
                    "complexity": None, "vns_improvements": 0,
                    "expression": f"Error: {str(e)[:80]}",
                })

    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# 4. TABLAS EN CONSOLA
# ---------------------------------------------------------------------------

def print_tables(df):
    valid = df[df["r2"].notna()].copy()
    libs  = ["EvoSymbolic", "EvoSymbolicVNS"]

    # ── Tabla 1: R² por dataset ───────────────────────────────────────────
    print("\n" + "="*70)
    print("  TABLA 1 — R² en test por dataset")
    print("  (mayor = mejor  ·  negrita indica ganador por fila)")
    print("="*70)

    pivot = valid.pivot_table(
        index="dataset", columns="library", values="r2", aggfunc="first"
    ).reindex(columns=libs)

    print(f"\n  {'Dataset':<35} {'EvoSymbolic':>13} {'EvoSymbolicVNS':>15}  Ganador")
    print(f"  {'─'*68}")
    for ds, row in pivot.iterrows():
        base = row.get("EvoSymbolic", None)
        vns  = row.get("EvoSymbolicVNS", None)
        if base is None or vns is None:
            continue
        winner = "VNS" if vns > base else ("BASE" if base > vns else "EMPATE")
        diff   = vns - base
        sign   = "+" if diff >= 0 else ""
        print(f"  {ds:<35} {base:>13.4f} {vns:>15.4f}  "
              f"{winner:6s} ({sign}{diff:.4f})")

    # ── Tabla 2: Resumen por grupo ────────────────────────────────────────
    print("\n" + "="*70)
    print("  TABLA 2 — Resumen por grupo de dificultad")
    print("="*70)
    print(f"\n  {'Grupo':<10} {'Métrica':<12} {'EvoSymbolic':>13} "
          f"{'EvoSymbolicVNS':>15} {'Diferencia':>12}")
    print(f"  {'─'*65}")

    for group in ["Simple", "Media", "Compleja"]:
        sub = valid[valid["group"] == group]
        for metric, label, fmt in [
            ("r2",         "R² medio",   "{:.4f}"),
            ("rmse",       "RMSE medio", "{:.4f}"),
            ("time_s",     "Tiempo (s)", "{:.2f}"),
            ("complexity", "Compl. med", "{:.1f}"),
        ]:
            b = sub[sub["library"]=="EvoSymbolic"][metric].mean()
            v = sub[sub["library"]=="EvoSymbolicVNS"][metric].mean()
            d = v - b
            sign = "+" if d >= 0 else ""
            print(f"  {group:<10} {label:<12} {fmt.format(b):>13} "
                  f"{fmt.format(v):>15} {sign+fmt.format(d):>12}")
        print(f"  {'─'*65}")

    # ── Tabla 3: Mejoras VNS acumuladas ───────────────────────────────────
    print("\n" + "="*70)
    print("  TABLA 3 — Mejoras VNS acumuladas por dataset")
    print("  (cuántas veces VNS encontró una mejora estructural)")
    print("="*70)
    vns_data = valid[valid["library"] == "EvoSymbolicVNS"][
        ["dataset", "group", "vns_improvements", "r2"]
    ].sort_values("vns_improvements", ascending=False)

    print(f"\n  {'Dataset':<35} {'Grupo':<10} {'Mejoras VNS':>12} {'R²':>8}")
    print(f"  {'─'*68}")
    for _, row in vns_data.iterrows():
        print(f"  {row['dataset']:<35} {row['group']:<10} "
              f"{int(row['vns_improvements']):>12} {row['r2']:>8.4f}")

    total_imp = vns_data["vns_improvements"].sum()
    print(f"\n  Total mejoras VNS en todos los datasets: {int(total_imp)}")

    # ── Tabla 4: Análisis global ──────────────────────────────────────────
    print("\n" + "="*70)
    print("  TABLA 4 — Comparativa global (media sobre los 15 datasets)")
    print("="*70)
    print(f"\n  {'Métrica':<20} {'EvoSymbolic':>13} {'EvoSymbolicVNS':>15} "
          f"{'Diferencia':>12} {'Ventaja':>10}")
    print(f"  {'─'*72}")
    for metric, label, fmt, higher_is_better in [
        ("r2",         "R² medio",        "{:.4f}", True),
        ("rmse",       "RMSE medio",      "{:.4f}", False),
        ("time_s",     "Tiempo medio (s)","_{:.2f}", False),
        ("complexity", "Complejidad med", "{:.1f}", False),
    ]:
        b = valid[valid["library"]=="EvoSymbolic"][metric].mean()
        v = valid[valid["library"]=="EvoSymbolicVNS"][metric].mean()
        d = v - b
        sign = "+" if d >= 0 else ""
        if higher_is_better:
            ventaja = "VNS" if v > b else ("BASE" if b > v else "EMPATE")
        else:
            ventaja = "VNS" if v < b else ("BASE" if b < v else "EMPATE")
        fmt_clean = fmt.replace("_", "")
        print(f"  {label:<20} {fmt_clean.format(b):>13} "
              f"{fmt_clean.format(v):>15} {sign+fmt_clean.format(d):>12} "
              f"{ventaja:>10}")

# ---------------------------------------------------------------------------
# 5. GRÁFICAS
# ---------------------------------------------------------------------------

def plot_all(df):
    valid  = df[df["r2"].notna()].copy()
    libs   = ["EvoSymbolic", "EvoSymbolicVNS"]
    dsets  = list(df["dataset"].unique())
    n      = len(dsets)
    x      = np.arange(n)
    w      = 0.38

    def vals(metric, lib):
        out = []
        for ds in dsets:
            sub = valid[(valid["library"]==lib) & (valid["dataset"]==ds)]
            out.append(float(sub[metric].values[0]) if not sub.empty else 0.0)
        return np.array(out)

    # Líneas de separación entre grupos (después de F05 y F10)
    group_lines = [4.5, 9.5]
    group_labels = [(2, "Simple"), (7, "Media"), (12, "Compleja")]

    def add_group_lines(ax):
        for xl in group_lines:
            ax.axvline(xl, color="gray", lw=0.8, ls="--", alpha=0.5)
        ymin, ymax = ax.get_ylim()
        for xi, label in group_labels:
            ax.text(xi, ymax * 0.97, label, ha="center", va="top",
                    fontsize=8, color="gray",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # ── Plot 1: R² ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    b_vals = vals("r2", "EvoSymbolic")
    v_vals = vals("r2", "EvoSymbolicVNS")
    ax.bar(x - w/2, b_vals, w, label="EvoSymbolic",
           color=COLORS["EvoSymbolic"], alpha=0.85)
    ax.bar(x + w/2, v_vals, w, label="EvoSymbolicVNS",
           color=COLORS["EvoSymbolicVNS"], alpha=0.85)
    for i, (b, v) in enumerate(zip(b_vals, v_vals)):
        ax.text(i - w/2, b + 0.005, f"{b:.3f}", ha="center",
                va="bottom", fontsize=6.5, rotation=90)
        ax.text(i + w/2, v + 0.005, f"{v:.3f}", ha="center",
                va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(dsets, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("R²  (mayor = mejor)", fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.axhline(1.0, color="gray", lw=0.7, ls="--")
    ax.set_title("R² en test — EvoSymbolic vs EvoSymbolicVNS\n"
                  "(15 datasets ordenados de simple a complejo)",
                  fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    add_group_lines(ax)
    fig.tight_layout()
    fig.savefig("plot_vns_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 2: RMSE ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    b_vals = vals("rmse", "EvoSymbolic")
    v_vals = vals("rmse", "EvoSymbolicVNS")
    ax.bar(x - w/2, b_vals, w, label="EvoSymbolic",
           color=COLORS["EvoSymbolic"], alpha=0.85)
    ax.bar(x + w/2, v_vals, w, label="EvoSymbolicVNS",
           color=COLORS["EvoSymbolicVNS"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(dsets, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("RMSE  (menor = mejor)", fontsize=10)
    ax.set_title("RMSE en test — EvoSymbolic vs EvoSymbolicVNS",
                  fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    add_group_lines(ax)
    fig.tight_layout()
    fig.savefig("plot_vns_rmse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 3: Tiempo ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    b_vals = vals("time_s", "EvoSymbolic")
    v_vals = vals("time_s", "EvoSymbolicVNS")
    ax.bar(x - w/2, b_vals, w, label="EvoSymbolic",
           color=COLORS["EvoSymbolic"], alpha=0.85)
    ax.bar(x + w/2, v_vals, w, label="EvoSymbolicVNS",
           color=COLORS["EvoSymbolicVNS"], alpha=0.85)
    for i, (b, v) in enumerate(zip(b_vals, v_vals)):
        ax.text(i - w/2, b + 0.05, f"{b:.1f}s", ha="center",
                va="bottom", fontsize=6.5, rotation=90)
        ax.text(i + w/2, v + 0.05, f"{v:.1f}s", ha="center",
                va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(dsets, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Tiempo de entrenamiento (s)", fontsize=10)
    ax.set_title("Tiempo de entrenamiento — EvoSymbolic vs EvoSymbolicVNS",
                  fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    add_group_lines(ax)
    fig.tight_layout()
    fig.savefig("plot_vns_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 4: Complejidad ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    b_vals = vals("complexity", "EvoSymbolic")
    v_vals = vals("complexity", "EvoSymbolicVNS")
    ax.bar(x - w/2, b_vals, w, label="EvoSymbolic",
           color=COLORS["EvoSymbolic"], alpha=0.85)
    ax.bar(x + w/2, v_vals, w, label="EvoSymbolicVNS",
           color=COLORS["EvoSymbolicVNS"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(dsets, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Complejidad — nº nodos del árbol", fontsize=10)
    ax.set_title("Complejidad de la expresión encontrada",
                  fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    add_group_lines(ax)
    fig.tight_layout()
    fig.savefig("plot_vns_complexity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 5: Mejoras VNS acumuladas ────────────────────────────────────
    vns_sub = valid[valid["library"] == "EvoSymbolicVNS"].set_index("dataset")
    imp_vals = np.array([
        int(vns_sub.loc[ds, "vns_improvements"]) if ds in vns_sub.index else 0
        for ds in dsets
    ])
    r2_diff = vals("r2", "EvoSymbolicVNS") - vals("r2", "EvoSymbolic")

    fig, ax1 = plt.subplots(figsize=(16, 5))
    ax2 = ax1.twinx()

    bars = ax1.bar(x, imp_vals, 0.6, color=COLORS["EvoSymbolicVNS"],
                    alpha=0.75, label="Mejoras VNS acumuladas")
    ax2.plot(x, r2_diff, "o-", color="#333333", lw=1.5,
              ms=5, label="ΔR² (VNS − Base)", zorder=3)
    ax2.axhline(0, color="gray", lw=0.7, ls="--")

    ax1.set_xticks(x)
    ax1.set_xticklabels(dsets, rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("Mejoras VNS acumuladas", fontsize=10,
                    color=COLORS["EvoSymbolicVNS"])
    ax2.set_ylabel("ΔR² = VNS − Base  (>0 VNS gana)", fontsize=10)
    ax1.set_title("Mejoras VNS acumuladas y su impacto en R²",
                   fontsize=11, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9)
    ax1.grid(axis="y", alpha=0.2)
    add_group_lines(ax1)
    fig.tight_layout()
    fig.savefig("plot_vns_improvement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 6: Radar de promedios por grupo ──────────────────────────────
    groups      = ["Simple", "Media", "Compleja"]
    dim_labels  = ["R²", "1−RMSE\nnorm.", "1−Tiempo\nnorm.", "1−Compl.\nnorm."]

    def norm_high(col, sub_df):
        mn, mx = valid[col].min(), valid[col].max()
        v = sub_df[col].mean()
        return (v - mn) / (mx - mn + 1e-10)

    def norm_low(col, sub_df):
        mn, mx = valid[col].min(), valid[col].max()
        v = sub_df[col].mean()
        return 1 - (v - mn) / (mx - mn + 1e-10)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, group in zip(axes, groups):
        sub_g = valid[valid["group"] == group]
        x_pos = np.arange(len(dim_labels))
        for lib in libs:
            sub_l = sub_g[sub_g["library"] == lib]
            profile = [
                norm_high("r2",         sub_l),
                norm_low("rmse",        sub_l),
                norm_low("time_s",      sub_l),
                norm_low("complexity",  sub_l),
            ]
            ax.bar(x_pos + (0 if lib == "EvoSymbolic" else 0.35),
                    profile, 0.35,
                    label=lib, color=COLORS[lib], alpha=0.85)

        ax.set_xticks(x_pos + 0.175)
        ax.set_xticklabels(dim_labels, fontsize=8)
        ax.set_ylim(0, 1.2)
        ax.axhline(1.0, color="gray", lw=0.6, ls="--")
        ax.set_title(f"Grupo: {group}", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    axes[0].set_ylabel("Puntuación normalizada (1.0 = mejor)", fontsize=9)
    handles = [mpatches.Patch(color=COLORS[l], label=l) for l in libs]
    fig.legend(handles=handles, loc="upper center", ncol=2,
                frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Perfil de rendimiento por grupo de dificultad",
                  fontsize=12, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig("plot_vns_radar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n  Gráficas guardadas:")
    for f in ["plot_vns_r2.png", "plot_vns_rmse.png", "plot_vns_time.png",
              "plot_vns_complexity.png", "plot_vns_improvement.png",
              "plot_vns_radar.png"]:
        print(f"    · {f}")

# ---------------------------------------------------------------------------
# 6. GUARDAR CSVs
# ---------------------------------------------------------------------------

def save_csvs(df):
    df.to_csv("results_vns_raw.csv", index=False)

    valid = df[df["r2"].notna()].copy()
    rows  = []
    for ds in df["dataset"].unique():
        for lib in ["EvoSymbolic", "EvoSymbolicVNS"]:
            sub = valid[(valid["dataset"]==ds) & (valid["library"]==lib)]
            if sub.empty: continue
            r = sub.iloc[0]
            rows.append({
                "Dataset"         : ds,
                "Grupo"           : r["group"],
                "Biblioteca"      : lib,
                "R2"              : r["r2"],
                "RMSE"            : r["rmse"],
                "Tiempo_s"        : r["time_s"],
                "Complejidad"     : r["complexity"],
                "Mejoras_VNS"     : r["vns_improvements"],
                "Expresion"       : r["expression"],
            })
    pd.DataFrame(rows).to_csv("results_vns_metrics.csv", index=False)

    print("\n  CSVs guardados:")
    print("    · results_vns_raw.csv")
    print("    · results_vns_metrics.csv")

# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Experimento: EvoSymbolic vs EvoSymbolicVNS")
    print(f"  Datasets    : 15 funciones (Simple / Media / Compleja)")
    print(f"  Muestras    : {N_SAMPLES} por dataset  |  Split: "
          f"{int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} %")
    print(f"  Semilla     : {RANDOM_STATE}")
    print(f"  Raíz        : {ROOT}")
    print("=" * 65)

    df = run_experiment()

    print_tables(df)
    plot_all(df)
    save_csvs(df)

    print("\n" + "="*65)
    print("  Experimento finalizado.")
    print("  Graficas  : plot_vns_r2 · plot_vns_rmse · plot_vns_time ·")
    print("              plot_vns_complexity · plot_vns_improvement · plot_vns_radar")
    print("  Tablas    : results_vns_raw · results_vns_metrics")
    print("="*65)
