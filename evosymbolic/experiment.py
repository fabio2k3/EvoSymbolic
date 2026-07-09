"""
Experimento Comparativo de Regresión Simbólica — versión extendida
===================================================================
Compara 4 bibliotecas sobre 5 datasets sintéticos:
  - Nuestra biblioteca  (SymbolicRegressor)
  - gplearn             (pip install gplearn)
  - PySR                (pip install pysr + python -m pysr install)
  - DEAP                (pip install deap)

Análisis incluidos:
  1. Métricas de calidad     : R², RMSE, MAE
  2. Métricas de eficiencia  : tiempo de entrenamiento (s)
  3. Métricas de parsimonia  : complejidad (nº nodos), profundidad
  4. Métricas de robustez    : error porcentual medio (MAPE)
  5. Ranking global ponderado por las 4 dimensiones
  6. Tabla de expresiones encontradas vs expresión objetivo
  7. Resumen estadístico     : media ± desviación estándar
  8. 6 gráficas detalladas
  9. 3 CSVs estructurados
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
N_SAMPLES    = 150
TEST_SIZE    = 0.25

COLORS = {
    "Nuestra" : "#E0654F",
    "gplearn" : "#1D9E75",
    "PySR"    : "#7B61FF",
    "DEAP"    : "#C97A1D",
}

DATASETS_TARGETS = {
    "F1: x²"              : "x²",
    "F2: x²+sin(x)"       : "x² + sin(x)",
    "F3: x·sin(x)+cos(x)" : "x·sin(x) + cos(x)",
    "F4: x0·x1+sin(x0)"   : "x0·x1 + sin(x0)",
    "F5: x³-x²+x-1"       : "x³ - x² + x - 1",
}

# ---------------------------------------------------------------------------
# 1. DATASETS
# ---------------------------------------------------------------------------

def make_datasets():
    np.random.seed(RANDOM_STATE)
    datasets = {}

    X = np.random.uniform(-5, 5, (N_SAMPLES, 1))
    datasets["F1: x²"] = (X, X[:, 0] ** 2)

    X = np.random.uniform(-4, 4, (N_SAMPLES, 1))
    datasets["F2: x²+sin(x)"] = (X, X[:, 0] ** 2 + np.sin(X[:, 0]))

    X = np.random.uniform(-5, 5, (N_SAMPLES, 1))
    datasets["F3: x·sin(x)+cos(x)"] = (X, X[:, 0] * np.sin(X[:, 0]) + np.cos(X[:, 0]))

    X = np.random.uniform(-3, 3, (N_SAMPLES, 2))
    datasets["F4: x0·x1+sin(x0)"] = (X, X[:, 0] * X[:, 1] + np.sin(X[:, 0]))

    X = np.random.uniform(-3, 3, (N_SAMPLES, 1))
    datasets["F5: x³-x²+x-1"] = (X, X[:, 0]**3 - X[:, 0]**2 + X[:, 0] - 1)

    return datasets


# ---------------------------------------------------------------------------
# 2. WRAPPERS
# ---------------------------------------------------------------------------

def run_ours(X_train, y_train, X_test, n_features):
    from sklearn_api import SymbolicRegressor
    reg = SymbolicRegressor(
        population_size           = 300,
        generations               = 30,
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
    reg.fit(X_train, y_train)
    y_pred     = reg.predict(X_test)
    complexity = reg.get_complexity()["size"]
    depth      = reg.get_complexity()["depth"]
    expression = reg.get_expression()
    return y_pred, complexity, depth, expression


def run_gplearn(X_train, y_train, X_test, n_features):
    from gplearn.genetic import SymbolicRegressor as GPLearnSR
    reg = GPLearnSR(
        population_size       = 300,
        generations           = 30,
        tournament_size       = 3,
        function_set          = ("add", "sub", "mul", "div", "sin", "cos", "sqrt", "log"),
        parsimony_coefficient = 0.001,
        random_state          = RANDOM_STATE,
        verbose               = 0,
    )
    reg.fit(X_train, y_train)
    y_pred     = reg.predict(X_test)
    complexity = reg._program.length_
    depth      = reg._program.depth_
    expression = str(reg._program)
    return y_pred, complexity, depth, expression


def run_pysr(X_train, y_train, X_test, n_features):
    from pysr import PySRRegressor
    reg = PySRRegressor(
        niterations      = 30,
        binary_operators = ["+", "-", "*", "/"],
        unary_operators  = ["sin", "cos", "exp", "sqrt", "log"],
        random_state     = RANDOM_STATE,
        verbosity        = 0,
        progress         = False,
    )
    reg.fit(X_train, y_train)
    y_pred     = reg.predict(X_test)
    best       = reg.get_best()
    complexity = int(best["complexity"])
    depth      = complexity // 2
    expression = str(best["equation"])
    return y_pred, complexity, depth, expression


def run_deap(X_train, y_train, X_test, n_features):
    import operator, math, random
    from deap import algorithms, base, creator, gp, tools

    pset = gp.PrimitiveSet("MAIN", n_features)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(lambda a, b: a / b if abs(b) > 1e-10 else 1.0, 2, name="div")
    pset.addPrimitive(math.sin,  1)
    pset.addPrimitive(math.cos,  1)
    pset.addPrimitive(lambda x: math.sqrt(abs(x)), 1, name="sqrt")
    pset.addPrimitive(lambda x: math.log(abs(x) + 1e-10), 1, name="log")
    pset.addEphemeralConstant("rand", lambda: np.random.uniform(-5, 5))
    for i in range(n_features):
        pset.renameArguments(**{f"ARG{i}": f"x{i}"})

    for attr in ("FitnessMin", "Individual"):
        if hasattr(creator, attr):
            delattr(creator, attr)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr",       gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile",    gp.compile, pset=pset)

    def eval_ind(ind):
        func = toolbox.compile(expr=ind)
        try:
            preds = np.array([func(*row) for row in X_train])
            if not np.all(np.isfinite(preds)):
                return (1e10,)
            return (float(np.mean((y_train - preds) ** 2)),)
        except Exception:
            return (1e10,)

    toolbox.register("evaluate", eval_ind)
    toolbox.register("select",   tools.selTournament, tournsize=3)
    toolbox.register("mate",     gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate",   gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate",   gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

    backup = np.random.get_state()
    np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.1,
                        ngen=30, halloffame=hof, verbose=False)
    np.random.set_state(backup)

    best_ind   = hof[0]
    func       = toolbox.compile(expr=best_ind)
    y_pred     = np.array([func(*row) for row in X_test])
    y_pred     = np.where(np.isfinite(y_pred), y_pred, 0.0)
    complexity = len(best_ind)
    depth      = best_ind.height
    expression = str(best_ind)
    return y_pred, complexity, depth, expression


LIBRARIES = {
    "Nuestra" : run_ours,
    "gplearn" : run_gplearn,
    "PySR"    : run_pysr,
    "DEAP"    : run_deap,
}


# ---------------------------------------------------------------------------
# 3. MÉTRICA AUXILIAR
# ---------------------------------------------------------------------------

def mape(y_true, y_pred, eps=1e-8):
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------------------------------------------------------------------------
# 4. MOTOR DEL EXPERIMENTO
# ---------------------------------------------------------------------------

def run_experiment():
    datasets = make_datasets()
    records  = []

    for ds_name, (X, y) in datasets.items():
        print(f"\n{'='*65}")
        print(f"  Dataset: {ds_name}  ({X.shape[0]} muestras, {X.shape[1]} variable/s)")
        print(f"{'='*65}")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        for lib_name, run_fn in LIBRARIES.items():
            print(f"  [{lib_name:8s}] ", end="", flush=True)
            try:
                t0 = time.time()
                y_pred, complexity, depth, expression = run_fn(
                    X_tr, y_tr, X_te, X.shape[1]
                )
                elapsed = time.time() - t0

                r2   = r2_score(y_te, y_pred)
                rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
                mae  = float(mean_absolute_error(y_te, y_pred))
                mp   = mape(y_te, y_pred)

                print(f"R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  "
                      f"MAPE={mp:.2f}%  t={elapsed:.1f}s  "
                      f"size={complexity}  depth={depth}")

                records.append({
                    "dataset"   : ds_name,
                    "library"   : lib_name,
                    "r2"        : round(r2,   4),
                    "rmse"      : round(rmse, 4),
                    "mae"       : round(mae,  4),
                    "mape"      : round(mp,   2) if not np.isnan(mp) else None,
                    "time_s"    : round(elapsed, 2),
                    "complexity": complexity,
                    "depth"     : depth,
                    "expression": expression[:100],
                    "target"    : DATASETS_TARGETS.get(ds_name, "?"),
                })

            except ImportError as e:
                print(f"NO INSTALADA — {e}")
                records.append({
                    "dataset": ds_name, "library": lib_name,
                    "r2": None, "rmse": None, "mae": None, "mape": None,
                    "time_s": None, "complexity": None, "depth": None,
                    "expression": "ImportError",
                    "target": DATASETS_TARGETS.get(ds_name, "?"),
                })
            except Exception as e:
                print(f"ERROR — {e}")
                records.append({
                    "dataset": ds_name, "library": lib_name,
                    "r2": None, "rmse": None, "mae": None, "mape": None,
                    "time_s": None, "complexity": None, "depth": None,
                    "expression": f"Error: {str(e)[:80]}",
                    "target": DATASETS_TARGETS.get(ds_name, "?"),
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. TABLAS EN CONSOLA
# ---------------------------------------------------------------------------

def print_tables(df):
    libs  = [l for l in LIBRARIES if l in df["library"].unique()]
    valid = df[df["r2"].notna()].copy()

    # ── Tabla 1: Calidad del ajuste ─────────────────────────────────────────
    print("\n" + "="*70)
    print("  TABLA 1 — Calidad del ajuste")
    print("  (R² mayor=mejor · RMSE/MAE/MAPE menor=mejor)")
    print("="*70)
    for metric, label, fmt in [
        ("r2",   "R²        (mayor es mejor)", "{:.4f}"),
        ("rmse", "RMSE      (menor es mejor)", "{:.4f}"),
        ("mae",  "MAE       (menor es mejor)", "{:.4f}"),
        ("mape", "MAPE (%)  (menor es mejor)", "{:.2f}"),
    ]:
        print(f"\n  {label}")
        pivot = valid.pivot_table(
            index="dataset", columns="library", values=metric, aggfunc="first"
        )
        pivot = pivot.reindex(columns=[l for l in libs if l in pivot.columns])
        print(pivot.to_string(float_format=lambda x: fmt.format(x)))

    # ── Tabla 2: Eficiencia y parsimonia ────────────────────────────────────
    print("\n" + "="*70)
    print("  TABLA 2 — Eficiencia computacional y parsimonia")
    print("  (todos: menor es mejor)")
    print("="*70)
    for metric, label, fmt in [
        ("time_s",    "Tiempo (s)",         "{:.2f}"),
        ("complexity","Complejidad (nodos)", "{:.0f}"),
        ("depth",     "Profundidad",         "{:.0f}"),
    ]:
        print(f"\n  {label}")
        pivot = valid.pivot_table(
            index="dataset", columns="library", values=metric, aggfunc="first"
        )
        pivot = pivot.reindex(columns=[l for l in libs if l in pivot.columns])
        print(pivot.to_string(float_format=lambda x: fmt.format(x)))

    # ── Tabla 3: Ranking global ponderado ───────────────────────────────────
    print("\n" + "="*70)
    print("  TABLA 3 — Ranking global ponderado")
    print("  Pesos: R²=40%  RMSE=20%  Tiempo=20%  Complejidad=20%")
    print("="*70)

    def norm_high(series, lib):
        mn, mx = series.min(), series.max()
        return (series[lib] - mn) / (mx - mn + 1e-10)

    def norm_low(series, lib):
        mn, mx = series.min(), series.max()
        return 1 - (series[lib] - mn) / (mx - mn + 1e-10)

    scores = {}
    all_r2   = valid.groupby("library")["r2"].mean()
    all_rmse = valid.groupby("library")["rmse"].mean()
    all_time = valid.groupby("library")["time_s"].mean()
    all_comp = valid.groupby("library")["complexity"].mean()

    for lib in libs:
        if lib not in all_r2.index:
            continue
        scores[lib] = round(
            0.40 * norm_high(all_r2,   lib) +
            0.20 * norm_low(all_rmse,  lib) +
            0.20 * norm_low(all_time,  lib) +
            0.20 * norm_low(all_comp,  lib), 4
        )

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    medals  = ["1er", "2do", "3er", "4to"]
    print()
    for pos, (lib, score) in enumerate(ranking):
        bar = "█" * int(score * 35)
        print(f"  {medals[pos]}  {lib:10s}  {bar:<35s}  {score:.4f}")

    # ── Tabla 4: Expresiones encontradas ────────────────────────────────────
    print("\n" + "="*70)
    print("  TABLA 4 — Expresiones encontradas vs objetivo real")
    print("="*70)
    for ds in df["dataset"].unique():
        target = DATASETS_TARGETS.get(ds, "?")
        print(f"\n  {ds}")
        print(f"  Objetivo : {target}")
        print(f"  {'─'*62}")
        sub = df[df["dataset"] == ds]
        for _, row in sub.iterrows():
            expr = str(row["expression"])
            if "Error" in expr or "Import" in expr:
                continue
            if len(expr) > 85:
                expr = expr[:82] + "..."
            r2_tag = f"R²={row['r2']:.4f}" if row["r2"] is not None else "R²=N/A"
            print(f"  [{row['library']:8s}] {r2_tag}  ->  {expr}")

    # ── Tabla 5: Resumen estadístico ────────────────────────────────────────
    print("\n" + "="*70)
    print("  TABLA 5 — Resumen estadístico (media ± desv. std entre datasets)")
    print("="*70)
    print(f"\n  {'Biblioteca':<12} {'R² medio':>12} {'RMSE medio':>13} "
          f"{'Tiempo medio':>14} {'Complejidad':>13}")
    print(f"  {'─'*66}")
    for lib in libs:
        sub = valid[valid["library"] == lib]
        if sub.empty:
            continue
        print(
            f"  {lib:<12} "
            f"{sub['r2'].mean():>7.4f}±{sub['r2'].std():.4f}  "
            f"{sub['rmse'].mean():>8.4f}±{sub['rmse'].std():.4f}  "
            f"{sub['time_s'].mean():>8.2f}±{sub['time_s'].std():.2f}s  "
            f"{sub['complexity'].mean():>7.1f}±{sub['complexity'].std():.1f}"
        )


# ---------------------------------------------------------------------------
# 6. GRÁFICAS (6 plots)
# ---------------------------------------------------------------------------

def plot_all(df):
    valid = df[df["r2"].notna()].copy()
    libs  = [l for l in LIBRARIES if l in valid["library"].unique()]
    dsets = list(df["dataset"].unique())
    x     = np.arange(len(dsets))
    w     = 0.8 / len(libs)

    def bar_vals(metric, lib):
        out = []
        for ds in dsets:
            sub = valid[(valid["library"] == lib) & (valid["dataset"] == ds)]
            out.append(float(sub[metric].values[0]) if not sub.empty else 0.0)
        return out

    def draw_bars(ax, metric, libs, ylabel, title, annotate_fmt="{:.3f}", ylim=None):
        for i, lib in enumerate(libs):
            vals   = bar_vals(metric, lib)
            offset = (i - len(libs)/2 + 0.5) * w
            bars   = ax.bar(x + offset, vals, w * 0.88,
                             label=lib, color=COLORS.get(lib, "#999"), alpha=0.88)
            for bar, v in zip(bars, vals):
                if v > 0.001:
                    ax.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + (0.005 if ylim else 0.05),
                             annotate_fmt.format(v),
                             ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(dsets, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    # ── Plot 1: R² ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    draw_bars(ax, "r2", libs, "R²  (mayor = mejor)",
              "R² en test por dataset y biblioteca",
              annotate_fmt="{:.3f}", ylim=(0, 1.14))
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    fig.tight_layout()
    fig.savefig("plot_r2.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── Plot 2: RMSE ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    draw_bars(ax, "rmse", libs, "RMSE  (menor = mejor)",
              "RMSE en test por dataset y biblioteca",
              annotate_fmt="{:.3f}")
    fig.tight_layout()
    fig.savefig("plot_rmse.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── Plot 3: Tiempo ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    draw_bars(ax, "time_s", libs, "Tiempo (s)  (menor = mejor)",
              "Tiempo de entrenamiento por dataset y biblioteca",
              annotate_fmt="{:.1f}s")
    fig.tight_layout()
    fig.savefig("plot_time.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── Plot 4: R² vs Complejidad (scatter anotado) ─────────────────────────
    fig, ax = plt.subplots(figsize=(11, 7))
    for lib in libs:
        sub = valid[valid["library"] == lib]
        ax.scatter(sub["complexity"], sub["r2"],
                   label=lib, color=COLORS.get(lib, "#999"),
                   s=130, alpha=0.9, edgecolors="white", linewidths=1.2, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['dataset'].split(':')[0]}  t={row['time_s']:.1f}s",
                (row["complexity"], row["r2"]),
                textcoords="offset points", xytext=(8, 4),
                fontsize=7.5, color=COLORS.get(lib, "#999"),
            )
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", label="R²=1.0 (perfecto)")
    ax.set_xlabel("Complejidad — nº nodos del árbol  (menor = más simple)", fontsize=10)
    ax.set_ylabel("R² en test  (mayor = mejor)", fontsize=10)
    ax.set_title("Trade-off: Calidad vs Complejidad\n"
                  "(etiquetas incluyen tiempo de entrenamiento)",
                  fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig("plot_r2_vs_size.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── Plot 5: Perfil de rendimiento por biblioteca ─────────────────────────
    dim_labels = ["R²\n(calidad)", "1-RMSE\nnorm.", "1-Tiempo\nnorm.", "1-Complejidad\nnorm."]

    def norm_high(col):
        mn, mx = valid[col].min(), valid[col].max()
        return (valid.groupby("library")[col].mean() - mn) / (mx - mn + 1e-10)

    def norm_low(col):
        mn, mx = valid[col].min(), valid[col].max()
        s = valid.groupby("library")[col].mean()
        return 1 - (s - mn) / (mx - mn + 1e-10)

    profiles = pd.DataFrame({
        "R²"           : norm_high("r2"),
        "1-RMSE"       : norm_low("rmse"),
        "1-Tiempo"     : norm_low("time_s"),
        "1-Complejidad": norm_low("complexity"),
    })

    fig, axes = plt.subplots(1, len(libs), figsize=(14, 4.5), sharey=True)
    for ax, lib in zip(axes, libs):
        if lib not in profiles.index:
            ax.set_title(lib); continue
        vals = profiles.loc[lib].values
        bars = ax.bar(dim_labels, vals, color=COLORS.get(lib, "#999"), alpha=0.85, width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        ax.set_ylim(0, 1.25)
        ax.set_title(lib, fontsize=12, fontweight="bold", color=COLORS.get(lib, "#333"))
        ax.axhline(1.0, color="gray", lw=0.6, ls="--")
        ax.grid(axis="y", alpha=0.2)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Perfil de rendimiento por biblioteca\n"
                  "(1.0 = mejor en esa dimensión entre las 4 bibliotecas)",
                  fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig("plot_profiles.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── Plot 6: Ranking global (barras horizontales) ─────────────────────────
    scores_series = (
        0.40 * norm_high("r2") +
        0.20 * norm_low("rmse") +
        0.20 * norm_low("time_s") +
        0.20 * norm_low("complexity")
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    bar_colors = [COLORS.get(l, "#999") for l in scores_series.index]
    bars = ax.barh(scores_series.index, scores_series.values,
                    color=bar_colors, alpha=0.88, height=0.45)
    for bar, (lib, v) in zip(bars, scores_series.items()):
        ax.text(v + 0.008, bar.get_y() + bar.get_height()/2,
                 f"{v:.4f}", va="center", fontsize=11, fontweight="bold",
                 color=COLORS.get(lib, "#333"))
    ax.set_xlim(0, 1.18)
    ax.set_xlabel(
        "Puntuación global ponderada\n"
        "R²×40%  +  (1/RMSE)×20%  +  (1/Tiempo)×20%  +  (1/Complejidad)×20%",
        fontsize=9
    )
    ax.set_title("Ranking global de bibliotecas", fontsize=12, fontweight="bold")
    ax.axvline(1.0, color="gray", lw=0.7, ls="--")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig("plot_ranking.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    print("\n  Graficas guardadas:")
    for f in ["plot_r2.png", "plot_rmse.png", "plot_time.png",
              "plot_r2_vs_size.png", "plot_profiles.png", "plot_ranking.png"]:
        print(f"    * {f}")


# ---------------------------------------------------------------------------
# 7. GUARDAR CSVs
# ---------------------------------------------------------------------------

def save_csvs(df):
    valid = df[df["r2"].notna()].copy()
    libs  = [l for l in LIBRARIES if l in valid["library"].unique()]

    # Raw completo
    df.to_csv("results_raw.csv", index=False)

    # Métricas cruzadas
    rows = []
    for ds in df["dataset"].unique():
        for lib in libs:
            sub = valid[(valid["dataset"] == ds) & (valid["library"] == lib)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append({
                "Dataset"      : ds,
                "Biblioteca"   : lib,
                "R2"           : r["r2"],
                "RMSE"         : r["rmse"],
                "MAE"          : r["mae"],
                "MAPE_pct"     : r["mape"],
                "Tiempo_s"     : r["time_s"],
                "Complejidad"  : r["complexity"],
                "Profundidad"  : r["depth"],
            })
    pd.DataFrame(rows).to_csv("results_metrics.csv", index=False)

    # Expresiones
    rows_e = []
    for ds in df["dataset"].unique():
        for lib in libs:
            sub = df[(df["dataset"] == ds) & (df["library"] == lib)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows_e.append({
                "Dataset"            : ds,
                "Objetivo"           : r["target"],
                "Biblioteca"         : lib,
                "Expresion_encontrada": r["expression"],
                "R2"                 : r["r2"],
            })
    pd.DataFrame(rows_e).to_csv("results_expressions.csv", index=False)

    print("\n  CSVs guardados:")
    for f in ["results_raw.csv", "results_metrics.csv", "results_expressions.csv"]:
        print(f"    * {f}")


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Experimento Comparativo de Regresion Simbolica")
    print(f"  Bibliotecas : {', '.join(LIBRARIES)}")
    print(f"  Datasets    : 5 funciones sinteticas ({N_SAMPLES} muestras c/u)")
    print(f"  Split       : {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} % train/test")
    print(f"  Semilla     : {RANDOM_STATE}")
    print("=" * 65)

    df = run_experiment()

    print_tables(df)
    plot_all(df)
    save_csvs(df)

    print("\n" + "="*65)
    print("  Experimento finalizado.")
    print("  Graficas : plot_r2 · plot_rmse · plot_time ·")
    print("             plot_r2_vs_size · plot_profiles · plot_ranking")
    print("  Tablas   : results_raw · results_metrics · results_expressions")
    print("="*65)