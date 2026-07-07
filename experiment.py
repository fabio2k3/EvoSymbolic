"""
Experimento Comparativo de Regresión Simbólica
================================================
Compara 4 bibliotecas sobre 5 datasets sintéticos:
  - Nuestra biblioteca   (SymbolicRegressor)
  - gplearn              (pip install gplearn)
  - PySR                 (pip install pysr)
  - DEAP                 (pip install deap)

Métricas:  R² test · RMSE test · tiempo (s) · complejidad (nº nodos)

Uso:
  1. Instala las dependencias:
       pip install gplearn pysr deap
     PySR además requiere Julia:
       python -m pysr install
  2. Pon este archivo en tu carpeta SR-Library junto a los demás módulos.
  3. Ejecuta:
       python experiment.py

Los resultados se guardan en:
  results_table.csv     <- tabla completa
  plot_r2.png           <- barras R² por dataset
  plot_time.png         <- barras tiempo por dataset
  plot_r2_vs_size.png   <- scatter R² vs complejidad
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. DATASETS SINTÉTICOS
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
N_SAMPLES    = 120   # muestras totales por dataset
TEST_SIZE    = 0.25  # 25 % para test

np.random.seed(RANDOM_STATE)

def make_datasets():
    datasets = {}

    # F1: y = x²
    X = np.random.uniform(-5, 5, (N_SAMPLES, 1))
    datasets["F1: x²"] = (X, X[:, 0] ** 2)

    # F2: y = x² + sin(x)
    X = np.random.uniform(-4, 4, (N_SAMPLES, 1))
    datasets["F2: x²+sin(x)"] = (X, X[:, 0] ** 2 + np.sin(X[:, 0]))

    # F3: y = x·sin(x) + cos(x)
    X = np.random.uniform(-5, 5, (N_SAMPLES, 1))
    datasets["F3: x·sin(x)+cos(x)"] = (X, X[:, 0] * np.sin(X[:, 0]) + np.cos(X[:, 0]))

    # F4: y = x0·x1 + sin(x0)   (2 variables)
    X = np.random.uniform(-3, 3, (N_SAMPLES, 2))
    datasets["F4: x0·x1+sin(x0)"] = (X, X[:, 0] * X[:, 1] + np.sin(X[:, 0]))

    # F5: y = x³ − x² + x − 1
    X = np.random.uniform(-3, 3, (N_SAMPLES, 1))
    datasets["F5: x³−x²+x−1"] = (X, X[:, 0]**3 - X[:, 0]**2 + X[:, 0] - 1)

    return datasets


# ---------------------------------------------------------------------------
# 2. WRAPPERS — misma interfaz para las 4 bibliotecas
# ---------------------------------------------------------------------------

# ── 2a. Nuestra biblioteca ──────────────────────────────────────────────────
def run_ours(X_train, y_train, X_test, n_features):
    from sklearn_api import SymbolicRegressor

    reg = SymbolicRegressor(
        population_size          = 300,
        generations              = 30,
        const_range              = (-5.0, 5.0),
        operators                = ["add", "sub", "mul", "div"],
        functions                = ["sin", "cos", "exp", "log", "sqrt"],
        parsimony_coeff          = 0.001,
        use_constant_optimization= True,
        local_search_every       = 5,
        local_search_top_k       = 5,
        random_state             = RANDOM_STATE,
        verbose                  = False,
    )
    reg.fit(X_train, y_train)
    y_pred     = reg.predict(X_test)
    complexity = reg.get_complexity()["size"]
    expression = reg.get_expression()
    return y_pred, complexity, expression


# ── 2b. gplearn ─────────────────────────────────────────────────────────────
def run_gplearn(X_train, y_train, X_test, n_features):
    from gplearn.genetic import SymbolicRegressor as GPLearnSR

    reg = GPLearnSR(
        population_size  = 300,
        generations       = 30,
        tournament_size   = 3,
        function_set      = ("add", "sub", "mul", "div", "sin", "cos", "sqrt", "log"),
        parsimony_coefficient = 0.001,
        random_state      = RANDOM_STATE,
        verbose           = 0,
    )
    reg.fit(X_train, y_train)
    y_pred     = reg.predict(X_test)
    complexity = reg._program.length_
    expression = str(reg._program)
    return y_pred, complexity, expression


# ── 2c. PySR ────────────────────────────────────────────────────────────────
def run_pysr(X_train, y_train, X_test, n_features):
    from pysr import PySRRegressor

    reg = PySRRegressor(
        niterations      = 30,
        binary_operators  = ["+", "-", "*", "/"],
        unary_operators   = ["sin", "cos", "exp", "sqrt", "log"],
        random_state      = RANDOM_STATE,
        verbosity         = 0,
        progress          = False,
    )
    reg.fit(X_train, y_train)
    y_pred     = reg.predict(X_test)
    best       = reg.get_best()
    complexity = int(best["complexity"])
    expression = str(best["equation"])
    return y_pred, complexity, expression


# ── 2d. DEAP ─────────────────────────────────────────────────────────────────
def run_deap(X_train, y_train, X_test, n_features):
    """
    GP estándar con DEAP usando gp.PrimitiveSet y eaSimple.
    Configuración equivalente a las demás: 300 individuos, 30 generaciones.
    """
    import operator
    import math
    from deap import algorithms, base, creator, gp, tools

    # ── Conjunto de primitivas ──
    pset = gp.PrimitiveSet("MAIN", n_features)
    pset.addPrimitive(operator.add,  2)
    pset.addPrimitive(operator.sub,  2)
    pset.addPrimitive(operator.mul,  2)
    pset.addPrimitive(lambda a, b: a / b if abs(b) > 1e-10 else 1.0, 2, name="div")
    pset.addPrimitive(math.sin,  1)
    pset.addPrimitive(math.cos,  1)
    pset.addPrimitive(lambda x: math.sqrt(abs(x)), 1, name="sqrt")
    pset.addPrimitive(lambda x: math.log(abs(x) + 1e-10), 1, name="log")
    pset.addEphemeralConstant("rand", lambda: np.random.uniform(-5, 5))
    for i in range(n_features):
        pset.renameArguments(**{f"ARG{i}": f"x{i}"})

    # ── Limpiar creator entre llamadas (evita error de redefinición) ──
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

    # Función de fitness: MSE
    def eval_individual(individual):
        func = toolbox.compile(expr=individual)
        try:
            preds = np.array([func(*row) for row in X_train])
            if not np.all(np.isfinite(preds)):
                return (1e10,)
            return (float(np.mean((y_train - preds) ** 2)),)
        except Exception:
            return (1e10,)

    toolbox.register("evaluate", eval_individual)
    toolbox.register("select",   tools.selTournament, tournsize=3)
    toolbox.register("mate",     gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate",   gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate",   gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

    rng_deap = np.random.RandomState(RANDOM_STATE)
    random_state_backup = np.random.get_state()
    np.random.seed(RANDOM_STATE)
    import random; random.seed(RANDOM_STATE)

    pop  = toolbox.population(n=300)
    hof  = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.1,
                         ngen=30, halloffame=hof, verbose=False)

    np.random.set_state(random_state_backup)

    best_ind = hof[0]
    func     = toolbox.compile(expr=best_ind)
    y_pred   = np.array([func(*row) for row in X_test])
    y_pred   = np.where(np.isfinite(y_pred), y_pred, 0.0)
    complexity = len(best_ind)
    expression = str(best_ind)
    return y_pred, complexity, expression


# ---------------------------------------------------------------------------
# 3. MOTOR DEL EXPERIMENTO
# ---------------------------------------------------------------------------

LIBRARIES = {
    "Nuestra"  : run_ours,
    "gplearn"  : run_gplearn,
    "PySR"     : run_pysr,
    "DEAP"     : run_deap,
}

def run_experiment():
    datasets = make_datasets()
    records  = []

    for ds_name, (X, y) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}  |  X: {X.shape}  |  y: {y.shape}")
        print(f"{'='*60}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        n_features = X.shape[1]

        for lib_name, run_fn in LIBRARIES.items():
            print(f"  [{lib_name}] ", end="", flush=True)
            try:
                t0     = time.time()
                y_pred, complexity, expression = run_fn(X_train, y_train, X_test, n_features)
                elapsed = time.time() - t0

                r2   = r2_score(y_test, y_pred)
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

                print(f"R²={r2:.4f}  RMSE={rmse:.4f}  t={elapsed:.1f}s  size={complexity}")
                records.append({
                    "dataset"   : ds_name,
                    "library"   : lib_name,
                    "r2"        : round(r2,   4),
                    "rmse"      : round(rmse, 4),
                    "time_s"    : round(elapsed, 2),
                    "complexity": complexity,
                    "expression": expression[:80],
                })

            except ImportError as e:
                print(f"NO INSTALADA ({e})")
                records.append({
                    "dataset": ds_name, "library": lib_name,
                    "r2": None, "rmse": None,
                    "time_s": None, "complexity": None,
                    "expression": "ImportError",
                })
            except Exception as e:
                print(f"ERROR: {e}")
                records.append({
                    "dataset": ds_name, "library": lib_name,
                    "r2": None, "rmse": None,
                    "time_s": None, "complexity": None,
                    "expression": f"Error: {str(e)[:60]}",
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. GRÁFICAS
# ---------------------------------------------------------------------------

COLORS = {
    "Nuestra" : "#E0654F",
    "gplearn" : "#1D9E75",
    "PySR"    : "#7B61FF",
    "DEAP"    : "#C97A1D",
}

def plot_results(df: pd.DataFrame):
    libraries = [l for l in LIBRARIES if l in df["library"].unique()]
    datasets  = df["dataset"].unique().tolist()
    x         = np.arange(len(datasets))
    width     = 0.8 / len(libraries)

    # ── Gráfica 1: R² por dataset ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, lib in enumerate(libraries):
        sub = df[df["library"] == lib]
        r2s = [sub[sub["dataset"] == ds]["r2"].values[0]
               if len(sub[sub["dataset"] == ds]) else None
               for ds in datasets]
        vals   = [v if v is not None else 0 for v in r2s]
        offset = (i - len(libraries) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width * 0.9,
                         label=lib, color=COLORS.get(lib, "#999"))
        for bar, v in zip(bars, r2s):
            if v is not None and v > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.01,
                         f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylabel("R² (mayor = mejor)")
    ax.set_ylim(0, 1.12)
    ax.set_title("R² en test por dataset y biblioteca")
    ax.legend(frameon=False)
    ax.axhline(1.0, color="gray", linewidth=0.7, linestyle="--")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig("plot_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nGuardado: plot_r2.png")

    # ── Gráfica 2: Tiempo por dataset ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, lib in enumerate(libraries):
        sub  = df[df["library"] == lib]
        times = [sub[sub["dataset"] == ds]["time_s"].values[0]
                 if len(sub[sub["dataset"] == ds]) else None
                 for ds in datasets]
        vals   = [v if v is not None else 0 for v in times]
        offset = (i - len(libraries) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9,
                label=lib, color=COLORS.get(lib, "#999"))

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylabel("Tiempo (segundos)")
    ax.set_title("Tiempo de entrenamiento por dataset y biblioteca")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig("plot_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Guardado: plot_time.png")

    # ── Gráfica 3: R² vs Complejidad (scatter) ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for lib in libraries:
        sub = df[(df["library"] == lib) & df["r2"].notna() & df["complexity"].notna()]
        ax.scatter(sub["complexity"], sub["r2"],
                    label=lib, color=COLORS.get(lib, "#999"),
                    s=80, alpha=0.85, edgecolors="white", linewidths=0.8)
        for _, row in sub.iterrows():
            ax.annotate(row["dataset"].split(":")[0],
                         (row["complexity"], row["r2"]),
                         textcoords="offset points", xytext=(5, 3), fontsize=7)

    ax.set_xlabel("Complejidad (nº nodos del árbol)")
    ax.set_ylabel("R² en test")
    ax.set_title("Calidad vs Complejidad de la expresión encontrada")
    ax.legend(frameon=False)
    ax.axhline(1.0, color="gray", linewidth=0.7, linestyle="--")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig("plot_r2_vs_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Guardado: plot_r2_vs_size.png")


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Experimento Comparativo de Regresión Simbólica")
    print("=" * 60)
    print(f"  Datasets   : 5 funciones sintéticas")
    print(f"  Bibliotecas: {', '.join(LIBRARIES)}")
    print(f"  Semilla    : {RANDOM_STATE}")
    print(f"  Train/Test : {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} %")

    # 1. Se ejecuta el experimento y se define la variable 'df' en el scope global/main
    df = run_experiment()

    # ===========================================================================
    # 6. GENERACIÓN DE REPORTES DETALLADOS
    # ===========================================================================
    print("\n" + "=" * 70)
    print("  TABLA COMPARATIVA MULTI-MÉTRICA (Resumen de Rendimiento)")
    print("=" * 70)
    
    # Transformamos usando los nombres reales de tus columnas: 'time_s' y 'complexity'
    df_melted = df.melt(
        id_vars=["dataset", "library"], 
        value_vars=["r2", "rmse", "time_s", "complexity"], 
        var_name="metric", 
        value_name="valor"
    )
    
    # Pivoteamos para estructurar filas (dataset, métrica) y columnas (librerías)
    pivot_detallada = df_melted.pivot_table(
        index=["dataset", "metric"], 
        columns="library", 
        values="valor", 
        aggfunc="first"
    )
    
    # Ordenamos las métricas jerárquicamente por estética
    metric_order = ["r2", "rmse", "complexity", "time_s"]
    pivot_detallada = pivot_detallada.reindex(metric_order, level="metric")
    
    # Configurar formato decimal limpio para la consola
    pd.set_option('display.float_format', lambda val: f"{val:.4f}" if isinstance(val, (float, np.float64)) else f"{val}")
    print(pivot_detallada.to_string())

    print("\n" + "=" * 70)
    print("  EXPRESIONES MATEMÁTICAS ENCONTRADAS")
    print("=" * 70)
    
    # Tabla exclusiva para las fórmulas resultantes
    pivot_formulas = df.pivot_table(
        index="dataset", 
        columns="library", 
        values="expression", 
        aggfunc="first"
    )
    
    # Imprimir las expresiones formateadas
    for dataset_name in pivot_formulas.index:
        print(f"\n🔹 Dataset: {dataset_name}")
        for lib_name in pivot_formulas.columns:
            expr = pivot_formulas.loc[dataset_name, lib_name]
            expr_trimmed = expr if len(str(expr)) < 90 else f"{str(expr)[:87]}..."
            print(f"  [{lib_name:7s}] -> {expr_trimmed}")

    # Guardar todos los reportes estructurados
    pivot_detallada.to_csv("results_detailed_metrics.csv")
    pivot_formulas.to_csv("results_detailed_expressions.csv")
    df.to_csv("results_table.csv", index=False)

    print("\n" + "-" * 70)
    print("[OK] Nuevos reportes detallados guardados con éxito:")
    print("     - results_detailed_metrics.csv      (Métricas cruzadas)")
    print("     - results_detailed_expressions.csv  (Fórmulas encontradas)")
    print("     - results_table.csv                 (Datos planos originales)")

    # 2. Llamamos a la función de gráficas pasándole el DataFrame correctamente definido
    plot_results(df)

    print("\n[OK] Experimento finalizado por completo.")