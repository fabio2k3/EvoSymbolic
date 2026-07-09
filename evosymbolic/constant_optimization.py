"""
Optimización de Constantes
==========================

GP es muy bueno encontrando la ESTRUCTURA de una expresión
(por ejemplo: "x0 * x1 + algo"), pero suele ser débil afinando
valores numéricos exactos.

Esta fase añade una búsqueda local para ajustar constantes y mejorar
la precisión final de los árboles ya encontrados por GP.

Componentes:
  - extract_constants(tree)        -> obtiene nodos Constant y sus valores
  - set_constants(const_nodes, x)  -> actualiza constantes in-place
  - make_objective(...)            -> construye una función objetivo para scipy
  - optimize_constants(...)        -> optimiza un árbol con scipy.optimize.minimize
  - optimize_population(...)       -> aplica optimización a varios individuos
  - GPEngineWithLocalSearch        -> motor GP con optimización periódica
"""

import numpy as np
import random
from typing import List, Tuple, Optional
from scipy.optimize import minimize

from expression_tree import ExpressionTree, Constant
from fitness import mse, FitnessEvaluator, evaluate_population, best_individual
from motor_gp import GPEngine, GenerationStats


# ---------------------------------------------------------------------------
# 1. EXTRACCIÓN Y REINSERCIÓN DE CONSTANTES
# ---------------------------------------------------------------------------

def extract_constants(tree: ExpressionTree) -> Tuple[List[Constant], np.ndarray]:
    """
    Recorre el árbol y devuelve:
      - const_nodes: lista de referencias a nodos Constant
      - values: array con los valores actuales de esas constantes

    Importante:
    las referencias apuntan al mismo objeto que vive dentro del árbol.
    Por tanto, modificar const_nodes[i].value modifica el árbol directamente.
    """
    const_nodes = [
        node for (node, _parent, _pos) in tree.get_all_nodes()
        if isinstance(node, Constant)
    ]
    values = np.array([c.value for c in const_nodes], dtype=float)
    return const_nodes, values


def set_constants(const_nodes: List[Constant], values: np.ndarray) -> None:
    """
    Actualiza in-place el valor de cada nodo Constant.

    Como const_nodes contiene referencias directas a los objetos del árbol,
    esta función modifica el árbol sin necesidad de buscar nodos otra vez.
    """
    for node, val in zip(const_nodes, values):
        node.value = float(val)


# ---------------------------------------------------------------------------
# 2. FUNCIÓN OBJETIVO PARA SCIPY
# ---------------------------------------------------------------------------

def make_objective(
    tree: ExpressionTree,
    const_nodes: List[Constant],
    X: np.ndarray,
    y_true: np.ndarray,
):
    """
    Construye una función objective(params) -> float lista para scipy.minimize.

    El comportamiento es:
      1. Escribe params en los nodos Constant del árbol.
      2. Evalúa el árbol completo sobre X.
      3. Devuelve el MSE frente a y_true.

    Si la evaluación produce NaN/Inf o lanza una excepción, se devuelve
    un valor muy alto para alejar al optimizador de esa región.
    """
    def objective(params: np.ndarray) -> float:
        # Insertar los nuevos valores en el árbol
        set_constants(const_nodes, params)

        try:
            y_pred = tree.evaluate(X)
        except Exception:
            return 1e10

        if not np.all(np.isfinite(y_pred)):
            return 1e10

        return mse(y_true, y_pred)

    return objective


# ---------------------------------------------------------------------------
# 3. OPTIMIZACIÓN DE UN ÁRBOL
# ---------------------------------------------------------------------------

def optimize_constants(
    tree: ExpressionTree,
    X: np.ndarray,
    y_true: np.ndarray,
    method: str = "Nelder-Mead",
    maxiter: int = 100,
) -> Tuple[ExpressionTree, bool]:
    """
    Optimiza las constantes de un árbol con scipy.optimize.minimize.

    Parámetros
    ----------
    tree : ExpressionTree
        Árbol cuyas constantes se optimizarán.
    X, y_true : np.ndarray
        Datos de entrenamiento.
    method : str
        Algoritmo de scipy.
        - 'Nelder-Mead': robusto y no requiere gradiente.
        - 'L-BFGS-B': más rápido, pero asume suavidad.
    maxiter : int
        Número máximo de iteraciones del optimizador.

    Devuelve
    --------
    (optimized_tree, improved)
      optimized_tree : clon del árbol con constantes optimizadas,
                       o el original si no había constantes o falló.
      improved       : True si el MSE mejoró estrictamente.

    Diseño defensivo:
      - Si el árbol no tiene constantes, devuelve (tree.clone(), False)
        sin llamar a scipy.
      - Si scipy falla o el resultado es peor, se restaura el estado original.
    """
    working_tree = tree.clone()
    const_nodes, x0 = extract_constants(working_tree)

    if len(const_nodes) == 0:
        return working_tree, False

    objective = make_objective(working_tree, const_nodes, X, y_true)

    # MSE original antes de optimizar
    original_mse = objective(x0.copy())

    try:
        result = minimize(
            objective,
            x0,
            method=method,
            options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-8}
                    if method == "Nelder-Mead" else {"maxiter": maxiter},
        )
        new_mse = result.fun
        new_x = result.x
    except Exception:
        # Optimización falló: restaurar valores originales
        set_constants(const_nodes, x0)
        return working_tree, False

    if new_mse < original_mse and np.all(np.isfinite(new_x)):
        set_constants(const_nodes, new_x)
        return working_tree, True
    else:
        # No mejoró: restaurar valores originales
        set_constants(const_nodes, x0)
        return working_tree, False


# ---------------------------------------------------------------------------
# 4. OPTIMIZACIÓN DE UNA POBLACIÓN
# ---------------------------------------------------------------------------

def optimize_population(
    population: List[ExpressionTree],
    X: np.ndarray,
    y_true: np.ndarray,
    evaluator: FitnessEvaluator,
    fitnesses: List[float],
    top_k: Optional[int] = None,
    method: str = "Nelder-Mead",
    maxiter: int = 100,
) -> Tuple[List[ExpressionTree], List[float], int]:
    """
    Aplica optimize_constants() a una población.

    Parámetros
    ----------
    top_k : int o None
        Si se especifica, solo se optimizan los top_k mejores individuos
        según su fitness actual. Esto reduce mucho el coste computacional.

    Devuelve
    --------
    (new_population, new_fitnesses, n_improved)
      new_population : lista con la misma longitud que la original
      new_fitnesses  : fitness recalculado para los individuos mejorados
      n_improved     : cuántos individuos mejoraron su fitness
    """
    new_population = list(population)
    new_fitnesses = list(fitnesses)
    n_improved = 0

    if top_k is None:
        indices = range(len(population))
    else:
        # Índices de los top_k mejores individuos (fitness más bajo primero)
        indices = np.argsort(fitnesses)[:top_k]

    for i in indices:
        optimized, improved = optimize_constants(
            population[i], X, y_true, method=method, maxiter=maxiter
        )
        if improved:
            new_fit = evaluator.evaluate(optimized, X, y_true)
            if new_fit < new_fitnesses[i]:
                new_population[i] = optimized
                new_fitnesses[i] = new_fit
                n_improved += 1

    return new_population, new_fitnesses, n_improved


# ---------------------------------------------------------------------------
# 5. GPEngine CON BÚSQUEDA LOCAL
# ---------------------------------------------------------------------------

class GPEngineWithLocalSearch(GPEngine):
    """
    Extiende GPEngine para aplicar búsqueda local sobre constantes de forma
    periódica durante la evolución.

    Cada `local_search_every` generaciones, se optimizan los `local_search_top_k`
    mejores individuos usando scipy.optimize.minimize.

    Esto implementa una estrategia memética:
      - GP global para explorar la estructura.
      - Búsqueda local para explotar y ajustar parámetros numéricos.
    """

    def __init__(
        self,
        *args,
        local_search_every: int = 5,
        local_search_top_k: int = 5,
        local_search_method: str = "Nelder-Mead",
        local_search_maxiter: int = 100,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.local_search_every = local_search_every
        self.local_search_top_k = local_search_top_k
        self.local_search_method = local_search_method
        self.local_search_maxiter = local_search_maxiter
        self.n_local_search_improved_ = 0  # contador acumulado

    def run(self, X: np.ndarray, y: np.ndarray, callback=None) -> "GPEngineWithLocalSearch":
        """
        Ejecuta el bucle evolutivo completo agregando búsqueda local.

        La optimización de constantes se aplica después de evaluar la población
        y antes de actualizar el mejor individuo global, para que el mejor
        guardado ya refleje las constantes ajustadas.
        """
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        import time
        start_time = time.time()

        # Inicialización de la población
        self.population_ = [
            self.generator.ramped_half_and_half(self.init_min_depth, self.init_max_depth)
            for _ in range(self.population_size)
        ]

        if self.verbose:
            print(
                f"GPEngineWithLocalSearch: poblacion={self.population_size}, "
                f"local_search cada {self.local_search_every} gens "
                f"(top_{self.local_search_top_k}, {self.local_search_method})"
            )
            print("-" * 60)

        for gen in range(self.generations + 1):
            # Evaluación normal de la población
            self.fitnesses_ = evaluate_population(self.population_, X, y, self.evaluator)

            # Búsqueda local periódica
            apply_ls = (
                self.local_search_every > 0
                and gen % self.local_search_every == 0
                and gen > 0
            )

            if apply_ls:
                self.population_, self.fitnesses_, n_imp = optimize_population(
                    self.population_, X, y, self.evaluator, self.fitnesses_,
                    top_k=self.local_search_top_k,
                    method=self.local_search_method,
                    maxiter=self.local_search_maxiter,
                )
                self.n_local_search_improved_ += n_imp

            # Actualizar mejor global
            gen_best_tree, gen_best_fit = best_individual(self.population_, self.fitnesses_)
            if gen_best_fit < self.best_fitness_:
                self.best_fitness_ = gen_best_fit
                self.best_tree_ = gen_best_tree.clone()

            # Estadísticas de la generación
            stats = self._compute_stats(gen, start_time)
            self.history_.append(stats)

            if self.verbose:
                self._print_progress(stats)
                if apply_ls:
                    print(f"          ({n_imp} individuos mejorados por búsqueda local)")

            if callback is not None:
                callback(self, stats)

            # Criterios de parada
            if gen >= self.generations:
                self.stopped_reason_ = "max_generations"
                break

            if (
                self.stopping_fitness is not None
                and self.best_fitness_ <= self.stopping_fitness
            ):
                self.stopped_reason_ = "stopping_fitness"
                if self.verbose:
                    print(
                        f"\n¡Fitness objetivo alcanzado! "
                        f"({self.best_fitness_:.6f} <= {self.stopping_fitness})"
                    )
                break

            # Evolución de la siguiente generación
            self.population_ = self.gen_operators.evolve(self.population_, self.fitnesses_)

        if self.verbose:
            elapsed = time.time() - start_time
            print("-" * 60)
            print(f"Finalizado ({self.stopped_reason_}) en {elapsed:.2f}s")
            print(f"Mejoras totales por busqueda local: {self.n_local_search_improved_}")
            print(f"Mejor expresion : {self.best_tree_.to_string()}")
            print(f"Mejor fitness   : {self.best_fitness_:.6f}")

        return self


# ---------------------------------------------------------------------------
# 6. DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Optimización de Constantes (scipy)")
    print("=" * 60)

    from expression_tree import OperatorNode, FunctionNode, Variable

    # ------------------------------------------------------------------
    # 6.1  Optimización directa de un árbol con una constante incorrecta
    # ------------------------------------------------------------------
    print("\n--- Optimización directa: ((x0 * x1) + abs(C)) con C=1.5 ---\n")

    np.random.seed(0)
    X = np.random.uniform(-5, 5, size=(60, 2))
    y_true = X[:, 0] * X[:, 1] + 1.0  # el valor correcto de la constante es 1.0

    bad_tree = ExpressionTree(
        OperatorNode("add",
            OperatorNode("mul", Variable(0), Variable(1)),
            FunctionNode("abs", Constant(1.5))
        )
    )

    evaluator = FitnessEvaluator(metric="mse", parsimony_coeff=0.001)

    mse_before = mse(y_true, bad_tree.evaluate(X))
    print(f"  Antes    : {bad_tree.to_string()}")
    print(f"  MSE antes: {mse_before:.6f}")

    optimized, improved = optimize_constants(bad_tree, X, y_true, method="Nelder-Mead")
    mse_after = mse(y_true, optimized.evaluate(X))

    print(f"  Después  : {optimized.to_string()}")
    print(f"  MSE después: {mse_after:.6f}")
    print(f"  ¿Mejoró? : {improved}")

    # ------------------------------------------------------------------
    # 6.2  Múltiples constantes en el mismo árbol
    # ------------------------------------------------------------------
    print("\n--- Múltiples constantes: (C1 * x0) + (x1 / C2) ---\n")
    print("  Objetivo real: y = 2*x0 + x1/3\n")

    y_true2 = 2.0 * X[:, 0] + X[:, 1] / 3.0

    multi_tree = ExpressionTree(
        OperatorNode("add",
            OperatorNode("mul", Constant(0.3), Variable(0)),
            OperatorNode("div", Variable(1), Constant(7.0))
        )
    )

    mse_before2 = mse(y_true2, multi_tree.evaluate(X))
    print(f"  Antes    : {multi_tree.to_string()}")
    print(f"  MSE antes: {mse_before2:.6f}")

    optimized2, improved2 = optimize_constants(
        multi_tree, X, y_true2, method="Nelder-Mead", maxiter=200
    )
    mse_after2 = mse(y_true2, optimized2.evaluate(X))

    print(f"  Después  : {optimized2.to_string()}")
    print(f"  MSE después: {mse_after2:.10f}")
    print(f"  ¿Mejoró? : {improved2}")

    # ------------------------------------------------------------------
    # 6.3  Árbol sin constantes
    # ------------------------------------------------------------------
    print("\n--- Árbol sin constantes (debe devolverse sin cambios) ---\n")
    no_const_tree = ExpressionTree(OperatorNode("mul", Variable(0), Variable(1)))
    opt_nc, imp_nc = optimize_constants(no_const_tree, X, y_true2)
    print(f"  Árbol   : {opt_nc.to_string()}")
    print(f"  Mejoró? : {imp_nc}  (esperado: False, sin llamar a scipy)")

    # ------------------------------------------------------------------
    # 6.4  Comparativa: GP normal vs GP con búsqueda local
    # ------------------------------------------------------------------
    print("\n--- Comparativa: GP normal vs GP + búsqueda local ---")
    print("    Problema: y = x0 * x1 + 1\n")

    X3 = np.random.uniform(-5, 5, size=(60, 2))
    y3 = X3[:, 0] * X3[:, 1] + 1

    print("### GP normal (sin búsqueda local) ###\n")
    engine_normal = GPEngine(
        n_features=2,
        population_size=150,
        generations=20,
        const_range=(-2.0, 2.0),
        random_state=5,
        verbose=False,
    )
    engine_normal.run(X3, y3)
    print(f"  Mejor expresion : {engine_normal.best_tree_.to_string()}")
    print(f"  Mejor fitness   : {engine_normal.best_fitness_:.6f}")

    print("\n### GP + búsqueda local cada 5 generaciones (top 5) ###\n")
    engine_ls = GPEngineWithLocalSearch(
        n_features=2,
        population_size=150,
        generations=20,
        const_range=(-2.0, 2.0),
        random_state=5,
        local_search_every=5,
        local_search_top_k=5,
        local_search_method="Nelder-Mead",
        verbose=False,
    )
    engine_ls.run(X3, y3)
    print(f"  Mejor expresion : {engine_ls.best_tree_.to_string()}")
    print(f"  Mejor fitness   : {engine_ls.best_fitness_:.6f}")
    print(f"  Mejoras locales : {engine_ls.n_local_search_improved_}")

    print("\n--- Ejecución detallada con búsqueda local ---\n")
    engine_ls2 = GPEngineWithLocalSearch(
        n_features=2,
        population_size=150,
        generations=15,
        const_range=(-2.0, 2.0),
        random_state=5,
        local_search_every=5,
        local_search_top_k=5,
        local_search_method="Nelder-Mead",
        stopping_fitness=1e-7,
        verbose=True,
    )
    engine_ls2.run(X3, y3)

    print("\n[OK] Optimización constantes completada.")