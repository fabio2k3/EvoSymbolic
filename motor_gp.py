"""
Motor de Programación Genética
==============================

Este módulo une todas las piezas del sistema de regresión simbólica en
un ciclo evolutivo completo:

1. Inicializa una población de árboles aleatorios.
2. Evalúa su calidad mediante una función de fitness.
3. Guarda estadísticas de cada generación.
4. Aplica selección, crossover y mutación.
5. Devuelve el mejor árbol encontrado.

Clase principal:
    - GPEngine
"""

import time
import random
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass

from expression_tree import ExpressionTree, TreeGenerator
from fitness import FitnessEvaluator, evaluate_population, best_individual
from genetic_operators import GeneticOperators

# ---------------------------------------------------------------------------
# 1. ESTRUCTURA DE HISTORIAL
# ---------------------------------------------------------------------------

@dataclass
class GenerationStats:
    """
    Estadísticas resumidas de una generación.

    Estos datos se almacenan en GPEngine.history_ y permiten analizar
    la convergencia del algoritmo y construir gráficas posteriores.
    """
    generation: int
    best_fitness: float
    mean_fitness: float
    median_fitness: float
    worst_fitness: float
    best_size: int
    mean_size: float
    best_expr: str
    n_invalid: int          # cantidad de individuos con fitness penalizado
    elapsed_time: float     # tiempo acumulado desde el inicio del proceso


# ---------------------------------------------------------------------------
# 2. MOTOR GP
# ---------------------------------------------------------------------------

class GPEngine:
    """
    Motor de Programación Genética para regresión simbólica.

    Parámetros
    ----------
    n_features : int
        Número de variables de entrada.
    population_size : int
        Tamaño de la población.
    generations : int
        Número máximo de generaciones.
    init_min_depth : int
        Profundidad mínima para la población inicial.
    init_max_depth : int
        Profundidad máxima para la población inicial.
    max_depth : int
        Profundidad máxima permitida durante la evolución.
    operators : list
        Operadores binarios permitidos. Si es None, se usan todos.
    functions : list
        Funciones unarias permitidas. Si es None, se usan todas.
    const_range : tuple
        Rango de valores para las constantes aleatorias.
    metric : str
        Métrica de fitness a minimizar.
    parsimony_coeff : float
        Coeficiente de penalización por tamaño del árbol.
    tournament_size : int
        Tamaño del torneo de selección.
    p_crossover : float
        Probabilidad de aplicar crossover.
    p_subtree_mut : float
        Probabilidad de mutación de subárbol.
    p_point_mut : float
        Probabilidad de mutación de punto.
    p_constant_mut : float
        Probabilidad de mutación de constante.
    elite_size : int
        Número de mejores individuos preservados por elitismo.
    stopping_fitness : float
        Umbral de parada anticipada.
    random_state : int
        Semilla para reproducibilidad.
    verbose : bool
        Si es True, imprime el progreso por pantalla.

    Atributos tras run()
    --------------------
    best_tree_ : ExpressionTree
        Mejor árbol encontrado durante toda la ejecución.
    best_fitness_ : float
        Fitness del mejor árbol encontrado.
    history_ : list[GenerationStats]
        Historial de estadísticas por generación.
    population_ : list
        Población final.
    fitnesses_ : list
        Fitness de la población final.
    stopped_reason_ : str
        Motivo de parada: 'max_generations' o 'stopping_fitness'.
    """

    def __init__(
        self,
        n_features:      int = 1,
        population_size: int = 200,
        generations:     int = 40,
        init_min_depth:  int = 1,
        init_max_depth:  int = 4,
        max_depth:       int = 8,
        operators:       Optional[list] = None,
        functions:       Optional[list] = None,
        const_range:     tuple = (-5.0, 5.0),
        metric:          str = "mse",
        parsimony_coeff: float = 0.001,
        tournament_size: int = 3,
        p_crossover:     float = 0.80,
        p_subtree_mut:   float = 0.10,
        p_point_mut:     float = 0.05,
        p_constant_mut:  float = 0.05,
        elite_size:      int = 1,
        stopping_fitness: Optional[float] = None,
        random_state:    Optional[int] = None,
        verbose:         bool = True,
    ):
        # Parámetros de configuración general
        self.n_features = n_features
        self.population_size = population_size
        self.generations = generations
        self.init_min_depth = init_min_depth
        self.init_max_depth = init_max_depth
        self.max_depth = max_depth
        self.metric = metric
        self.parsimony_coeff = parsimony_coeff
        self.tournament_size = tournament_size
        self.p_crossover = p_crossover
        self.p_subtree_mut = p_subtree_mut
        self.p_point_mut = p_point_mut
        self.p_constant_mut = p_constant_mut
        self.elite_size = elite_size
        self.stopping_fitness = stopping_fitness
        self.random_state = random_state
        self.verbose = verbose

        # Componentes internos reutilizando los módulos anteriores
        self.generator = TreeGenerator(
            n_features=n_features,
            operators=operators,
            functions=functions,
            const_range=const_range,
        )

        self.evaluator = FitnessEvaluator(
            metric=metric,
            parsimony_coeff=parsimony_coeff,
        )

        self.gen_operators = GeneticOperators(
            generator=self.generator,
            tournament_size=tournament_size,
            p_crossover=p_crossover,
            p_subtree_mut=p_subtree_mut,
            p_point_mut=p_point_mut,
            p_constant_mut=p_constant_mut,
            max_depth=max_depth,
            elite_size=elite_size,
        )

        # Resultados que se rellenan durante run()
        self.best_tree_ = None
        self.best_fitness_ = float("inf")
        self.history_ = []
        self.population_ = []
        self.fitnesses_ = []
        self.stopped_reason_ = None

    # ------------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # ------------------------------------------------------------------
    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        callback: Optional[Callable[["GPEngine", GenerationStats], None]] = None,
    ) -> "GPEngine":
        """
        Ejecuta el proceso evolutivo completo.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada con forma (n_samples, n_features).
        y : np.ndarray
            Vector de valores reales.
        callback : callable, opcional
            Función que se ejecuta al final de cada generación.
            Recibe (self, stats).

        Retorna
        -------
        GPEngine
            La propia instancia, para permitir encadenamiento.
        """
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        start_time = time.time()

        # --------------------------------------------------------------
        # 1. Inicialización de población
        # --------------------------------------------------------------
        self.population_ = [
            self.generator.ramped_half_and_half(self.init_min_depth, self.init_max_depth)
            for _ in range(self.population_size)
        ]

        if self.verbose:
            print(f"GPEngine: población inicial de {self.population_size} individuos")
            print(f"          {self.generations} generaciones máx., métrica={self.metric}")
            print("-" * 60)

        # --------------------------------------------------------------
        # 2. Bucle evolutivo
        # --------------------------------------------------------------
        for gen in range(self.generations + 1):  # +1 para incluir la generación 0
            # Evaluar la población actual
            self.fitnesses_ = evaluate_population(self.population_, X, y, self.evaluator)

            # Actualizar el mejor individuo global si mejora
            gen_best_tree, gen_best_fit = best_individual(self.population_, self.fitnesses_)
            if gen_best_fit < self.best_fitness_:
                self.best_fitness_ = gen_best_fit
                self.best_tree_ = gen_best_tree.clone()

            # Calcular estadísticas de la generación
            stats = self._compute_stats(gen, start_time)
            self.history_.append(stats)

            if self.verbose:
                self._print_progress(stats)

            if callback is not None:
                callback(self, stats)

            # ----------------------------------------------------------
            # 3. Criterios de parada
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # 4. Evolución de la población
            # ----------------------------------------------------------
            self.population_ = self.gen_operators.evolve(self.population_, self.fitnesses_)

        if self.verbose:
            elapsed = time.time() - start_time
            print("-" * 60)
            print(f"Finalizado ({self.stopped_reason_}) en {elapsed:.2f}s")
            print(f"Mejor expresión : {self.best_tree_.to_string()}")
            print(f"Mejor fitness   : {self.best_fitness_:.6f}")

        return self

    # ------------------------------------------------------------------
    # PREDICCIÓN
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Evalúa el mejor árbol encontrado sobre nuevos datos.
        """
        if self.best_tree_ is None:
            raise RuntimeError("El motor no ha sido ejecutado. Llama a run(X, y) primero.")
        return self.best_tree_.evaluate(X)

    # ------------------------------------------------------------------
    # UTILIDADES INTERNAS
    # ------------------------------------------------------------------
    def _compute_stats(self, gen: int, start_time: float) -> GenerationStats:
        """
        Calcula las métricas resumidas de la generación actual.
        """
        fits = np.array(self.fitnesses_)
        sizes = np.array([t.size() for t in self.population_])

        # Individuos inválidos: fitness muy grande (penalización)
        n_invalid = int(np.sum(fits >= 1e9))

        # Para métricas agregadas usamos solo individuos válidos si existen
        valid_fits = fits[fits < 1e9]
        if len(valid_fits) == 0:
            valid_fits = fits

        best_idx = int(np.argmin(fits))

        return GenerationStats(
            generation=gen,
            best_fitness=float(np.min(fits)),
            mean_fitness=float(np.mean(valid_fits)),
            median_fitness=float(np.median(valid_fits)),
            worst_fitness=float(np.max(fits)),
            best_size=int(sizes[best_idx]),
            mean_size=float(np.mean(sizes)),
            best_expr=self.population_[best_idx].to_string(),
            n_invalid=n_invalid,
            elapsed_time=time.time() - start_time,
        )

    def _print_progress(self, stats: GenerationStats) -> None:
        """
        Imprime un resumen compacto de la generación actual.
        """
        expr_preview = stats.best_expr if len(stats.best_expr) <= 45 else stats.best_expr[:42] + "..."
        print(
            f"Gen {stats.generation:3d} | "
            f"best={stats.best_fitness:10.5f} | "
            f"mean={stats.mean_fitness:10.5f} | "
            f"size={stats.best_size:2d} | "
            f"inv={stats.n_invalid:3d} | "
            f"{expr_preview}"
        )


# ---------------------------------------------------------------------------
# 3. DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Motor GP (Bucle Evolutivo Completo)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 3.1  Problema fácil: y = x^2
    # ------------------------------------------------------------------
    print("\n### Problema 1: y = x^2 ###\n")

    X1 = np.linspace(-5, 5, 50).reshape(-1, 1)
    y1 = X1[:, 0] ** 2

    engine1 = GPEngine(
        n_features=1,
        population_size=100,
        generations=15,
        const_range=(-2.0, 2.0),
        stopping_fitness=1e-6,
        random_state=42,
        verbose=True,
    )
    engine1.run(X1, y1)

    # ------------------------------------------------------------------
    # 3.2  Problema medio: y = x^2 + sin(x)
    # ------------------------------------------------------------------
    print("\n### Problema 2: y = x^2 + sin(x) ###\n")

    X2 = np.linspace(-3, 3, 60).reshape(-1, 1)
    y2 = X2[:, 0] ** 2 + np.sin(X2[:, 0])

    engine2 = GPEngine(
        n_features=1,
        population_size=300,
        generations=30,
        const_range=(-3.0, 3.0),
        parsimony_coeff=0.0005,
        stopping_fitness=1e-5,
        random_state=1,
        verbose=True,
    )
    engine2.run(X2, y2)

    # ------------------------------------------------------------------
    # 3.3  Problema con 2 variables: y = x0 * x1 + 1
    # ------------------------------------------------------------------
    print("\n### Problema 3: y = x0 * x1 + 1 (2 variables) ###\n")

    np.random.seed(0)
    X3 = np.random.uniform(-5, 5, size=(60, 2))
    y3 = X3[:, 0] * X3[:, 1] + 1

    engine3 = GPEngine(
        n_features=2,
        population_size=200,
        generations=20,
        const_range=(-2.0, 2.0),
        stopping_fitness=1e-6,
        random_state=5,
        verbose=True,
    )
    engine3.run(X3, y3)

    # ------------------------------------------------------------------
    # 3.4  Predicción sobre nuevos datos
    # ------------------------------------------------------------------
    print("\n### Predicción con best_tree_ (Problema 1) ###\n")
    X_new = np.array([[10.0], [-7.0], [0.5]])
    y_pred = engine1.predict(X_new)
    y_real = X_new[:, 0] ** 2

    for xi, yp, yr in zip(X_new[:, 0], y_pred, y_real):
        print(f"  x={xi:6.2f}  ->  predicho={yp:10.4f}   real={yr:10.4f}")

    # ------------------------------------------------------------------
    # 3.5  Uso del callback
    # ------------------------------------------------------------------
    print("\n### Callback personalizado (Problema 1, re-ejecutado silencioso) ###\n")

    convergence_log = []

    def my_callback(engine, stats):
        convergence_log.append((stats.generation, stats.best_fitness))

    engine_silent = GPEngine(
        n_features=1,
        population_size=100,
        generations=10,
        const_range=(-2.0, 2.0),
        stopping_fitness=1e-6,
        random_state=42,
        verbose=False,   # silencioso, el callback guarda el progreso
    )
    engine_silent.run(X1, y1, callback=my_callback)

    print("  Convergencia capturada por callback:")
    for gen, fit in convergence_log:
        print(f"    gen {gen:2d}: fitness = {fit:.6f}")

    print("\n[OK] Motor GP completado. Siguiente paso: optimización de constantes si se desea.")