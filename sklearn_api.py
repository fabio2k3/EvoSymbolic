"""
API estilo scikit-learn
=======================

Este módulo envuelve el motor de Programación Genética en un estimador
compatible con la interfaz de scikit-learn.

La clase principal, SymbolicRegressor, implementa:
  - fit(X, y)
  - predict(X)
  - score(X, y)

Además, mantiene compatibilidad con:
  - BaseEstimator
  - RegressorMixin
  - get_params()
  - set_params()
  - clone()
  - Pipeline
  - cross_val_score
"""

import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from expression_tree import ExpressionTree
from fitness import r2_score as _r2_score
from motor_gp import GPEngine
from constant_optimization import GPEngineWithLocalSearch


class SymbolicRegressor(BaseEstimator, RegressorMixin):
    """
    Regresor de regresión simbólica con interfaz estilo scikit-learn.

    Internamente utiliza Programación Genética y, de forma opcional,
    optimización local de constantes.

    Parámetros
    ----------
    population_size : int
        Tamaño de la población.
    generations : int
        Número máximo de generaciones.
    init_min_depth : int
        Profundidad mínima de la población inicial.
    init_max_depth : int
        Profundidad máxima de la población inicial.
    max_depth : int
        Profundidad máxima permitida durante la evolución.
    operators : list
        Operadores binarios permitidos. Si es None, se usan todos.
    functions : list
        Funciones unarias permitidas. Si es None, se usan todas.
    const_range : tuple
        Rango de valores para constantes aleatorias.
    metric : str
        Métrica de fitness a minimizar.
    parsimony_coeff : float
        Penalización por tamaño del árbol.
    tournament_size : int
        Tamaño del torneo de selección.
    p_crossover : float
        Probabilidad de crossover.
    p_subtree_mut : float
        Probabilidad de mutación de subárbol.
    p_point_mut : float
        Probabilidad de mutación de punto.
    p_constant_mut : float
        Probabilidad de mutación de constante.
    elite_size : int
        Número de individuos preservados por elitismo.
    stopping_fitness : float
        Umbral para detener el entrenamiento antes de tiempo.
    use_constant_optimization : bool
        Activa búsqueda local de constantes.
    local_search_every : int
        Cada cuántas generaciones aplicar búsqueda local.
    local_search_top_k : int
        Cuántos individuos optimizar en cada paso local.
    local_search_method : str
        Método de scipy.optimize.
    random_state : int
        Semilla para reproducibilidad.
    verbose : bool
        Si es True, imprime progreso durante el entrenamiento.
    """

    def __init__(
        self,
        population_size: int = 200,
        generations: int = 40,
        init_min_depth: int = 1,
        init_max_depth: int = 4,
        max_depth: int = 8,
        operators: Optional[list] = None,
        functions: Optional[list] = None,
        const_range: tuple = (-5.0, 5.0),
        metric: str = "mse",
        parsimony_coeff: float = 0.001,
        tournament_size: int = 3,
        p_crossover: float = 0.80,
        p_subtree_mut: float = 0.10,
        p_point_mut: float = 0.05,
        p_constant_mut: float = 0.05,
        elite_size: int = 1,
        stopping_fitness: Optional[float] = None,
        use_constant_optimization: bool = False,
        local_search_every: int = 5,
        local_search_top_k: int = 5,
        local_search_method: str = "Nelder-Mead",
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        # Regla de oro de sklearn: __init__ solo guarda hiperparámetros.
        # No se debe entrenar, validar ni crear objetos pesados aquí.
        self.population_size = population_size
        self.generations = generations
        self.init_min_depth = init_min_depth
        self.init_max_depth = init_max_depth
        self.max_depth = max_depth
        self.operators = operators
        self.functions = functions
        self.const_range = const_range
        self.metric = metric
        self.parsimony_coeff = parsimony_coeff
        self.tournament_size = tournament_size
        self.p_crossover = p_crossover
        self.p_subtree_mut = p_subtree_mut
        self.p_point_mut = p_point_mut
        self.p_constant_mut = p_constant_mut
        self.elite_size = elite_size
        self.stopping_fitness = stopping_fitness
        self.use_constant_optimization = use_constant_optimization
        self.local_search_every = local_search_every
        self.local_search_top_k = local_search_top_k
        self.local_search_method = local_search_method
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SymbolicRegressor":
        """
        Entrena el modelo ejecutando el ciclo evolutivo completo.

        Sigue el contrato de sklearn:
          - valida X e y con check_X_y
          - guarda n_features_in_
          - crea los objetos pesados aquí, no en __init__
          - devuelve self
        """
        X, y = check_X_y(X, y, ensure_2d=True, dtype=float)
        self.n_features_in_ = X.shape[1]

        # Selección de motor: GP estándar o GP con búsqueda local.
        EngineClass = (
            GPEngineWithLocalSearch if self.use_constant_optimization else GPEngine
        )

        engine_kwargs = dict(
            n_features=self.n_features_in_,
            population_size=self.population_size,
            generations=self.generations,
            init_min_depth=self.init_min_depth,
            init_max_depth=self.init_max_depth,
            max_depth=self.max_depth,
            operators=self.operators,
            functions=self.functions,
            const_range=self.const_range,
            metric=self.metric,
            parsimony_coeff=self.parsimony_coeff,
            tournament_size=self.tournament_size,
            p_crossover=self.p_crossover,
            p_subtree_mut=self.p_subtree_mut,
            p_point_mut=self.p_point_mut,
            p_constant_mut=self.p_constant_mut,
            elite_size=self.elite_size,
            stopping_fitness=self.stopping_fitness,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        if self.use_constant_optimization:
            engine_kwargs.update(
                local_search_every=self.local_search_every,
                local_search_top_k=self.local_search_top_k,
                local_search_method=self.local_search_method,
            )

        self.engine_ = EngineClass(**engine_kwargs)
        self.engine_.run(X, y)

        # Atributos aprendidos durante fit(), siguiendo la convención sklearn.
        self.best_tree_ = self.engine_.best_tree_
        self.best_fitness_ = self.engine_.best_fitness_
        self.history_ = self.engine_.history_

        return self

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice la salida para nuevos datos usando el mejor árbol encontrado.
        """
        check_is_fitted(self, attributes=["best_tree_"])
        X = check_array(X, ensure_2d=True, dtype=float)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X tiene {X.shape[1]} columnas, pero el modelo fue "
                f"entrenado con {self.n_features_in_}."
            )

        return self.best_tree_.evaluate(X)

    # ------------------------------------------------------------------
    # SCORE
    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Devuelve el coeficiente R², igual que cualquier regresor de sklearn.
        """
        y_pred = self.predict(X)
        return _r2_score(np.asarray(y, dtype=float), y_pred)

    # ------------------------------------------------------------------
    # UTILIDADES EXTRA
    # ------------------------------------------------------------------
    def get_expression(self) -> str:
        """Devuelve la expresión final como texto legible."""
        check_is_fitted(self, attributes=["best_tree_"])
        return self.best_tree_.to_string()

    def get_complexity(self) -> dict:
        """Devuelve tamaño y profundidad del mejor árbol encontrado."""
        check_is_fitted(self, attributes=["best_tree_"])
        return {
            "size": self.best_tree_.size(),
            "depth": self.best_tree_.depth(),
        }


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  API estilo scikit-learn")
    print("=" * 60)

    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # ------------------------------------------------------------------
    # 1. Uso básico: fit / predict / score
    # ------------------------------------------------------------------
    print("\n--- Uso básico: fit / predict / score ---\n")

    np.random.seed(0)
    X = np.random.uniform(-5, 5, size=(100, 1))
    y = X[:, 0] ** 2 + np.sin(X[:, 0])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    reg = SymbolicRegressor(
        population_size=200,
        generations=25,
        const_range=(-3.0, 3.0),
        random_state=42,
        verbose=False,
    )
    reg.fit(X_train, y_train)

    print(f"  Expresión encontrada : {reg.get_expression()}")
    print(f"  Complejidad          : {reg.get_complexity()}")
    print(f"  R^2 en train         : {reg.score(X_train, y_train):.4f}")
    print(f"  R^2 en test          : {reg.score(X_test, y_test):.4f}")

    y_pred = reg.predict(X_test[:5])
    print(f"\n  Primeras 5 predicciones de test:")
    for xi, yp, yr in zip(X_test[:5, 0], y_pred, y_test[:5]):
        print(f"    x={xi:7.3f}  ->  pred={yp:8.4f}   real={yr:8.4f}")

    # ------------------------------------------------------------------
    # 2. get_params / set_params
    # ------------------------------------------------------------------
    print("\n--- get_params / set_params ---\n")

    params = reg.get_params()
    print(f"  Número de hiperparámetros expuestos: {len(params)}")
    print(f"  population_size actual: {params['population_size']}")

    reg2 = SymbolicRegressor()
    reg2.set_params(population_size=50, generations=5, random_state=1)
    print(
        f"  reg2 tras set_params: population_size={reg2.population_size}, "
        f"generations={reg2.generations}"
    )

    # ------------------------------------------------------------------
    # 3. clone()
    # ------------------------------------------------------------------
    print("\n--- sklearn.base.clone() ---\n")
    from sklearn.base import clone

    reg_clone = clone(reg)
    print(f"  Clon creado sin fit: {not hasattr(reg_clone, 'best_tree_')}")
    print(f"  Mismos hiperparámetros: {reg_clone.get_params() == reg.get_params()}")

    # ------------------------------------------------------------------
    # 4. Uso dentro de un Pipeline
    # ------------------------------------------------------------------
    print("\n--- Uso dentro de un Pipeline de sklearn ---\n")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("symbolic", SymbolicRegressor(
            population_size=100,
            generations=15,
            const_range=(-3.0, 3.0),
            random_state=0,
            verbose=False,
        )),
    ])
    pipe.fit(X_train, y_train)
    print(f"  R^2 con pipeline (escalado + SR): {pipe.score(X_test, y_test):.4f}")
    print(f"  Expresión (sobre datos escalados): {pipe.named_steps['symbolic'].get_expression()}")

    # ------------------------------------------------------------------
    # 5. cross_val_score
    # ------------------------------------------------------------------
    print("\n--- cross_val_score (3-fold) ---\n")
    print("  (población pequeña para que la demo sea rápida)\n")

    reg_cv = SymbolicRegressor(
        population_size=80,
        generations=10,
        const_range=(-3.0, 3.0),
        random_state=0,
        verbose=False,
    )
    scores = cross_val_score(reg_cv, X, y, cv=3, scoring="r2")
    print(f"  R^2 por fold : {np.round(scores, 4)}")
    print(f"  R^2 promedio : {scores.mean():.4f}")

    # ------------------------------------------------------------------
    # 6. Con optimización de constantes activada
    # ------------------------------------------------------------------
    print("\n--- Con use_constant_optimization=True ---\n")

    X2 = np.random.uniform(-5, 5, size=(80, 2))
    y2 = X2[:, 0] * X2[:, 1] + 1.0

    reg_ls = SymbolicRegressor(
        population_size=150,
        generations=20,
        const_range=(-2.0, 2.0),
        use_constant_optimization=True,
        local_search_every=5,
        local_search_top_k=5,
        random_state=5,
        verbose=False,
    )
    reg_ls.fit(X2, y2)
    print(f"  Expresión : {reg_ls.get_expression()}")
    print(f"  R^2        : {reg_ls.score(X2, y2):.6f}")

    print("\n[OK] API estilo scikit-learn completada.")