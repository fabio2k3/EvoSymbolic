"""
Fase 2 — Evaluación de Aptitud (Fitness)
=========================================
Depende de: phase1_expression_tree.py

Dado un árbol de expresión y los datos reales (X, y),
calcula qué tan buena es la expresión.

Contenido:
  - FitnessEvaluator  → calcula MSE, RMSE, MAE, R²
  - parsimony_penalty → penaliza árboles grandes
  - evaluate_population → evalúa una lista de árboles
  - casos degenerados → árbol constante, NaN, Inf

Convención: fitness MÁS BAJO = MEJOR individuo.
"""

import numpy as np
from typing import List, Tuple, Optional
from expression_tree import ExpressionTree, TreeGenerator
import random

# ---------------------------------------------------------------------------
# 1. MÉTRICAS BASE
# ---------------------------------------------------------------------------

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error — métrica de fitness principal.
    Penaliza errores grandes de forma cuadrática.
    Rango: [0, +inf)  →  0 = predicción perfecta.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error — misma escala que y.
    Más interpretable que MSE para comunicar resultados.
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error — robusto a outliers.
    Penaliza todos los errores por igual (no cuadrático).
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coeficiente de determinación R².
    Rango: (-inf, 1]  →  1 = predicción perfecta, 0 = media basal.
    Valores negativos: el árbol es PEOR que predecir siempre la media.

    Nota: en GP lo usamos como métrica de reporte, no como fitness
    directo, porque maximizar R² = minimizar MSE normalizado.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10:
        # y_true es constante: R² indefinido, devolvemos 1 si predice exacto
        return 1.0 if ss_res < 1e-10 else 0.0
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# 2. PARSIMONY PENALTY
# ---------------------------------------------------------------------------

def parsimony_penalty(
    base_fitness:    float,
    tree_size:       int,
    parsimony_coeff: float = 0.001,
) -> float:
    """
    Penalización por parsimonia: árboles más grandes pagan un coste extra.

    fitness_final = base_fitness + parsimony_coeff * tree_size

    Propósito:
      - Evita el "bloat": el crecimiento descontrolado del árbol a lo largo
        de las generaciones (fenómeno clásico en GP).
      - Favorece soluciones simples entre candidatos de igual calidad.
      - parsimony_coeff pequeño (0.001–0.01) da preferencia ligera a
        árboles compactos sin sacrificar fitness real.

    Ejemplo:
      árbol A: MSE=0.05, tamaño=8  →  fitness = 0.05 + 0.001*8  = 0.058
      árbol B: MSE=0.05, tamaño=30 →  fitness = 0.05 + 0.001*30 = 0.080
      → Se elige A aunque ambos ajustan igual de bien.
    """
    return base_fitness + parsimony_coeff * tree_size


# ---------------------------------------------------------------------------
# 3. EVALUADOR DE FITNESS
# ---------------------------------------------------------------------------

METRIC_FUNCTIONS = {
    "mse":  mse,
    "rmse": rmse,
    "mae":  mae,
}

PENALTY_VALUE = 1e10  # fitness asignado cuando la evaluación falla


class FitnessEvaluator:
    """
    Evalúa la aptitud de un árbol de expresión sobre datos (X, y).

    Parámetros
    ----------
    metric          : str    → métrica a minimizar ('mse', 'rmse', 'mae')
    parsimony_coeff : float  → coeficiente de penalización por tamaño
                               0 = sin penalización

    Uso
    ---
    evaluator = FitnessEvaluator(metric='mse', parsimony_coeff=0.001)
    fitness   = evaluator.evaluate(tree, X, y)
    """

    def __init__(
        self,
        metric:          str   = "mse",
        parsimony_coeff: float = 0.001,
    ):
        if metric not in METRIC_FUNCTIONS:
            raise ValueError(f"Métrica '{metric}' no válida. Elige entre: {list(METRIC_FUNCTIONS)}")
        self.metric          = metric
        self.metric_fn       = METRIC_FUNCTIONS[metric]
        self.parsimony_coeff = parsimony_coeff

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------
    def evaluate(
        self,
        tree:   ExpressionTree,
        X:      np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """
        Calcula el fitness del árbol.
        Devuelve PENALTY_VALUE (1e10) ante cualquier fallo numérico.

        Pasos:
          1. Evalúa el árbol → y_pred
          2. Detecta casos degenerados
          3. Calcula métrica base
          4. Aplica parsimony penalty
        """
        try:
            y_pred = tree.evaluate(X)  # ya protege NaN/Inf internamente
        except Exception:
            return PENALTY_VALUE

        # --- Detección de casos degenerados ---
        if not self._is_valid_prediction(y_pred, y_true):
            return PENALTY_VALUE

        # --- Métrica base ---
        base = self.metric_fn(y_true, y_pred)

        # Si la métrica en sí da NaN/Inf (y_true constante con MAE=0, etc.)
        if not np.isfinite(base):
            return PENALTY_VALUE

        # --- Parsimony ---
        return parsimony_penalty(base, tree.size(), self.parsimony_coeff)

    def evaluate_with_details(
        self,
        tree:   ExpressionTree,
        X:      np.ndarray,
        y_true: np.ndarray,
    ) -> dict:
        """
        Igual que evaluate() pero devuelve todas las métricas para análisis.

        Útil durante el desarrollo y para reportar resultados en la tesis.
        """
        try:
            y_pred = tree.evaluate(X)
        except Exception:
            return self._failed_details(tree)

        if not self._is_valid_prediction(y_pred, y_true):
            return self._failed_details(tree)

        base       = self.metric_fn(y_true, y_pred)
        fitness    = parsimony_penalty(base, tree.size(), self.parsimony_coeff)

        return {
            "fitness":    fitness,
            "base_metric": base,
            "metric_name": self.metric,
            "mse":         mse(y_true, y_pred),
            "rmse":        rmse(y_true, y_pred),
            "mae":         mae(y_true, y_pred),
            "r2":          r2_score(y_true, y_pred),
            "size":        tree.size(),
            "depth":       tree.depth(),
            "expression":  tree.to_string(),
            "valid":       True,
        }

    # ------------------------------------------------------------------
    # Detección de predicciones degeneradas
    # ------------------------------------------------------------------
    def _is_valid_prediction(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> bool:
        """
        Devuelve False si y_pred es inutilizable. Casos:

        1. Contiene NaN o Inf después de evaluate()
           → las funciones protegidas deberían evitarlo, pero por si acaso.

        2. Es completamente constante (varianza ≈ 0)
           → el árbol colapsa a un número, no modela nada.
           → Excepción: si y_true también es constante y coincide, es válido.

        3. Rango extremo (max > 1e8)
           → indica overflow no capturado; el MSE sería artificialmente enorme.
        """
        # 1. NaN / Inf
        if not np.all(np.isfinite(y_pred)):
            return False

        # 2. Predicción constante
        pred_std = np.std(y_pred)
        if pred_std < 1e-10:
            # Permitir solo si y_true también es (casi) constante e igual
            true_std   = np.std(y_true)
            pred_mean  = np.mean(y_pred)
            true_mean  = np.mean(y_true)
            if true_std < 1e-10 and abs(pred_mean - true_mean) < 1e-6:
                return True   # raro, pero válido
            return False

        # 3. Overflow
        if np.max(np.abs(y_pred)) > 1e8:
            return False

        return True

    def _failed_details(self, tree: ExpressionTree) -> dict:
        return {
            "fitness":     PENALTY_VALUE,
            "base_metric": PENALTY_VALUE,
            "metric_name": self.metric,
            "mse":         PENALTY_VALUE,
            "rmse":        PENALTY_VALUE,
            "mae":         PENALTY_VALUE,
            "r2":          -PENALTY_VALUE,
            "size":        tree.size(),
            "depth":       tree.depth(),
            "expression":  tree.to_string(),
            "valid":       False,
        }


# ---------------------------------------------------------------------------
# 4. EVALUACIÓN DE POBLACIÓN
# ---------------------------------------------------------------------------

def evaluate_population(
    population:  List[ExpressionTree],
    X:           np.ndarray,
    y_true:      np.ndarray,
    evaluator:   FitnessEvaluator,
) -> List[float]:
    """
    Evalúa todos los árboles de una población y devuelve sus fitness.

    La lista de fitness tiene el mismo orden que la población.
    Este orden es fundamental para selección (Fase 3).

    Retorna: lista de floats, uno por árbol.
    """
    return [evaluator.evaluate(tree, X, y_true) for tree in population]


def best_individual(
    population: List[ExpressionTree],
    fitnesses:  List[float],
) -> Tuple[ExpressionTree, float]:
    """Devuelve (mejor_árbol, mejor_fitness)."""
    best_idx = int(np.argmin(fitnesses))
    return population[best_idx], fitnesses[best_idx]


# ---------------------------------------------------------------------------
# 5. DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  FASE 2 — Evaluación de Aptitud (Fitness)")
    print("=" * 60)

    # Datos de ejemplo: y = x² + sin(x)
    np.random.seed(0)
    X      = np.linspace(-3, 3, 40).reshape(-1, 1)
    y_true = X[:, 0] ** 2 + np.sin(X[:, 0])

    evaluator = FitnessEvaluator(metric="mse", parsimony_coeff=0.001)

    # ------------------------------------------------------------------
    # 5.1  Árbol perfecto: x² + sin(x)
    # ------------------------------------------------------------------
    from expression_tree import (
        ExpressionTree, OperatorNode, FunctionNode, Variable, Constant
    )

    perfect_tree = ExpressionTree(
        OperatorNode("add",
            OperatorNode("mul", Variable(0), Variable(0)),
            FunctionNode("sin", Variable(0))
        )
    )
    details_perfect = evaluator.evaluate_with_details(perfect_tree, X, y_true)
    print("\n--- Árbol perfecto: x² + sin(x) ---")
    print(f"  Expresión : {details_perfect['expression']}")
    print(f"  MSE       : {details_perfect['mse']:.6f}")
    print(f"  RMSE      : {details_perfect['rmse']:.6f}")
    print(f"  MAE       : {details_perfect['mae']:.6f}")
    print(f"  R²        : {details_perfect['r2']:.6f}")
    print(f"  Fitness   : {details_perfect['fitness']:.6f}  (mse + 0.001 × {details_perfect['size']} nodos)")

    # ------------------------------------------------------------------
    # 5.2  Árbol aproximado: x²
    # ------------------------------------------------------------------
    approx_tree = ExpressionTree(
        OperatorNode("mul", Variable(0), Variable(0))
    )
    details_approx = evaluator.evaluate_with_details(approx_tree, X, y_true)
    print("\n--- Árbol aproximado: x² ---")
    print(f"  Expresión : {details_approx['expression']}")
    print(f"  MSE       : {details_approx['mse']:.6f}")
    print(f"  R²        : {details_approx['r2']:.6f}")
    print(f"  Fitness   : {details_approx['fitness']:.6f}")

    # ------------------------------------------------------------------
    # 5.3  Árbol malo: constante 0.5
    # ------------------------------------------------------------------
    bad_tree = ExpressionTree(Constant(0.5))
    details_bad = evaluator.evaluate_with_details(bad_tree, X, y_true)
    print("\n--- Árbol malo: constante 0.5 ---")
    print(f"  Expresión : {details_bad['expression']}")
    print(f"  Válido    : {details_bad['valid']}")
    print(f"  Fitness   : {details_bad['fitness']:.2e}  ← penalización máxima")

    # ------------------------------------------------------------------
    # 5.4  Casos degenerados
    # ------------------------------------------------------------------
    print("\n--- Casos degenerados ---")

    # División por cero sin protección manual (las protecciones las maneja Fase 1)
    div_zero = ExpressionTree(OperatorNode("div", Variable(0), Constant(0.0)))
    f_dz = evaluator.evaluate(div_zero, X, y_true)
    print(f"  x / 0.0           → fitness = {f_dz:.2e}")

    # Árbol que produce overflow
    big_exp = ExpressionTree(FunctionNode("exp", FunctionNode("exp", Variable(0))))
    f_big = evaluator.evaluate(big_exp, X, y_true)
    print(f"  exp(exp(x))       → fitness = {f_big:.2e}")

    # ------------------------------------------------------------------
    # 5.5  Parsimony: dos árboles con igual MSE, distinto tamaño
    # ------------------------------------------------------------------
    print("\n--- Parsimony pressure ---")

    # Árbol A: (x * x) — tamaño 3
    tree_a = ExpressionTree(OperatorNode("mul", Variable(0), Variable(0)))
    # Árbol B: ((x * x) + 0) — tamaño 5, semánticamente igual
    tree_b = ExpressionTree(
        OperatorNode("add",
            OperatorNode("mul", Variable(0), Variable(0)),
            Constant(0.0)
        )
    )
    # Usamos los datos de x² para que ambos sean casi perfectos
    y_sq = X[:, 0] ** 2
    fa = evaluator.evaluate(tree_a, X, y_sq)
    fb = evaluator.evaluate(tree_b, X, y_sq)
    print(f"  A: (x0 * x0)         size={tree_a.size()}  fitness={fa:.5f}")
    print(f"  B: ((x0 * x0) + 0)   size={tree_b.size()}  fitness={fb:.5f}")
    print(f"  ¿Se prefiere A?: {fa < fb}  ← parsimony elige el más simple")

    # ------------------------------------------------------------------
    # 5.6  Evaluación de una población aleatoria
    # ------------------------------------------------------------------
    print("\n--- Población aleatoria (10 árboles) ---")
    random.seed(7)
    gen = TreeGenerator(n_features=1)
    population = [gen.ramped_half_and_half(1, 4) for _ in range(10)]
    fitnesses  = evaluate_population(population, X, y_true, evaluator)

    for i, (tree, fit) in enumerate(zip(population, fitnesses)):
        r2  = r2_score(y_true, tree.evaluate(X))
        tag = "INVALIDO" if fit >= 1e9 else f"R²={r2:+.3f}"
        print(f"  [{i:2d}] fit={fit:10.4f}  size={tree.size():2d}  {tag}  {tree.to_string()[:50]}")

    best_tree, best_fit = best_individual(population, fitnesses)
    print(f"\n  Mejor individuo:")
    print(f"    {best_tree.to_string()}")
    print(f"    fitness = {best_fit:.5f}")

    print("\n✓ Fase 2 completada. Siguiente: Fase 3 — Operadores genéticos.")
