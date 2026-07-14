"""
Variable Neighborhood Search (VNS)
====================================
Módulo de búsqueda local avanzada para EvoSymbolic.

VNS es una metaheurística que explora sistemáticamente distintas
"vecindades" de una solución. La idea central es:

  - Si una vecindad pequeña no mejora → prueba una más grande.
  - Si alguna vecindad mejora         → vuelve a empezar desde la pequeña.

Esto evita quedarse atascado en mínimos locales que la búsqueda local
clásica (solo scipy sobre constantes) no puede escapar, porque aquí
también se modifica la ESTRUCTURA del árbol, no solo sus constantes.

Vecindades implementadas (de menor a mayor perturbación):
  k=1  point_mutation       -> cambia UN operador/función (mínima perturbación)
  k=2  subtree_mutation     -> reemplaza un subárbol pequeño (perturbación media)
  k=3  subtree_mutation grande -> reemplaza un subárbol hasta profundidad 3
  k=4  optimize_constants   -> scipy sobre las constantes del árbol perturbado

Flujo VNS estándar (Basic VNS / BVNS):
  k = 1
  mientras k <= k_max:
      x' = shake(x, k)          # perturbación aleatoria en vecindad k
      x'' = local_search(x')    # mejora local (aquí: optimize_constants)
      if f(x'') < f(x):
          x = x''
          k = 1                 # vuelve a la vecindad más pequeña
      else:
          k = k + 1             # prueba la siguiente vecindad

Integración con EvoSymbolic:
  - Se aplica al MEJOR individuo de cada generación (o top_k)
    cada vns_every generaciones dentro del bucle GP.
  - No reemplaza a GeneticOperators, lo complementa:
      GP     → explora el espacio de estructuras (exploración global)
      scipy  → afina constantes numéricas (explotación numérica)
      VNS    → escapa de mínimos locales estructurales (puente entre ambos)
"""

import random
import numpy as np
from typing import List, Tuple, Optional

from expression_tree import ExpressionTree, TreeGenerator
from fitness import mse, FitnessEvaluator
from genetic_operators import (
    point_mutation,
    subtree_mutation,
    constant_mutation,
)
from constant_optimization import optimize_constants, extract_constants


# ---------------------------------------------------------------------------
# 1. VECINDADES (shake functions)
# ---------------------------------------------------------------------------

def _neighborhood_1(tree: ExpressionTree, generator: TreeGenerator) -> ExpressionTree:
    """
    Vecindad k=1 — perturbación mínima.
    Cambia UN solo operador o función (point_mutation).
    El árbol resultante tiene la misma estructura, solo cambia
    un símbolo interno.
    """
    return point_mutation(tree, generator)


def _neighborhood_2(tree: ExpressionTree, generator: TreeGenerator) -> ExpressionTree:
    """
    Vecindad k=2 — perturbación media.
    Reemplaza un subárbol de profundidad máxima 2.
    Modifica una rama del árbol manteniendo el esqueleto global.
    """
    return subtree_mutation(tree, generator, max_depth=generator.n_features + 4)


def _neighborhood_3(tree: ExpressionTree, generator: TreeGenerator) -> ExpressionTree:
    """
    Vecindad k=3 — perturbación alta.
    Reemplaza un subárbol de profundidad máxima 3 (más disruptivo).
    Puede cambiar partes importantes de la estructura.
    """
    mutant = tree.clone()
    from expression_tree import Variable, Constant, FunctionNode, OperatorNode
    import copy

    nodes = mutant.get_all_nodes()
    # Sesgo ligero hacia nodos internos para perturbaciones más ricas
    internals = [(n, p, pos) for (n, p, pos) in nodes if not n.is_terminal()]
    if internals:
        target_node, parent, position = random.choice(internals)
    else:
        target_node, parent, position = random.choice(nodes)

    new_subtree = generator.ramped_half_and_half(min_depth=1, max_depth=3).root

    from genetic_operators import _replace_node
    _replace_node(mutant, target_node, new_subtree)

    max_d = generator.n_features + 5
    if mutant.depth() > max_d:
        return tree.clone()

    return mutant


def _neighborhood_4(tree: ExpressionTree, generator: TreeGenerator) -> ExpressionTree:
    """
    Vecindad k=4 — perturbación máxima estructural.
    Combina subtree_mutation k=3 + perturbación gaussiana de constantes.
    Útil cuando las vecindades anteriores se han agotado.
    """
    mutant = _neighborhood_3(tree, generator)
    mutant = constant_mutation(mutant, generator, sigma=2.0)
    return mutant


# Registro de vecindades ordenadas de menor a mayor perturbación
NEIGHBORHOODS = [
    _neighborhood_1,
    _neighborhood_2,
    _neighborhood_3,
    _neighborhood_4,
]


# ---------------------------------------------------------------------------
# 2. BÚSQUEDA LOCAL INTERNA DE VNS
# ---------------------------------------------------------------------------

def _local_search(
    tree:    ExpressionTree,
    X:       np.ndarray,
    y_true:  np.ndarray,
    method:  str = "Nelder-Mead",
    maxiter: int = 50,
) -> Tuple[ExpressionTree, float]:
    """
    Paso de mejora local dentro de VNS.

    Aplica optimize_constants (scipy) al árbol perturbado.
    Si el árbol no tiene constantes, devuelve el árbol tal cual
    con su MSE calculado (sin llamar a scipy, costo cero).

    Devuelve (árbol_mejorado, mse_final).
    """
    const_nodes, _ = extract_constants(tree)

    if len(const_nodes) > 0:
        improved_tree, _ = optimize_constants(tree, X, y_true, method=method, maxiter=maxiter)
    else:
        improved_tree = tree.clone()

    try:
        y_pred = improved_tree.evaluate(X)
        if not np.all(np.isfinite(y_pred)):
            return tree.clone(), 1e10
        current_mse = mse(y_true, y_pred)
    except Exception:
        return tree.clone(), 1e10

    return improved_tree, current_mse


# ---------------------------------------------------------------------------
# 3. BUCLE VNS PRINCIPAL (Basic VNS)
# ---------------------------------------------------------------------------

def vns(
    tree:       ExpressionTree,
    X:          np.ndarray,
    y_true:     np.ndarray,
    generator:  TreeGenerator,
    k_max:      int   = 4,
    max_iter:   int   = 10,
    ls_method:  str   = "Nelder-Mead",
    ls_maxiter: int   = 50,
) -> Tuple[ExpressionTree, float, int]:
    """
    Basic Variable Neighborhood Search (BVNS) sobre un árbol de expresión.

    Parámetros
    ----------
    tree      : ExpressionTree
        Árbol de partida (solución actual).
    X, y_true : np.ndarray
        Datos de entrenamiento.
    generator : TreeGenerator
        Generador de árboles aleatorios para las vecindades.
    k_max     : int
        Número máximo de vecindades a explorar (1..k_max).
    max_iter  : int
        Número máximo de iteraciones del bucle VNS completo.
        Cada iteración puede probar hasta k_max vecindades.
    ls_method : str
        Método de scipy para la búsqueda local interna.
    ls_maxiter: int
        Iteraciones máximas de scipy en cada paso local.

    Devuelve
    --------
    (best_tree, best_mse, n_improvements)
      best_tree      : mejor árbol encontrado durante la búsqueda
      best_mse       : su MSE (sin parsimony, para comparación pura)
      n_improvements : cuántas veces se encontró una mejora
    """
    # Evaluar la solución de partida
    try:
        y0 = tree.evaluate(X)
        current_mse = mse(y_true, y0) if np.all(np.isfinite(y0)) else 1e10
    except Exception:
        current_mse = 1e10

    current_tree  = tree.clone()
    best_tree     = tree.clone()
    best_mse      = current_mse
    n_improvements = 0

    k_max_actual = min(k_max, len(NEIGHBORHOODS))

    for _ in range(max_iter):
        k = 0  # índice 0-based internamente

        while k < k_max_actual:
            # ── Shake: perturbación aleatoria en vecindad k ──────────────
            neighbor_fn = NEIGHBORHOODS[k]
            candidate   = neighbor_fn(current_tree, generator)

            # ── Local search: optimizar constantes del candidato ─────────
            candidate_improved, candidate_mse = _local_search(
                candidate, X, y_true,
                method=ls_method, maxiter=ls_maxiter,
            )

            # ── Move or not ──────────────────────────────────────────────
            if candidate_mse < current_mse:
                # Mejora encontrada: actualizamos solución actual y
                # volvemos a la vecindad más pequeña
                current_tree = candidate_improved
                current_mse  = candidate_mse
                n_improvements += 1

                if current_mse < best_mse:
                    best_tree = current_tree.clone()
                    best_mse  = current_mse

                k = 0  # reiniciar desde vecindad más pequeña
            else:
                # Sin mejora: probar vecindad más grande
                k += 1

    return best_tree, best_mse, n_improvements


# ---------------------------------------------------------------------------
# 4. VNS SOBRE UNA POBLACIÓN (top_k individuos)
# ---------------------------------------------------------------------------

def vns_population(
    population:  List[ExpressionTree],
    fitnesses:   List[float],
    X:           np.ndarray,
    y_true:      np.ndarray,
    evaluator:   FitnessEvaluator,
    generator:   TreeGenerator,
    top_k:       int   = 5,
    k_max:       int   = 4,
    vns_iter:    int   = 5,
    ls_method:   str   = "Nelder-Mead",
    ls_maxiter:  int   = 50,
) -> Tuple[List[ExpressionTree], List[float], int]:
    """
    Aplica VNS a los `top_k` mejores individuos de la población.

    Solo se tocan los mejores para controlar el coste computacional:
    VNS sobre un árbol implica múltiples evaluaciones (shake + local
    search por iteración), así que aplicarlo a todos sería prohibitivo.

    Parámetros
    ----------
    top_k    : cuántos individuos reciben VNS
    k_max    : vecindades máximas por individuo
    vns_iter : iteraciones del bucle VNS por individuo

    Devuelve
    --------
    (new_population, new_fitnesses, n_total_improvements)
    """
    new_population = list(population)
    new_fitnesses  = list(fitnesses)
    n_total        = 0

    # Seleccionar los top_k índices con menor fitness
    indices = np.argsort(fitnesses)[:top_k]

    for i in indices:
        tree_i = population[i]

        improved_tree, improved_mse, n_imp = vns(
            tree       = tree_i,
            X          = X,
            y_true     = y_true,
            generator  = generator,
            k_max      = k_max,
            max_iter   = vns_iter,
            ls_method  = ls_method,
            ls_maxiter = ls_maxiter,
        )

        # Solo reemplazar si el fitness con parsimony también mejora
        new_fit = evaluator.evaluate(improved_tree, X, y_true)
        if new_fit < new_fitnesses[i]:
            new_population[i] = improved_tree
            new_fitnesses[i]  = new_fit
            n_total          += n_imp

    return new_population, new_fitnesses, n_total
