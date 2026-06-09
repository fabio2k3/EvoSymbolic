"""
Operadores Genéticos
==============================
Depende de: phase1_expression_tree.py  /  phase2_fitness.py

Implementa los tres operadores que hacen evolucionar la población:

  1. Selección por torneo   → elige padres aptos para reproducirse
  2. Crossover de subárboles → combina dos padres en un hijo
  3. Mutación               → introduce variación en un individuo
     · subtree_mutation     → reemplaza un subárbol al azar
     · point_mutation       → cambia un operador o función
     · constant_mutation    → perturba una constante

  + GeneticOperators        → clase que agrupa todo con probabilidades
"""

import random
import copy
import numpy as np
from typing import List, Tuple, Optional

from expression_tree import (
    ExpressionTree, Node,
    Variable, Constant, FunctionNode, OperatorNode,
    TreeGenerator,
    BINARY_OPERATORS, UNARY_FUNCTIONS,
)
from fitness import FitnessEvaluator, evaluate_population

MAX_DEPTH_DEFAULT = 6   # profundidad máxima permitida tras un operador


# ---------------------------------------------------------------------------
# UTILIDAD: reemplazar un nodo dentro de un árbol
# ---------------------------------------------------------------------------

def _replace_node(tree: ExpressionTree, target: Node, new_node: Node) -> None:
    """
    Modifica el árbol IN-PLACE: busca 'target' y lo reemplaza por 'new_node'.
    Usa get_all_nodes() que ya devuelve (nodo, padre, posición).
    """
    for node, parent, position in tree.get_all_nodes():
        if node is target:
            if position == "root":
                tree.root = new_node
            elif position == "left":
                parent.left = new_node
            elif position == "right":
                parent.right = new_node
            elif position == "child":
                parent.child = new_node
            return


# ---------------------------------------------------------------------------
# 1. SELECCIÓN POR TORNEO
# ---------------------------------------------------------------------------

def tournament_selection(
    population: List[ExpressionTree],
    fitnesses:  List[float],
    k:          int = 3,
) -> ExpressionTree:
    """
    Selección por torneo de tamaño k.

    Algoritmo:
      1. Elige k individuos al azar de la población (con reemplazo).
      2. Devuelve el que tiene menor fitness (recordamos: minimizamos).

    Ventaja frente a la ruleta:
      · No requiere que todos los fitness sean positivos.
      · La presión selectiva se controla fácilmente con k:
          k=2  →  presión baja  (más exploración)
          k=7  →  presión alta  (más explotación)

    Devuelve una REFERENCIA al individuo ganador (no una copia).
    El motor GP hará clone() antes de aplicar operadores.
    """
    candidates_idx = random.choices(range(len(population)), k=k)
    best_idx = min(candidates_idx, key=lambda i: fitnesses[i])
    return population[best_idx]


# ---------------------------------------------------------------------------
# 2. CROSSOVER DE SUBÁRBOLES
# ---------------------------------------------------------------------------

def subtree_crossover(
    parent_a:  ExpressionTree,
    parent_b:  ExpressionTree,
    max_depth: int = MAX_DEPTH_DEFAULT,
) -> Tuple[ExpressionTree, ExpressionTree]:
    """
    Crossover de subárboles estándar (Koza 1992).

    Algoritmo:
      1. Copia profunda de ambos padres → hijo_a, hijo_b
      2. Elige un nodo aleatorio en hijo_a  → punto_a
      3. Elige un nodo aleatorio en hijo_b  → punto_b
      4. Intercambia los subárboles:
           hijo_a: punto_a ← subárbol de punto_b
           hijo_b: punto_b ← subárbol de punto_a
      5. Si algún hijo supera max_depth, devuelve el padre original (fallback)

    Probabilidad interna: 90% nodos internos / 10% terminales.
    (Koza observó que el crossover en hojas produce poca diversidad.)

    Devuelve (hijo_a, hijo_b) — siempre dos hijos nuevos.
    """
    child_a = parent_a.clone()
    child_b = parent_b.clone()

    # Obtener todos los nodos de cada hijo
    nodes_a = child_a.get_all_nodes()   # [(nodo, padre, pos), ...]
    nodes_b = child_b.get_all_nodes()

    # Seleccionar puntos de corte con sesgo hacia nodos internos
    point_a = _biased_node_pick(nodes_a)
    point_b = _biased_node_pick(nodes_b)

    # Extraer los subárboles (copias profundas para intercambio)
    subtree_a = copy.deepcopy(point_a[0])
    subtree_b = copy.deepcopy(point_b[0])

    # Intercambiar
    _replace_node(child_a, point_a[0], subtree_b)
    _replace_node(child_b, point_b[0], subtree_a)

    # Fallback si se supera max_depth
    if child_a.depth() > max_depth:
        child_a = parent_a.clone()
    if child_b.depth() > max_depth:
        child_b = parent_b.clone()

    return child_a, child_b


def _biased_node_pick(nodes: list, p_internal: float = 0.9):
    """
    Elige un nodo con sesgo: p_internal de probabilidad de nodo interno.
    Si no hay nodos internos (árbol de un solo nodo), devuelve el único.
    """
    internals  = [(n, p, pos) for (n, p, pos) in nodes if not n.is_terminal()]
    terminals  = [(n, p, pos) for (n, p, pos) in nodes if n.is_terminal()]

    if not internals:
        return random.choice(terminals)
    if not terminals:
        return random.choice(internals)

    if random.random() < p_internal:
        return random.choice(internals)
    else:
        return random.choice(terminals)


# ---------------------------------------------------------------------------
# 3. MUTACIÓN
# ---------------------------------------------------------------------------

def subtree_mutation(
    tree:       ExpressionTree,
    generator:  TreeGenerator,
    max_depth:  int = MAX_DEPTH_DEFAULT,
) -> ExpressionTree:
    """
    Mutación de subárbol.

    Algoritmo:
      1. Clona el árbol.
      2. Elige un nodo al azar (sesgado a internos).
      3. Genera un nuevo subárbol aleatorio de profundidad <= 2.
      4. Reemplaza el nodo elegido por el nuevo subárbol.
      5. Fallback si supera max_depth.

    Es la mutación más disruptiva: puede cambiar la estructura radicalmente.
    Equivale a un crossover con un árbol generado al azar.
    """
    mutant = tree.clone()
    nodes  = mutant.get_all_nodes()
    target = _biased_node_pick(nodes)

    new_subtree = generator.ramped_half_and_half(
        min_depth=0, max_depth=2
    ).root

    _replace_node(mutant, target[0], new_subtree)

    if mutant.depth() > max_depth:
        return tree.clone()   # fallback al original

    return mutant


def point_mutation(
    tree:      ExpressionTree,
    generator: TreeGenerator,
) -> ExpressionTree:
    """
    Mutación de punto (point mutation).

    Reemplaza UN solo nodo por otro del mismo tipo y aridad:
      · OperatorNode  →  otro operador binario al azar
      · FunctionNode  →  otra función unaria al azar
      · Constant      →  nueva constante aleatoria en const_range
      · Variable      →  otra variable al azar (si n_features > 1)

    Es la mutación más conservadora: preserva la estructura del árbol.
    Útil para ajuste fino de expresiones ya cercanas a la solución.
    """
    mutant = tree.clone()
    all_nodes = mutant.get_all_nodes()

    # Elige cualquier nodo (sin sesgo especial)
    target_node, parent, position = random.choice(all_nodes)

    new_node = _mutate_node(target_node, generator)
    _replace_node(mutant, target_node, new_node)

    return mutant


def _mutate_node(node: Node, generator: TreeGenerator) -> Node:
    """Devuelve un nodo nuevo del mismo tipo y aridad que el original."""

    if isinstance(node, OperatorNode):
        ops = [k for k in generator.operators if k != node.name]
        if not ops:
            return copy.deepcopy(node)
        new_op = random.choice(ops)
        # Conserva los hijos — solo cambia el operador
        return OperatorNode(new_op, copy.deepcopy(node.left), copy.deepcopy(node.right))

    if isinstance(node, FunctionNode):
        fns = [k for k in generator.functions if k != node.name]
        if not fns:
            return copy.deepcopy(node)
        new_fn = random.choice(fns)
        return FunctionNode(new_fn, copy.deepcopy(node.child))

    if isinstance(node, Constant):
        new_val = random.uniform(*generator.const_range)
        return Constant(new_val)

    if isinstance(node, Variable):
        if generator.n_features > 1:
            new_idx = random.choice(
                [i for i in range(generator.n_features) if i != node.index]
            )
            return Variable(new_idx)
        return copy.deepcopy(node)

    return copy.deepcopy(node)


def constant_mutation(
    tree:       ExpressionTree,
    generator:  TreeGenerator,
    sigma:      float = 0.5,
) -> ExpressionTree:
    """
    Mutación de constante: perturba una constante existente con ruido gaussiano.

    Si el árbol no tiene constantes, devuelve una copia sin cambios.

    sigma controla la magnitud de la perturbación:
      · sigma pequeño (0.1)  →  ajuste fino
      · sigma grande (2.0)   →  salto brusco

    Este operador es clave cuando hay constantes en el árbol: sin él,
    las constantes generadas aleatoriamente en la inicialización nunca
    se ajustan (eso se hace bien en la Fase 5 con scipy, pero esta
    mutación da empuje básico sin optimización).
    """
    mutant    = tree.clone()
    all_nodes = mutant.get_all_nodes()
    constants = [(n, p, pos) for (n, p, pos) in all_nodes if isinstance(n, Constant)]

    if not constants:
        return mutant

    target_node, parent, position = random.choice(constants)
    new_val  = target_node.value + random.gauss(0, sigma)
    new_node = Constant(new_val)
    _replace_node(mutant, target_node, new_node)

    return mutant


# ---------------------------------------------------------------------------
# 4. CLASE UNIFICADORA: GeneticOperators
# ---------------------------------------------------------------------------

class GeneticOperators:
    """
    Agrupa selección, crossover y mutación con sus probabilidades.

    Parámetros
    ----------
    generator       : TreeGenerator  → para generar nuevos subárboles
    tournament_size : int            → k para selección por torneo
    p_crossover     : float          → prob. de aplicar crossover
    p_subtree_mut   : float          → prob. de mutación de subárbol
    p_point_mut     : float          → prob. de mutación de punto
    p_constant_mut  : float          → prob. de mutación de constante
    max_depth       : int            → profundidad máxima tras operadores
    elite_size      : int            → cuántos mejores pasan sin operadores

    Nota sobre probabilidades:
      Las tres mutaciones se aplican de forma independiente (no excluyente).
      Un individuo puede sufrir subtree + point en el mismo paso.
      p_crossover es la probabilidad de que el hijo venga de crossover;
      si no hay crossover, se reproduce sin cruce.
    """

    def __init__(
        self,
        generator:       TreeGenerator,
        tournament_size: int   = 3,
        p_crossover:     float = 0.80,
        p_subtree_mut:   float = 0.10,
        p_point_mut:     float = 0.05,
        p_constant_mut:  float = 0.05,
        max_depth:       int   = MAX_DEPTH_DEFAULT,
        elite_size:      int   = 1,
    ):
        self.generator       = generator
        self.tournament_size = tournament_size
        self.p_crossover     = p_crossover
        self.p_subtree_mut   = p_subtree_mut
        self.p_point_mut     = p_point_mut
        self.p_constant_mut  = p_constant_mut
        self.max_depth       = max_depth
        self.elite_size      = elite_size

    def evolve(
        self,
        population: List[ExpressionTree],
        fitnesses:  List[float],
    ) -> List[ExpressionTree]:
        """
        Genera una nueva generación completa del mismo tamaño.

        Pasos:
          1. Elitismo: copia los `elite_size` mejores directamente.
          2. Para cada lugar restante:
             a. Selecciona padre(s) por torneo.
             b. Aplica crossover con prob. p_crossover.
             c. Aplica mutaciones independientemente.
          3. Devuelve la nueva generación.
        """
        pop_size    = len(population)
        new_pop     = []

        # --- Elitismo ---
        elite = self._get_elite(population, fitnesses)
        new_pop.extend(elite)

        # --- Generar el resto ---
        while len(new_pop) < pop_size:
            # Selección
            parent_a = tournament_selection(population, fitnesses, self.tournament_size)

            # Crossover o reproducción
            if random.random() < self.p_crossover:
                parent_b = tournament_selection(population, fitnesses, self.tournament_size)
                child, _ = subtree_crossover(parent_a, parent_b, self.max_depth)
            else:
                child = parent_a.clone()

            # Mutaciones (independientes entre sí)
            if random.random() < self.p_subtree_mut:
                child = subtree_mutation(child, self.generator, self.max_depth)
            if random.random() < self.p_point_mut:
                child = point_mutation(child, self.generator)
            if random.random() < self.p_constant_mut:
                child = constant_mutation(child, self.generator)

            new_pop.append(child)

        return new_pop[:pop_size]

    def _get_elite(
        self,
        population: List[ExpressionTree],
        fitnesses:  List[float],
    ) -> List[ExpressionTree]:
        """Devuelve copias profundas de los `elite_size` mejores."""
        sorted_idx = np.argsort(fitnesses)[:self.elite_size]
        return [population[i].clone() for i in sorted_idx]


# ---------------------------------------------------------------------------
# 5. DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  FASE 3 — Operadores Genéticos")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    gen = TreeGenerator(n_features=1, const_range=(-3.0, 3.0))

    # ------------------------------------------------------------------
    # 5.1  Selección por torneo
    # ------------------------------------------------------------------
    print("\n--- Selección por torneo (k=3) ---")
    pop   = [gen.ramped_half_and_half(1, 3) for _ in range(8)]
    fits  = [round(random.uniform(0.5, 50.0), 2) for _ in range(8)]

    print("  Población:")
    for i, (t, f) in enumerate(zip(pop, fits)):
        print(f"    [{i}] fit={f:5.2f}  {t.to_string()[:40]}")

    winners = [tournament_selection(pop, fits, k=3) for _ in range(5)]
    avg_fit = sum(fits[pop.index(w)] for w in winners) / 5
    print(f"\n  5 ganadores, fitness promedio: {avg_fit:.2f}")
    print(f"  Fitness promedio población:    {sum(fits)/len(fits):.2f}")
    print(f"  → El torneo elige individuos mejores que la media")

    # ------------------------------------------------------------------
    # 5.2  Crossover de subárboles
    # ------------------------------------------------------------------
    print("\n--- Crossover de subárboles ---")
    from expression_tree import OperatorNode, FunctionNode, Variable, Constant

    pa = ExpressionTree(
        OperatorNode("mul", Variable(0), FunctionNode("sin", Variable(0)))
    )
    pb = ExpressionTree(
        OperatorNode("add", FunctionNode("cos", Variable(0)), Constant(2.0))
    )

    print(f"  Padre A : {pa.to_string()}")
    print(f"  Padre B : {pb.to_string()}")

    for i in range(4):
        random.seed(i * 7)
        child_a, child_b = subtree_crossover(pa, pb, max_depth=6)
        print(f"  Cruce {i+1}: hijo_a={child_a.to_string()[:40]}  |  hijo_b={child_b.to_string()[:40]}")

    # ------------------------------------------------------------------
    # 5.3  Mutaciones
    # ------------------------------------------------------------------
    print("\n--- Mutaciones sobre: (x0 * sin(x0)) ---")
    base = ExpressionTree(
        OperatorNode("mul", Variable(0), FunctionNode("sin", Variable(0)))
    )

    for seed in range(5):
        random.seed(seed)
        m_sub  = subtree_mutation(base, gen)
        m_pt   = point_mutation(base, gen)
        m_cst  = constant_mutation(
            ExpressionTree(OperatorNode("add", Variable(0), Constant(1.5))), gen
        )
        print(f"  subtree [{seed}]: {m_sub.to_string()[:50]}")

    print()
    for seed in range(3):
        random.seed(seed)
        m_pt = point_mutation(base, gen)
        print(f"  point   [{seed}]: {m_pt.to_string()[:50]}")

    print()
    cst_tree = ExpressionTree(OperatorNode("add", Variable(0), Constant(1.5)))
    for seed in range(3):
        random.seed(seed)
        m_cst = constant_mutation(cst_tree, gen, sigma=0.5)
        print(f"  constant[{seed}]: {m_cst.to_string()[:50]}")

    # ------------------------------------------------------------------
    # 5.4  Evolución de una generación completa
    # ------------------------------------------------------------------
    print("\n--- Evolución de una generación completa ---")
    X      = np.linspace(-3, 3, 40).reshape(-1, 1)
    y_true = X[:, 0] ** 2 + np.sin(X[:, 0])

    evaluator = FitnessEvaluator(metric="mse", parsimony_coeff=0.001)
    operators = GeneticOperators(
        generator=gen,
        tournament_size=3,
        p_crossover=0.8,
        p_subtree_mut=0.1,
        p_point_mut=0.05,
        p_constant_mut=0.05,
        elite_size=1,
    )

    random.seed(0)
    np.random.seed(0)
    population = [gen.ramped_half_and_half(1, 4) for _ in range(12)]
    fitnesses  = evaluate_population(population, X, y_true, evaluator)

    best_before = min(fitnesses)
    print(f"  Generación 0:  mejor fitness = {best_before:.4f}")

    # Simular 5 generaciones
    for gen_n in range(1, 6):
        population = operators.evolve(population, fitnesses)
        fitnesses  = evaluate_population(population, X, y_true, evaluator)
        best_now   = min(fitnesses)
        best_expr  = population[int(np.argmin(fitnesses))].to_string()
        print(f"  Generación {gen_n}:  mejor fitness = {best_now:.4f}  →  {best_expr[:55]}")

