"""
Operadores Genéticos
====================

Módulo encargado de aplicar los operadores evolutivos sobre la población
de árboles de expresión.

Incluye:
  - Selección por torneo
  - Crossover de subárboles
  - Mutaciones:
      · subtree_mutation
      · point_mutation
      · constant_mutation

Además, se define una clase unificadora para gestionar la evolución
completa de una generación.
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

# Profundidad máxima permitida tras aplicar operadores genéticos.
MAX_DEPTH_DEFAULT = 6

# ---------------------------------------------------------------------------
# UTILIDAD: reemplazar un nodo dentro de un árbol
# ---------------------------------------------------------------------------

def _replace_node(tree: ExpressionTree, target: Node, new_node: Node) -> None:
    """
    Reemplaza un nodo dentro del árbol, modificando la estructura in-place.

    Parámetros
    ----------
    tree : ExpressionTree
        Árbol que será modificado.
    target : Node
        Nodo actual que se desea reemplazar.
    new_node : Node
        Nodo que ocupará el lugar del nodo objetivo.

    Notas
    -----
    Se apoya en get_all_nodes(), que devuelve:
        (nodo, padre, posición)
    donde posición puede ser "root", "left", "right" o "child".
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

    Algoritmo
    ---------
    1. Elige k individuos al azar de la población, con reemplazo.
    2. Devuelve el individuo con menor fitness, recordando que aquí
       se minimiza la función de aptitud.

    Ventajas
    --------
    - No exige fitness positivos.
    - La presión selectiva se controla fácilmente con k.
      k pequeño  -> mayor exploración
      k grande   -> mayor explotación

    Retorna
    -------
    ExpressionTree
        Referencia al individuo ganador. El motor genético suele clonar
        antes de aplicar operadores.
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
    Crossover estándar de subárboles.

    Algoritmo
    ---------
    1. Clona ambos padres.
    2. Selecciona un nodo aleatorio en cada hijo.
    3. Intercambia los subárboles de esos nodos.
    4. Si algún hijo supera max_depth, se usa el padre original como fallback.

    El sesgo hacia nodos internos favorece cruces más informativos.
    """
    child_a = parent_a.clone()
    child_b = parent_b.clone()

    # Lista de nodos disponibles en cada hijo.
    nodes_a = child_a.get_all_nodes()
    nodes_b = child_b.get_all_nodes()

    # Elegimos puntos de corte con sesgo hacia nodos internos.
    point_a = _biased_node_pick(nodes_a)
    point_b = _biased_node_pick(nodes_b)

    # Copias profundas de los subárboles que se intercambiarán.
    subtree_a = copy.deepcopy(point_a[0])
    subtree_b = copy.deepcopy(point_b[0])

    # Intercambio de subárboles.
    _replace_node(child_a, point_a[0], subtree_b)
    _replace_node(child_b, point_b[0], subtree_a)

    # Fallback si algún hijo se vuelve demasiado profundo.
    if child_a.depth() > max_depth:
        child_a = parent_a.clone()
    if child_b.depth() > max_depth:
        child_b = parent_b.clone()

    return child_a, child_b


def _biased_node_pick(nodes: list, p_internal: float = 0.9):
    """
    Selecciona un nodo con sesgo hacia nodos internos.

    Parámetros
    ----------
    nodes : list
        Lista de tuplas (nodo, padre, posición).
    p_internal : float
        Probabilidad de seleccionar un nodo interno.

    Retorna
    -------
    tuple
        Tupla (nodo, padre, posición).
    """
    internals = [(n, p, pos) for (n, p, pos) in nodes if not n.is_terminal()]
    terminals = [(n, p, pos) for (n, p, pos) in nodes if n.is_terminal()]

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

    Algoritmo
    ---------
    1. Clona el árbol original.
    2. Selecciona un nodo al azar, con sesgo hacia nodos internos.
    3. Genera un subárbol aleatorio pequeño.
    4. Sustituye el nodo elegido por el nuevo subárbol.
    5. Si el árbol resultante supera max_depth, devuelve el original.

    Es una mutación bastante disruptiva porque puede cambiar la estructura
    del árbol de forma importante.
    """
    mutant = tree.clone()
    nodes = mutant.get_all_nodes()
    target = _biased_node_pick(nodes)

    new_subtree = generator.ramped_half_and_half(
        min_depth=0, max_depth=2
    ).root

    _replace_node(mutant, target[0], new_subtree)

    if mutant.depth() > max_depth:
        return tree.clone()

    return mutant


def point_mutation(
    tree:      ExpressionTree,
    generator: TreeGenerator,
) -> ExpressionTree:
    """
    Mutación de punto.

    Reemplaza un único nodo por otro del mismo tipo y aridad:
      - OperatorNode  -> otro operador binario
      - FunctionNode  -> otra función unaria
      - Constant      -> nueva constante aleatoria
      - Variable      -> otra variable, si existe más de una

    Esta mutación conserva la estructura general del árbol.
    """
    mutant = tree.clone()
    all_nodes = mutant.get_all_nodes()

    # Se elige cualquier nodo del árbol.
    target_node, parent, position = random.choice(all_nodes)

    new_node = _mutate_node(target_node, generator)
    _replace_node(mutant, target_node, new_node)

    return mutant


def _mutate_node(node: Node, generator: TreeGenerator) -> Node:
    """Genera un nodo nuevo del mismo tipo que el nodo original."""
    if isinstance(node, OperatorNode):
        ops = [k for k in generator.operators if k != node.name]
        if not ops:
            return copy.deepcopy(node)
        new_op = random.choice(ops)
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
    Mutación de constante.

    Perturba una constante existente con ruido gaussiano.
    Si el árbol no contiene constantes, devuelve una copia sin cambios.

    Parámetros
    ----------
    sigma : float
        Desviación estándar del ruido gaussiano.
        Valores pequeños favorecen ajustes finos.
        Valores grandes introducen cambios más bruscos.
    """
    mutant = tree.clone()
    all_nodes = mutant.get_all_nodes()
    constants = [(n, p, pos) for (n, p, pos) in all_nodes if isinstance(n, Constant)]

    if not constants:
        return mutant

    target_node, parent, position = random.choice(constants)
    new_val = target_node.value + random.gauss(0, sigma)
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
    generator       : TreeGenerator
        Generador de árboles aleatorios usado en mutaciones.
    tournament_size : int
        Tamaño del torneo para selección.
    p_crossover     : float
        Probabilidad de aplicar crossover.
    p_subtree_mut   : float
        Probabilidad de mutación de subárbol.
    p_point_mut     : float
        Probabilidad de mutación de punto.
    p_constant_mut  : float
        Probabilidad de mutación de constante.
    max_depth       : int
        Profundidad máxima permitida tras los operadores.
    elite_size      : int
        Número de mejores individuos que pasan directamente a la nueva generación.

    Nota
    ----
    Las mutaciones se aplican de forma independiente.
    Un individuo puede sufrir más de una mutación en el mismo ciclo.
    """

    def __init__(
        self,
        generator:      TreeGenerator,
        tournament_size: int   = 3,
        p_crossover:    float = 0.80,
        p_subtree_mut:   float = 0.10,
        p_point_mut:     float = 0.05,
        p_constant_mut:  float = 0.05,
        max_depth:      int   = MAX_DEPTH_DEFAULT,
        elite_size:     int   = 1,
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
        Genera una nueva población del mismo tamaño que la original.

        Pasos
        -----
        1. Copia los mejores individuos mediante elitismo.
        2. Selecciona padres por torneo.
        3. Aplica crossover con cierta probabilidad.
        4. Aplica mutaciones de forma independiente.
        """
        pop_size = len(population)
        new_pop = []

        # Elitismo: se preservan los mejores individuos.
        elite = self._get_elite(population, fitnesses)
        new_pop.extend(elite)

        # Generación del resto de la población.
        while len(new_pop) < pop_size:
            # Selección del primer padre.
            parent_a = tournament_selection(population, fitnesses, self.tournament_size)

            # Cruce o reproducción directa.
            if random.random() < self.p_crossover:
                parent_b = tournament_selection(population, fitnesses, self.tournament_size)
                child, _ = subtree_crossover(parent_a, parent_b, self.max_depth)
            else:
                child = parent_a.clone()

            # Aplicación de mutaciones independientes.
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
        """Devuelve copias profundas de los `elite_size` mejores individuos."""
        sorted_idx = np.argsort(fitnesses)[:self.elite_size]
        return [population[i].clone() for i in sorted_idx]


# ---------------------------------------------------------------------------
# 5. DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Operadores Genéticos")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    gen = TreeGenerator(n_features=1, const_range=(-3.0, 3.0))

    # ------------------------------------------------------------------
    # 5.1  Selección por torneo
    # ------------------------------------------------------------------
    print("\n--- Selección por torneo (k=3) ---")
    pop = [gen.ramped_half_and_half(1, 3) for _ in range(8)]
    fits = [round(random.uniform(0.5, 50.0), 2) for _ in range(8)]

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
        m_sub = subtree_mutation(base, gen)
        m_pt = point_mutation(base, gen)
        m_cst = constant_mutation(
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
    X = np.linspace(-3, 3, 40).reshape(-1, 1)
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
    fitnesses = evaluate_population(population, X, y_true, evaluator)

    best_before = min(fitnesses)
    print(f"  Generación 0:  mejor fitness = {best_before:.4f}")

    # Simulación de varias generaciones.
    for gen_n in range(1, 6):
        population = operators.evolve(population, fitnesses)
        fitnesses = evaluate_population(population, X, y_true, evaluator)
        best_now = min(fitnesses)
        best_expr = population[int(np.argmin(fitnesses))].to_string()
        print(f"  Generación {gen_n}:  mejor fitness = {best_now:.4f}  →  {best_expr[:55]}")