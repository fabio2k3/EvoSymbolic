"""
Fase 1 — Representación de Expresiones
=======================================
Biblioteca de Regresión Simbólica (aprendizaje)

Conceptos implementados:
  - Nodo base (Node)
  - Nodos terminales: Variable y Constant
  - Nodos función: FunctionNode (unaria) y OperatorNode (binaria)
  - Árbol de expresión: ExpressionTree
    · evaluate(X)
    · to_string() / __repr__
    · depth() / size()
    · generación aleatoria (ramped half-and-half)
    · copia profunda (clone)
    · listar todos los nodos con su índice (para crossover/mutación)
"""

import numpy as np
import random
import copy
from typing import Optional, Union

# ---------------------------------------------------------------------------
# 1. CONJUNTOS DE OPERACIONES
# ---------------------------------------------------------------------------
# Operadores binarios: (nombre, función, símbolo)
BINARY_OPERATORS = {
    "add": (np.add,       "+"),
    "sub": (np.subtract,  "-"),
    "mul": (np.multiply,  "*"),
    "div": ("protected",  "/"),   # división protegida (ver abajo)
}

# Funciones unarias: (nombre, función)
UNARY_FUNCTIONS = {
    "sin":  np.sin,
    "cos":  np.cos,
    "exp":  ("protected", np.exp),   # exp protegido contra overflow
    "log":  ("protected", np.log),   # log protegido contra dominio
    "abs":  np.abs,
    "neg":  np.negative,
    "sqrt": ("protected", np.sqrt),  # sqrt protegido
}


def protected_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """División protegida: si |b| < 1e-10 devuelve 1.0."""
    return np.where(np.abs(b) < 1e-10, 1.0, a / b)


def protected_exp(x: np.ndarray) -> np.ndarray:
    """Exponencial protegida: clampea el argumento para evitar overflow."""
    return np.exp(np.clip(x, -100, 100))


def protected_log(x: np.ndarray) -> np.ndarray:
    """Logaritmo protegido: toma el log del valor absoluto (evita dominio)."""
    return np.log(np.abs(x) + 1e-10)


def protected_sqrt(x: np.ndarray) -> np.ndarray:
    """Raíz cuadrada protegida: usa valor absoluto."""
    return np.sqrt(np.abs(x))


# ---------------------------------------------------------------------------
# 2. NODOS
# ---------------------------------------------------------------------------

class Node:
    """Clase base abstracta para todos los nodos del árbol."""

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evalúa el nodo sobre la matriz de datos X.
        X tiene forma (n_samples, n_features).
        Devuelve un array de forma (n_samples,).
        """
        raise NotImplementedError

    def to_string(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.to_string()

    def is_terminal(self) -> bool:
        """Devuelve True si el nodo es una hoja (no tiene hijos)."""
        raise NotImplementedError


class Variable(Node):
    """
    Nodo terminal: representa una variable de entrada (una columna de X).

    Ejemplo: Variable(0) accede a X[:, 0]  →  "x0"
    """

    def __init__(self, index: int, name: Optional[str] = None):
        self.index = index
        self.name = name if name else f"x{index}"

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.index]

    def to_string(self) -> str:
        return self.name

    def is_terminal(self) -> bool:
        return True


class Constant(Node):
    """
    Nodo terminal: representa una constante numérica.

    Ejemplo: Constant(3.14) → "3.14"
    """

    def __init__(self, value: float):
        self.value = float(value)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        # Devuelve un array del mismo tamaño que el lote
        return np.full(X.shape[0], self.value)

    def to_string(self) -> str:
        # Muestra hasta 4 decimales, eliminando ceros innecesarios
        return f"{self.value:.4g}"

    def is_terminal(self) -> bool:
        return True


class FunctionNode(Node):
    """
    Nodo función UNARIA: aplica una función matemática a un único hijo.

    Ejemplo: FunctionNode("sin", child) → "sin(x0)"
    """

    def __init__(self, name: str, child: Node):
        self.name = name
        self.child = child

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        child_val = self.child.evaluate(X)

        # Despacha a la función correcta (normal o protegida)
        entry = UNARY_FUNCTIONS[self.name]
        if isinstance(entry, tuple) and entry[0] == "protected":
            fn_name = self.name
            if fn_name == "exp":
                return protected_exp(child_val)
            elif fn_name == "log":
                return protected_log(child_val)
            elif fn_name == "sqrt":
                return protected_sqrt(child_val)
        else:
            return entry(child_val)

    def to_string(self) -> str:
        return f"{self.name}({self.child.to_string()})"

    def is_terminal(self) -> bool:
        return False

    def arity(self) -> int:
        return 1


class OperatorNode(Node):
    """
    Nodo operador BINARIO: combina dos hijos con una operación (+, -, *, /).

    Ejemplo: OperatorNode("add", left, right) → "(x0 + 3.14)"
    """

    def __init__(self, name: str, left: Node, right: Node):
        self.name = name
        self.left = left
        self.right = right
        _, self.symbol = BINARY_OPERATORS[self.name]

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        left_val  = self.left.evaluate(X)
        right_val = self.right.evaluate(X)

        entry, _ = BINARY_OPERATORS[self.name]
        if entry == "protected":
            return protected_div(left_val, right_val)
        else:
            return entry(left_val, right_val)

    def to_string(self) -> str:
        return f"({self.left.to_string()} {self.symbol} {self.right.to_string()})"

    def is_terminal(self) -> bool:
        return False

    def arity(self) -> int:
        return 2


# ---------------------------------------------------------------------------
# 3. ÁRBOL DE EXPRESIÓN
# ---------------------------------------------------------------------------

class ExpressionTree:
    """
    Árbol de expresión matemática.

    Atributos
    ---------
    root : Node
        Nodo raíz del árbol.

    Métodos principales
    -------------------
    evaluate(X)        → evalúa la expresión sobre datos
    to_string()        → representación legible
    depth()            → profundidad máxima
    size()             → número total de nodos
    clone()            → copia profunda independiente
    get_all_nodes()    → lista [(índice, nodo, padre, posición)]
                         útil para crossover y mutación en Fases 3/4
    """

    def __init__(self, root: Node):
        self.root = root

    # ------------------------------------------------------------------
    # Evaluación
    # ------------------------------------------------------------------
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evalúa el árbol sobre X (n_samples × n_features).
        Devuelve un array de forma (n_samples,).
        Si hay NaN o Inf, los reemplaza por un valor grande.
        """
        result = self.root.evaluate(X)
        result = np.where(np.isfinite(result), result, 1e10)
        return result

    # ------------------------------------------------------------------
    # Representación
    # ------------------------------------------------------------------
    def to_string(self) -> str:
        return self.root.to_string()

    def __repr__(self) -> str:
        return f"ExpressionTree({self.to_string()})"

    # ------------------------------------------------------------------
    # Métricas estructurales
    # ------------------------------------------------------------------
    def depth(self) -> int:
        """Profundidad máxima del árbol (hoja = 0)."""
        return self._depth(self.root)

    def _depth(self, node: Node) -> int:
        if node.is_terminal():
            return 0
        if isinstance(node, FunctionNode):
            return 1 + self._depth(node.child)
        if isinstance(node, OperatorNode):
            return 1 + max(self._depth(node.left), self._depth(node.right))

    def size(self) -> int:
        """Número total de nodos en el árbol."""
        return self._size(self.root)

    def _size(self, node: Node) -> int:
        if node.is_terminal():
            return 1
        if isinstance(node, FunctionNode):
            return 1 + self._size(node.child)
        if isinstance(node, OperatorNode):
            return 1 + self._size(node.left) + self._size(node.right)

    # ------------------------------------------------------------------
    # Clonación
    # ------------------------------------------------------------------
    def clone(self) -> "ExpressionTree":
        """Devuelve una copia profunda e independiente del árbol."""
        return ExpressionTree(copy.deepcopy(self.root))

    # ------------------------------------------------------------------
    # Listado de nodos (para operadores genéticos)
    # ------------------------------------------------------------------
    def get_all_nodes(self) -> list:
        """
        Devuelve una lista de tuplas:
            (nodo, padre, posición)

        donde 'posición' es:
            "root"  → es la raíz
            "child" → hijo de un FunctionNode
            "left"  → hijo izquierdo de OperatorNode
            "right" → hijo derecho de OperatorNode

        Útil en la Fase 3 (crossover, mutación) para elegir
        un subárbol al azar y saber cómo reemplazarlo.
        """
        nodes = []
        self._collect_nodes(self.root, None, "root", nodes)
        return nodes

    def _collect_nodes(self, node, parent, position, nodes):
        nodes.append((node, parent, position))
        if isinstance(node, FunctionNode):
            self._collect_nodes(node.child, node, "child", nodes)
        elif isinstance(node, OperatorNode):
            self._collect_nodes(node.left,  node, "left",  nodes)
            self._collect_nodes(node.right, node, "right", nodes)


# ---------------------------------------------------------------------------
# 4. GENERADOR ALEATORIO DE ÁRBOLES
# ---------------------------------------------------------------------------

class TreeGenerator:
    """
    Genera árboles de expresión aleatoriamente.

    Parámetros
    ----------
    n_features : int
        Número de variables de entrada (columnas de X).
    operators  : list[str]
        Operadores binarios a usar (por defecto todos).
    functions  : list[str]
        Funciones unarias a usar (por defecto todas).
    const_range : tuple
        Rango [min, max] para generar constantes aleatorias.
    p_terminal : float
        Probabilidad base de generar un terminal (constante o variable).
    p_variable : float
        Probabilidad de que un terminal sea variable (vs constante).
    p_unary    : float
        Probabilidad de que un nodo función sea unario (vs binario).
    """

    def __init__(
        self,
        n_features:  int   = 1,
        operators:   list  = None,
        functions:   list  = None,
        const_range: tuple = (-5.0, 5.0),
        p_terminal:  float = 0.3,
        p_variable:  float = 0.6,
        p_unary:     float = 0.3,
    ):
        self.n_features  = n_features
        self.operators   = operators  or list(BINARY_OPERATORS.keys())
        self.functions   = functions  or list(UNARY_FUNCTIONS.keys())
        self.const_range = const_range
        self.p_terminal  = p_terminal
        self.p_variable  = p_variable
        self.p_unary     = p_unary

    # ------------------------------------------------------------------
    # Método principal: ramped half-and-half
    # ------------------------------------------------------------------
    def ramped_half_and_half(
        self,
        min_depth: int = 1,
        max_depth: int = 4,
    ) -> ExpressionTree:
        """
        Genera un árbol usando ramped half-and-half:
          - 50% de las veces usa método 'full'   (todas las ramas hasta max_depth)
          - 50% de las veces usa método 'grow'   (ramas de longitud variable)

        La profundidad se elige uniformemente entre min_depth y max_depth.
        Esto produce una población inicial con diversidad estructural.
        """
        depth  = random.randint(min_depth, max_depth)
        method = random.choice(["full", "grow"])
        root   = self._generate_node(depth, method)
        return ExpressionTree(root)

    # ------------------------------------------------------------------
    # Métodos full y grow
    # ------------------------------------------------------------------
    def _generate_node(self, max_depth: int, method: str) -> Node:
        """
        Genera un nodo recursivamente.

        method='full': genera nodos internos hasta max_depth,
                       luego terminales obligatoriamente.
        method='grow': puede generar terminales antes de max_depth.
        """
        # Si llegamos al límite, generamos un terminal sí o sí
        if max_depth == 0:
            return self._random_terminal()

        # En 'full' siempre generamos un nodo interno
        # En 'grow' decidimos aleatoriamente
        if method == "full":
            force_internal = True
        else:
            force_internal = (random.random() > self.p_terminal)

        if force_internal:
            return self._random_internal(max_depth, method)
        else:
            return self._random_terminal()

    def _random_terminal(self) -> Node:
        """Genera un terminal: Variable o Constant."""
        if random.random() < self.p_variable:
            idx = random.randint(0, self.n_features - 1)
            return Variable(idx)
        else:
            value = random.uniform(*self.const_range)
            return Constant(value)

    def _random_internal(self, max_depth: int, method: str) -> Node:
        """Genera un nodo interno: FunctionNode o OperatorNode."""
        if random.random() < self.p_unary and self.functions:
            # Nodo unario
            fn_name = random.choice(self.functions)
            child   = self._generate_node(max_depth - 1, method)
            return FunctionNode(fn_name, child)
        else:
            # Nodo binario
            op_name = random.choice(self.operators)
            left    = self._generate_node(max_depth - 1, method)
            right   = self._generate_node(max_depth - 1, method)
            return OperatorNode(op_name, left, right)


# ---------------------------------------------------------------------------
# 5. DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  FASE 1 — Representación de Expresiones")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 5.1  Construcción manual de (x0 + 2) * sin(x0)
    # ------------------------------------------------------------------
    print("\n--- Árbol manual: (x0 + 2) * sin(x0) ---")

    tree_manual = ExpressionTree(
        OperatorNode("mul",
            OperatorNode("add",
                Variable(0),
                Constant(2.0)
            ),
            FunctionNode("sin",
                Variable(0)
            )
        )
    )

    print(f"  Expresión : {tree_manual.to_string()}")
    print(f"  Profundidad: {tree_manual.depth()}")
    print(f"  Tamaño    : {tree_manual.size()} nodos")

    # Evaluación sobre datos reales
    X = np.linspace(-3, 3, 6).reshape(-1, 1)          # 6 puntos, 1 variable
    y_pred = tree_manual.evaluate(X)
    y_real  = (X[:, 0] + 2) * np.sin(X[:, 0])

    print(f"\n  X         : {X[:, 0].round(2)}")
    print(f"  Predicho  : {y_pred.round(4)}")
    print(f"  Real      : {y_real.round(4)}")
    print(f"  ¿Correcto?: {np.allclose(y_pred, y_real)}")

    # ------------------------------------------------------------------
    # 5.2  Listado de nodos (para futuros operadores genéticos)
    # ------------------------------------------------------------------
    print("\n--- Nodos del árbol (útil en Fase 3) ---")
    for i, (node, parent, pos) in enumerate(tree_manual.get_all_nodes()):
        parent_str = parent.to_string() if parent else "—"
        print(f"  [{i}] {node.to_string():<20} | padre: {parent_str:<25} | pos: {pos}")

    # ------------------------------------------------------------------
    # 5.3  Clonación
    # ------------------------------------------------------------------
    clone = tree_manual.clone()
    clone.root.left.left = Constant(99.0)          # Modificar el clon
    print(f"\n--- Clonación ---")
    print(f"  Original : {tree_manual.to_string()}")
    print(f"  Clon mod.: {clone.to_string()}")
    print(f"  Son independientes: {tree_manual.to_string() != clone.to_string()}")

    # ------------------------------------------------------------------
    # 5.4  Generación aleatoria
    # ------------------------------------------------------------------
    print("\n--- Generación aleatoria (ramped half-and-half) ---")
    gen = TreeGenerator(n_features=2, const_range=(-3.0, 3.0))
    random.seed(42)

    for i in range(6):
        t = gen.ramped_half_and_half(min_depth=1, max_depth=3)
        print(f"  [{i+1}] depth={t.depth()}  size={t.size():2d}  →  {t.to_string()}")

    # ------------------------------------------------------------------
    # 5.5  Funciones protegidas
    # ------------------------------------------------------------------
    print("\n--- Funciones protegidas ---")
    X_danger = np.array([[0.0], [-1.0], [1e-15], [100.0]])

    div_tree = ExpressionTree(
        OperatorNode("div", Variable(0), Constant(0.0))
    )
    log_tree = ExpressionTree(FunctionNode("log", Variable(0)))
    exp_tree = ExpressionTree(FunctionNode("exp", Constant(999.0)))

    print(f"  x / 0     → {div_tree.evaluate(X_danger[:1])}")   # debe dar 1.0
    print(f"  log(-1)   → {log_tree.evaluate(X_danger[1:2])}")   # no debe dar NaN
    print(f"  exp(999)  → {exp_tree.evaluate(X_danger[:1])}")    # no debe dar Inf

    print("\n✓ Fase 1 completada.")
