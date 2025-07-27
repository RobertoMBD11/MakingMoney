import numpy as np
import itertools
import random
from typing import List, Dict, Callable
import ast

import numpy as np
from typing import List, Dict

class Individual:
    def __init__(self, param_bins: Dict[str, List[float]], genes_array: np.ndarray = None):
        """
        param_bins: dict de parámetros y sus bins (necesario para mapear codes a índices)
        genes_array: numpy array con los genes (acciones)
        """
        self.param_bins = param_bins
        self.num_genes = self._calc_num_genes()
        if genes_array is None:
            self.genes = np.zeros(self.num_genes, dtype=np.int8)  # inicializa con ceros
        else:
            assert len(genes_array) == self.num_genes, "genes_array no tiene la longitud correcta"
            self.genes = genes_array
        self.fitness = None

    def _calc_num_genes(self) -> int:
        bases = [len(self.param_bins[p]) + 1 for p in self.param_bins]
        total_codes = 1
        for b in bases:
            total_codes *= b
        return total_codes * 2  # por el estado (0 o 1)

    def code_to_index(self, code: str) -> int:
        try:
            parts = list(map(int, code.split('_')))
        except ValueError:
            raise ValueError(f"Código inválido recibido en code_to_index: '{code}'")
        state = parts[-1]
        param_indices = parts[:-1]

        num_params = len(self.param_bins)
        assert len(param_indices) == num_params, "Código incompatible con número de parámetros"

        bases = [len(self.param_bins[p]) + 1 for p in self.param_bins]

        index = 0
        for i in range(num_params):
            mult = 1
            for b in bases[i+1:]:
                mult *= b
            index += param_indices[i] * mult

        index = index * 2 + state
        return index

    def get(self, code: str, default=0) -> int:
        idx = self.code_to_index(code)
        if 0 <= idx < self.num_genes:
            return int(self.genes[idx])
        return default

    def set(self, code: str, action: int):
        idx = self.code_to_index(code)
        if 0 <= idx < self.num_genes:
            self.genes[idx] = action

    def set_fitness(self, fitness_value: float):
        self.fitness = fitness_value

    def __getitem__(self, code: str) -> int:
        return self.get(code)

    def __setitem__(self, code: str, action: int):
        self.set(code, action)

    def generate_all_codes(self) -> List[str]:
        from itertools import product

        param_names = list(self.param_bins.keys())
        bin_counts = [len(self.param_bins[p]) + 1 for p in param_names]  # +1 por np.digitize
        all_combinations = product(*[range(n) for n in bin_counts], [0, 1])
        codes = ["_".join(map(str, comb)) for comb in all_combinations]
        return codes

    def generate_dict(self) -> Dict[str, int]:
        codes = self.generate_all_codes()
        return {code: int(self.genes[idx]) for idx, code in enumerate(codes)}

    def __repr__(self):
        return f"Individual(num_genes={len(self.genes)}, fitness={self.fitness}, genes_dict={self.generate_dict()})"


class Population:
    def __init__(self, param_bins: Dict[str, List[float]], num_individuals: int):
        self.param_bins = param_bins
        self.all_codes = self.generate_all_codes()
        self.num_individuals = num_individuals
        self.code_to_index = {code: idx for idx, code in enumerate(self.all_codes)}
        self.num_genes = len(self.all_codes)
        self.individuals = self._generate_random_individuals()
        

    def __len__(self):
        return self.num_individuals

    def __getitem__(self, idx):
        return self.individuals[idx]

    def generate_all_codes(self) -> List[str]:
        """
        Genera todos los códigos posibles combinando los índices discretizados de cada parámetro
        y los dos posibles estados (0 o 1).
        """
        from itertools import product

        param_names = list(self.param_bins.keys())
        bin_counts = [len(self.param_bins[p]) + 1 for p in param_names]  # +1 porque np.digitize
        all_combinations = product(*[range(n) for n in bin_counts], [0, 1])
        codes = ["_".join(map(str, comb)) for comb in all_combinations]
        return codes

    def _generate_random_individuals(self):
        individuals = []

        for _ in range(self.num_individuals):
            genes_array = np.zeros(self.num_genes, dtype=np.int8)

            for i, code in enumerate(self.all_codes):
                estado = int(code.split('_')[-1])
                if estado == 0:
                    genes_array[i] = np.random.randint(0, 2)  # 0 o 1
                else:  # estado == 1
                    genes_array[i] = np.random.randint(0, 2)  # 2 o 3  era para comprobar que está bien

            ind = Individual(param_bins=self.param_bins, genes_array=genes_array)
            individuals.append(ind)

        return individuals

    def describe(self):
        print(f"Population:")
        print(f" - Num individuals: {self.num_individuals}")
        print(f" - Num genes per individual: {self.num_genes}")
        print(f" - Param bins: {self.param_bins}")

    def get_chromosomes(self) -> List[np.ndarray]:
        """
        Devuelve una lista con los arrays de genes (cromosomas) de cada individuo.
        """
        return [ind.genes.copy() for ind in self.individuals]

def discretize(value, bins):
    """Asigna un valor al índice de intervalo según los bins."""
    return np.digitize(value, bins, right=False)

def encode_state_action(indices, state):
    """Codifica un conjunto de índices + estado en un string único."""
    return "_".join(map(str, indices)) + f"_{state}"

def get_code_from_candle(candle, state, param_bins):
    """
    candle: dict con valores de cada parámetro (ej: {'rsi':45, 'sma':1.02, ...})
    state: 0 (no invertido) o 1 (invertido)
    param_bins: dict con los bins de cada parámetro
    Función que devuelve el codigo donde mirar dentro del cromosoma en función de la vela dada
    """
    indices = []

    for param in param_bins:
        value = candle[param]
        bins = param_bins[param]
        index = discretize(value, bins)
        indices.append(index)

    code = encode_state_action(indices, state)
    return code

def code_to_index(code: str, param_bins: Dict[str, List[float]]) -> int:
    try:
        parts = list(map(int, code.split('_')))
    except ValueError:
        raise ValueError(f"Código inválido recibido en code_to_index: '{code}'")

    state = parts[-1]
    param_indices = parts[:-1]

    # Número de parámetros
    num_params = len(param_bins)
    assert len(param_indices) == num_params, "El código no coincide con el número de parámetros"

    # Bases (número de clases por parámetro)
    bases = [len(param_bins[p]) + 1 for p in param_bins]

    # Calcular índice "posicional"
    index = 0
    for i in range(num_params):
        mult = 1
        for b in bases[i+1:]:
            mult *= b
        index += param_indices[i] * mult

    # Como hay 2 estados, cada conjunto de parámetros tiene dos índices consecutivos
    index = index * 2 + state

    return index

def get_semicode_from_candle(candle, param_bins):
    """
    Genera un "semicode" basado solo en los indicadores, sin usar el estado.
    """
    indices = []

    for param in param_bins:
        value = candle[param]
        bins = param_bins[param]
        index = discretize(value, bins)
        indices.append(index)

    return tuple(indices)  # usamos tupla para poder usar como clave o para codificar

def get_code_from_semicode_and_state(semicode, state):
    """
    Convierte semicode (en tupla o string tipo "(2, 2, 0)") + estado en código único: "2_2_0_1"
    """
    # Si es string, intentar convertirlo en tupla de forma segura
    if isinstance(semicode, str):
        try:
            semicode = ast.literal_eval(semicode)  # Seguro para strings como "(2, 1, 0)"
        except Exception as e:
            raise ValueError(f"Semicode string malformado: {semicode}") from e

    # Verificar que ahora sea tupla o lista
    if not isinstance(semicode, (list, tuple)):
        raise ValueError(f"Semicode debe ser lista o tupla, recibido: {type(semicode)} -> {semicode}")

    # Formatear a string tipo "2_1_0_1"
    code = "_".join(map(str, semicode)) + f"_{state}"
    return code



if __name__ == "__main__":
    acciones = {0: "NO HACER NADA", 1: "ENTRAR", 2: "MANTENER", 3: "SALIR"}

    # Define los bins para discretización
    param_bins = {
        'rsi': [20, 40, 60, 80],
        'sma': [0.95, 1.0, 1.05, 1.1],
        'atr': [0.005, 0.01, 0.02, 0.05]
    }

    # 1. Crear población
    population = Population(param_bins=param_bins, num_individuals=5)

    print("Número total de genes:", population.num_genes)
    print("Número de individuos:", len(population))

    # 2. Simulación de ejemplo: una vela
    candle = {
        "rsi": 45,
        "sma": 1.02,
        "atr": 0.008
    }
    state = 0

    # 3. Obtener código de esa vela
    code = get_code_from_candle(candle, state, param_bins)
    code = '0_1_0_1'
    print(f"Code: {code}")
    # 4. Escoger un individuo random y ver qué acción toma
    individuo_random = population[random.randint(0, len(population)-1)]
    action = individuo_random.get(code, 0)
    print("Acción:",action)
    print(individuo_random)