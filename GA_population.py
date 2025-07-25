import numpy as np
import itertools
import random
from typing import List, Dict, Callable

class Individual:
    def __init__(self, genes_dict=None):
        # genes_dict: dict code -> acción (int)
        self.genes = genes_dict if genes_dict else {}
        self.fitness = None

    def get(self, code, default=0):
        return self.genes.get(code, default)

    def set(self, code, action):
        self.genes[code] = action

    def set_fitness(self, fitness_value: float):
        self.fitness = fitness_value

    def __getitem__(self, code):
        # para poder hacer individuo[code]
        return self.genes.get(code, 0)

    def __setitem__(self, code, action):
        self.genes[code] = action

    def __repr__(self):
        return f"Individual(genes={len(self.genes)} genes, fitness={self.fitness})"



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
        mid = self.num_genes // 2
        for _ in range(self.num_individuals):
            # Creamos un array de acciones como antes
            genes_array = np.zeros(self.num_genes, dtype=np.int8)
            genes_array[:mid] = np.random.randint(0, 2, size=mid)     # estado 0 (acciones 0 o 1)
            genes_array[mid:] = np.random.randint(2, 4, size=self.num_genes - mid)  # estado 1 (acciones 2 o 3)

            # Convertimos genes_array a diccionario code->acción usando index_to_code
            genes_dict = {self.all_codes[i]: int(genes_array[i]) for i in range(self.num_genes)}

            # Creamos el individuo con el dict de genes
            ind = Individual(genes_dict=genes_dict)
            individuals.append(ind)
        return individuals

    def describe(self):
        print(f"Population:")
        print(f" - Num individuals: {self.num_individuals}")
        print(f" - Num genes per individual: {self.num_genes}")
        print(f" - Param bins: {self.param_bins}")

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
    parts = list(map(int, code.split('_')))
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
    Codifica un semicode + estado en un string único, con el estado al final.
    Ejemplo: "2_1_0_1" donde (2,1,0) es el semicode y 1 es el estado.
    """
    return "_".join(map(str, semicode)) + f"_{state}"

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
    print(f"Code: {code}")
    # 4. Escoger un individuo random y ver qué acción toma
    individuo_random = population[random.randint(0, len(population)-1)]
    
    idx = code_to_index(code,param_bins)
    print(population.all_codes[idx])
    action = individuo_random.get(code, 0)

    print(f"Código generado: {code}")
    print("Acción a tomar:", acciones.get(action, "CÓDIGO DESCONOCIDO"))