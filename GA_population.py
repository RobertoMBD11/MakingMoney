import numpy as np
import itertools
import random

def discretize(value, bins):
    """Asigna un valor al índice de intervalo según los bins."""
    return np.digitize(value, bins, right=False)

def encode_state_action(indices, state):
    """Codifica un conjunto de índices + estado en un string único."""
    return "_".join(map(str, indices)) + f"_{state}"

def generate_all_codes(param_bins):
    """
    param_bins: dict con nombre del parámetro como clave y lista de bins como valor
                Ej: {'rsi': [...], 'sma': [...], 'atr': [...]}
    Devuelve:
        - lista de códigos únicos (uno por gen)
    """
    num_intervals = [len(bins) + 1 for bins in param_bins.values()]  # +1 porque digitize devuelve de 0 a len(bins)
    
    all_param_indices = list(itertools.product(*[range(n) for n in num_intervals]))
    all_codes = []

    for state in [0, 1]:  # 0 = no invertido, 1 = invertido
        for indices in all_param_indices:
            code = encode_state_action(indices, state)
            all_codes.append(code)
    
    return all_codes

def generate_random_individuals(all_codes, num_individuals=10):
    """
    all_codes: lista de códigos generados por generate_all_codes()
    Devuelve:
        - Lista de individuos, donde cada individuo es un diccionario {code: acción}
    """
    individuals = []

    for _ in range(num_individuals):
        genome = {}
        for code in all_codes:
            state = int(code.split("_")[-1])  # último número es el estado (0 o 1)

            if state == 0:
                action = random.choice([0, 1])  # No hacer nada o entrar
            else:
                action = random.choice([2, 3])  # Mantener o salir, podría ser también 0 o 1

            genome[code] = action

        individuals.append(genome)

    return individuals

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

if __name__ == "__main__":
    acciones = {0: "NO HACER NADA", 1: "ENTRAR", 2: "MANTENER", 3: "SALIR"}

    # Define tus bins aquí (puedes cambiar libremente)
    param_bins = {
        'rsi': [20, 40, 60, 80],
        'sma': [0.95, 1.0, 1.05, 1.1],
        'atr': [0.005, 0.01, 0.02, 0.05]
    }

    # 1. Genera todos los códigos posibles
    all_codes = generate_all_codes(param_bins)

    # 2. Número total de genes (uno por código)
    print("Número total de genes:", len(all_codes))

    # 3. Genera individuos aleatorios (por ejemplo 5)
    population = generate_random_individuals(all_codes, num_individuals=5)

    # 4. Mostrar un individuo
    #print("Ejemplo de individuo:")
    #for k, v in list(population[0].items())[:-10]:  # muestra solo los primeros 10 genes
    #    print(k, "->", v)


    # Simulación de ejemplo
    candle = {
        "rsi": 45,
        "sma": 1.02,
        "atr": 0.008
    }
    state = 0

    code = get_code_from_candle(candle, state, param_bins)


    individuo_random = random.choice(population)
    action = individuo_random[code]

    print("Acción a tomar:", acciones.get(action, "CÓDIGO DESCONOCIDO"))