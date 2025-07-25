from day_visor import compute_RSI, compute_SMA_normalized, compute_ATR_normalized
from GA_population import get_code_from_candle, get_semicode_from_candle, get_code_from_semicode_and_state, Population, Individual
import pandas as pd
import numpy as np
import pygad
import matplotlib.pyplot as plt


def fitness(individuo, df, dinero_invertido):
    estado = 0
    precio_entrada = None
    unidades = 0
    beneficios = []

    for _, row in df.iterrows():
        semicode = row['semicode']  # Se asume que df ya tiene esta columna con el semicode (tupla o string)
        code = get_code_from_semicode_and_state(semicode, estado)  # Código que busca la acción en el individuo
        accion = individuo.get(code, 0)  # Acción por defecto es 0 (NO HACER NADA)

        precio_actual = row['close']

        if estado == 0 and accion == 1:  # ENTRAR
            unidades = dinero_invertido / precio_actual
            precio_entrada = precio_actual
            estado = 1

        elif estado == 1 and accion == 1:  # SALIR
            dinero_final = unidades * precio_actual
            beneficio = dinero_final - dinero_invertido
            beneficios.append(beneficio)
            estado = 0
            unidades = 0
            precio_entrada = None

    return sum(beneficios)


def fitness_func(ga_instance, solution, solution_idx):
    # Acceder a variables externas (df, dinero_invertido) vía ga_instance o globales
    df = ga_instance.df
    dinero_invertido = ga_instance.dinero_invertido
    
    # Aquí necesitas transformar el numpy array `solution` a tu objeto Individual para usar tu código
    # Por ejemplo, creando un individuo temporal (según tu clase Individual)
    individuo = Individual(param_bins=ga_instance.param_bins, genes_array=solution)
    
    # Ahora llama tu función fitness original que usaba individuo, df, dinero_invertido
    fitness_value = fitness(individuo, df, dinero_invertido)
    
    return fitness_value

def on_generation(ga_instance):
    # Obtenemos el fitness de toda la población actual
    current_fitness = ga_instance.last_generation_fitness
    avg_fitness = sum(current_fitness) / len(current_fitness)
    
    # Lo guardamos
    average_fitness_per_generation.append(avg_fitness)

    # (Opcional) Mostrar por consola
    print(f"Generación {ga_instance.generations_completed}: Avg = {avg_fitness:.2f} | Max = {ga_instance.best_solution()[1]:.2f}")


if __name__ == "__main__":
    acciones = {0: "NO HACER NADA", 1: "ENTRAR", 2: "MANTENER", 3: "SALIR"}

    param_bins = {
        'rsi': [20, 40, 60, 80],
        'sma': [0.95, 1.0, 1.05, 1.1],
        'atr': [0.005, 0.01, 0.02, 0.05]
    }

    # 1. Crear población
    population = Population(param_bins=param_bins, num_individuals=100)

    # 2. Cargar datos y calcular indicadores
    df = pd.read_csv("fake_crypto_day.csv", parse_dates=['timestamp'])
    df['rsi'] = compute_RSI(df['close'])
    df['sma'] = compute_SMA_normalized(df['close'], window=14)
    _, df['atr'] = compute_ATR_normalized(df, window=14)
    df.dropna(inplace=True)
    df['semicode'] = df.apply(lambda row: get_semicode_from_candle(row, param_bins), axis=1)

    # Other parameters
    initial_population = population.get_chromosomes()
    dinero_invertido = 1000
    param_bins = population.param_bins 

    average_fitness_per_generation = [] 

    ga_instance = pygad.GA(
        num_generations=10,
        gene_space=[0, 1],
        num_parents_mating=10,
        sol_per_pop=20,
        num_genes=population.num_genes,
        on_generation=on_generation,
        fitness_func=fitness_func,
        gene_type=int,
        mutation_type="random",
        mutation_probability=0.1,
        initial_population=initial_population,  # list of numpy arrays
        )
    
    ga_instance.df = df
    ga_instance.dinero_invertido = dinero_invertido
    ga_instance.param_bins = param_bins

    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Mejor solución: {solution}, fitness: {solution_fitness}")
    

    best_fitness = ga_instance.best_solutions_fitness

    plt.plot(best_fitness, label="Best Fitness")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Evolución del Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()



