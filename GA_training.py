from day_visor import compute_RSI, compute_SMA_normalized, compute_ATR_normalized
from GA_population import get_code_from_candle, get_semicode_from_candle, get_code_from_semicode_and_state, Population, Individual
import pandas as pd
import numpy as np
import pygad
import matplotlib.pyplot as plt
from GA_model import GA_Trainer
import random


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


"""def fitness_func(ga_instance, solution, solution_idx):
    # Acceder a variables externas (df, dinero_invertido) vía ga_instance o globales
    df = ga_instance.df
    dinero_invertido = ga_instance.dinero_invertido
    
    # Aquí necesitas transformar el numpy array `solution` a tu objeto Individual para usar tu código
    # Por ejemplo, creando un individuo temporal (según tu clase Individual)
    individuo = Individual(param_bins=ga_instance.param_bins, genes_array=solution)
    
    # Ahora llama tu función fitness original que usaba individuo, df, dinero_invertido
    fitness_value = fitness(individuo, df, dinero_invertido)
    
    return fitness_value"""

def on_generation(gen, best_individual, avg_train, avg_test):
    print(f"Gen {gen} | Train fitness: {avg_train:.4f}", end='')
    if avg_test is not None:
        print(f" | Test fitness: {avg_test:.4f}", end='')
    print()


def split_dataset_folder(folder_path, train_ratio=0.8, seed=42):
    random.seed(seed)
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    random.shuffle(all_files)

    split_point = int(len(all_files) * train_ratio)
    train_files = all_files[:split_point]
    test_files = all_files[split_point:]

    return train_files, test_files

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
    df_training = pd.read_csv("fake_crypto_day_1.csv", parse_dates=['timestamp'])
    df_training['rsi'] = compute_RSI(df_training['close'])
    df_training['sma'] = compute_SMA_normalized(df_training['close'], window=14)
    _, df_training['atr'] = compute_ATR_normalized(df_training, window=14)
    df_training.dropna(inplace=True)
    df_training['semicode'] = df_training.apply(lambda row: get_semicode_from_candle(row, param_bins), axis=1)

    df_test = pd.read_csv("fake_crypto_day_2.csv", parse_dates=['timestamp'])
    df_test['rsi'] = compute_RSI(df_test['close'])
    df_test['sma'] = compute_SMA_normalized(df_test['close'], window=14)
    _, df_test['atr'] = compute_ATR_normalized(df_test, window=14)
    df_test.dropna(inplace=True)
    df_test['semicode'] = df_test.apply(lambda row: get_semicode_from_candle(row, param_bins), axis=1)

    # Other parameters
    initial_population = population.get_chromosomes()
    dinero_invertido = 1000
    param_bins = population.param_bins 

    average_fitness_per_generation = [] 



    trainer = GA_Trainer(
        param_bins=param_bins,
        fitness_func=fitness,
        num_generations=20,
        sol_per_pop=30,
        num_parents_mating=10,
        mutation_probability=0.05,
        on_generation=on_generation
    )

    best_individual = trainer.run(data_train=df_training,
                              dinero_invertido=10000,
                              data_test=df_test)
    print("\nMejor individuo final:")
    print(best_individual)
        



