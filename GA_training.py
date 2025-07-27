from day_visor import compute_RSI, compute_SMA_normalized, compute_ATR_normalized
from GA_population import get_code_from_candle, get_semicode_from_candle, get_code_from_semicode_and_state, Population, Individual
import pandas as pd
import numpy as np
import pygad
import matplotlib.pyplot as plt
from GA_model import GA_Trainer
import random
import os


import pandas as pd

def fitness(individuo, archivos_csv, dinero_invertido):
    total_beneficio = 0
    num_archivos_validos = 0

    for file_path in archivos_csv:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error cargando {file_path}: {e}")
            continue

        estado = 0
        precio_entrada = None
        unidades = 0
        beneficios = []

        for _, row in df.iterrows():
            semicode = row['semicode']
            code = get_code_from_semicode_and_state(semicode, estado)
            accion = individuo.get(code, 0)

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

        total_beneficio += sum(beneficios)
        num_archivos_validos += 1

    if num_archivos_validos == 0:
        return 0  # Evitar división por cero si hubo error en todos
    return total_beneficio / num_archivos_validos

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

    # AÑADIR el path completo a cada archivo
    train_files = [os.path.join(folder_path, f) for f in train_files]
    test_files = [os.path.join(folder_path, f) for f in test_files]

    return train_files, test_files

def process_and_save_csvs(input_folder):
    # Carpeta de salida
    output_folder = input_folder.rstrip("/\\") + "_processed"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_folder, filename)
            #print(f"Procesando {input_path}...")

            # Leer CSV con parseo de fecha
            df = pd.read_csv(input_path, parse_dates=['timestamp'])

            # Calcular indicadores
            df['rsi'] = compute_RSI(df['close'])
            df['sma'] = compute_SMA_normalized(df['close'], window=14)
            _, df['atr'] = compute_ATR_normalized(df, window=14)

            # Quitar filas con NaN
            df.dropna(inplace=True)

            # Calcular semicode
            df['semicode'] = df.apply(lambda row: get_semicode_from_candle(row, param_bins), axis=1)

            # Guardar CSV procesado
            base_name = os.path.splitext(filename)[0]
            output_file = base_name + "_processed.csv"
            output_path = os.path.join(output_folder, output_file)
            df.to_csv(output_path, index=False)

            #print(f"Guardado en {output_path}")
    return output_folder

if __name__ == "__main__":
    acciones = {0: "NO HACER NADA", 1: "ENTRAR", 2: "MANTENER", 3: "SALIR"}

    param_bins = {
        'rsi': [20, 30, 40, 50, 60, 70, 80],
        'sma': [0.996, 0.998, 0.999, 1.0, 1.001, 1.002, 1.004],
        'atr': [0.0006, 0.001, 0.0015, 0.0014, 0.0018, 0.002, 0.0035]
    }

    print("Processing files...")
    input_foler = "data_small"
    #output_folder = "data_small_processed"
    output_folder = process_and_save_csvs(input_foler)
    train_files, test_files = split_dataset_folder(output_folder)
    print("All parameters calculated")


    trainer = GA_Trainer(
        param_bins=param_bins,
        fitness_func=fitness,
        num_generations=20,
        sol_per_pop=30,
        num_parents_mating=10,
        mutation_probability=0.05,
        on_generation=on_generation
    )
    

    best_individual = trainer.run(data_train=train_files,
                              dinero_invertido=1,
                              data_test=test_files)
    
    print("\nMejor individuo final:")
    print(best_individual)
        

    ### Análisis
    def analizar_conservacion_genetica(poblacion):
        """
        Dada una población de individuos con genes binarios, calcula
        el porcentaje de 1s y 0s en cada posición del cromosoma.

        Devuelve:
            - una lista de tuplas (porcentaje_de_1s, porcentaje_de_0s) por índice
            - y una lista con el alelo mayoritario por índice
        """
        # Convertir genes a array de forma (n_individuos, n_genes)
        matriz_genes = np.array([ind.genes for ind in poblacion])
        num_individuos = matriz_genes.shape[0]

        resultados = []
        alelo_mayoritario = []

        for i in range(matriz_genes.shape[1]):
            columna = matriz_genes[:, i]
            num_1s = np.sum(columna)
            num_0s = num_individuos - num_1s

            pct_1 = num_1s / num_individuos
            pct_0 = num_0s / num_individuos

            resultados.append((pct_1, pct_0))
            alelo_mayoritario.append(1 if pct_1 >= pct_0 else 0)

        return resultados, alelo_mayoritario
    
    resumen, mayoritarios = analizar_conservacion_genetica(trainer.population.individuals)

    for i, (p1, p0) in enumerate(resumen):
        print(f"Gen {i}: 1s = {p1:.2%}, 0s = {p0:.2%}, mayoritario = {mayoritarios[i]}")

