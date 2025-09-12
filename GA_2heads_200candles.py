import numpy as np
import random
import os
import pandas as pd
import datetime
import pickle
from concurrent.futures import ProcessPoolExecutor
from utils import resumen_datos

# ==============================
# INICIALIZACIÓN DE FUNCIÓN DE INDIVIDUO
# ==============================
def init_individual(layer_sizes, limit=2.0):
    params = []
    for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        W = np.random.uniform(-limit, limit, size=(in_size, out_size))
        b = np.random.uniform(-limit, limit, size=(out_size,))
        params.extend([W, b])
    return params

def mutate(individual, sigma=0.1, limit=2.0, mutation_rate=0.1):
    new_individual = []
    for param in individual:
        param_copy = param.copy()
        mask = np.random.rand(*param_copy.shape) < mutation_rate
        noise = np.random.normal(0, sigma, size=param_copy.shape)
        param_copy[mask] += noise[mask]
        param_copy = np.clip(param_copy, -limit, limit)
        new_individual.append(param_copy)
    return new_individual

# ==============================
# FUNCIONES DE RED
# ==============================
def forward(individuo, x):
    a = x
    for i in range(0, len(individuo) - 2, 2):
        W, b = individuo[i], individuo[i+1]
        z = np.dot(a, W) + b
        a = np.maximum(0, z)  # ReLU
    W, b = individuo[-2], individuo[-1]
    logits = np.dot(a, W) + b
    return logits

# ==============================
# EVALUACIÓN OPEN-CLOSE
# ==============================
def evaluate_pair(ind_open, ind_close, data, features, commission=0.001):
    closes = data["close"].to_numpy(dtype=np.float32)
    X = data[features].to_numpy(dtype=np.float32)

    position_open = False
    entry_price = 0.0
    total_pnl = 0.0

    for i in range(len(closes)):
        price = closes[i]
        feat = X[i]

        if position_open:
            # CLOSE network
            x = np.hstack((feat, np.array([(price - entry_price) / (entry_price + 1e-8)], dtype=np.float32)))
            logits = forward(ind_close, x)
            action = int(logits[1] > logits[0])  # 1 = cerrar, 0 = mantener
            if action == 1:
                pnl = (price - entry_price) / (entry_price + 1e-8) - commission
                total_pnl += pnl
                position_open = False
                entry_price = 0.0
        else:
            # OPEN network
            if ind_open is None:
                action = 1  # primera generación CLOSE → siempre abrir si no hay posición
            else:
                x = np.hstack((feat, np.array([1.0], dtype=np.float32)))
                logits = forward(ind_open, x)
                action = int(logits[1] > logits[0])  # 1 = abrir, 0 = no hacer nada

            if action == 1:
                position_open = True
                entry_price = price

    return total_pnl


# ==============================
# FUNCIONES DE EVALUACIÓN EN PARALELO
# ==============================
def _evaluate_individual_open(args):
    ind_open, current_close, sampled_dfs, features = args
    return np.mean([evaluate_pair(ind_open, current_close, df, features) for df in sampled_dfs])

def _evaluate_individual_close(args):
    ind_close, current_open, sampled_dfs, features = args
    return np.mean([evaluate_pair(current_open, ind_close, df, features) for df in sampled_dfs])


# ==============================
# LOOP DE ENTRENAMIENTO
# ==============================
if __name__ == "__main__":
    # CONFIG
    test_folder = "test_csv_extended"
    csv_folder = "training_csv_extended"
    features = [
        'MA_10', 'MA_25', 'MA_50', 'MA_100', 'MA_200',
        'ATR', 'RSI', 'momentum', 'vol_rel'
    ]

    population_size = 120
    epochs = 10
    n_dfs_per_epoch = 45
    mutation_sigma = 0.2
    mutation_rate = 0.2
    mutation_limit = 10
    top_fraction = 0.15
    layer_sizes = [len(features) + 1, 16, 2]
    n_jobs = None  # Cambia a número de cores si quieres forzar paralelización

    train_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"results_{train_id}"
    os.makedirs(results_folder, exist_ok=True)

    results_open = os.path.join(results_folder, "open")
    results_close = os.path.join(results_folder, "close")
    os.makedirs(results_open, exist_ok=True)
    os.makedirs(results_close, exist_ok=True)

    # Cargar datos
    all_dfs_train = [pd.read_csv(os.path.join(csv_folder, f)) for f in os.listdir(csv_folder) if f.endswith(".csv")]
    all_dfs_test = [pd.read_csv(os.path.join(test_folder, f)) for f in os.listdir(test_folder) if f.endswith(".csv")]

    # Inicializar poblaciones
    population_open = [init_individual(layer_sizes) for _ in range(population_size)]
    population_close = [init_individual(layer_sizes) for _ in range(population_size)]

    best_open, best_close = None, None
    best_fitness_open, best_fitness_close = -np.inf, -np.inf

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        sampled_dfs = random.sample(all_dfs_train, min(n_dfs_per_epoch, len(all_dfs_train)))

        if epoch % 2 == 0:
            # ENTRENAR CLOSE
            print("Entrenando población CLOSE")
            current_open = best_open  # puede ser None en la primera generación

            tasks = [(ind, current_open, sampled_dfs, features) for ind in population_close]
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                fitness_list = list(executor.map(_evaluate_individual_close, tasks))

            sorted_indices = np.argsort(fitness_list)[::-1]
            population_close = [population_close[i] for i in sorted_indices]
            fitness_list = [fitness_list[i] for i in sorted_indices]

            top_n = max(1, int(population_size * top_fraction))
            print(f"Best CLOSE: {fitness_list[0]:.4f} | Mean top {top_n}: {np.mean(fitness_list[:top_n]):.4f}")

            if fitness_list[0] > best_fitness_close:
                best_fitness_close = fitness_list[0]
                best_close = population_close[0]
                with open(os.path.join(results_close, f"best_close_{best_fitness_close:.4f}.pkl"), "wb") as f:
                    pickle.dump(best_close, f)

            # Nueva generación
            top_individuals = population_close[:top_n]
            new_population = []
            while len(new_population) < population_size:
                parent = random.choice(top_individuals)
                child = mutate(parent, sigma=mutation_sigma, limit=mutation_limit, mutation_rate=mutation_rate)
                new_population.append(child)
            population_close = new_population

        else:
            # ENTRENAR OPEN
            print("Entrenando población OPEN")
            current_close = best_close if best_close is not None else random.choice(population_close)

            tasks = [(ind, current_close, sampled_dfs, features) for ind in population_open]
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                fitness_list = list(executor.map(_evaluate_individual_open, tasks))

            sorted_indices = np.argsort(fitness_list)[::-1]
            population_open = [population_open[i] for i in sorted_indices]
            fitness_list = [fitness_list[i] for i in sorted_indices]

            top_n = max(1, int(population_size * top_fraction))
            print(f"Best OPEN: {fitness_list[0]:.4f} | Mean top {top_n}: {np.mean(fitness_list[:top_n]):.4f}")

            if fitness_list[0] > best_fitness_open:
                best_fitness_open = fitness_list[0]
                best_open = population_open[0]
                with open(os.path.join(results_open, f"best_open_{best_fitness_open:.4f}.pkl"), "wb") as f:
                    pickle.dump(best_open, f)

            # Nueva generación
            top_individuals = population_open[:top_n]
            new_population = []
            while len(new_population) < population_size:
                parent = random.choice(top_individuals)
                child = mutate(parent, sigma=mutation_sigma, limit=mutation_limit, mutation_rate=mutation_rate)
                new_population.append(child)
            population_open = new_population


    # Guardar top 10 de OPEN
    top_10_open = os.path.join(results_open, "top_10")
    os.makedirs(top_10_open, exist_ok=True)
    for idx, ind in enumerate(population_open[:10]):
        filepath = os.path.join(top_10_open, f"open_top{idx+1}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(ind, f)

    # Guardar top 10 de CLOSE
    top_10_close = os.path.join(results_close, "top_10")
    os.makedirs(top_10_close, exist_ok=True)
    for idx, ind in enumerate(population_close[:10]):
        filepath = os.path.join(top_10_close, f"close_top{idx+1}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(ind, f)

    print("Entrenamiento finalizado. Guardados top 10 OPEN y CLOSE.")
