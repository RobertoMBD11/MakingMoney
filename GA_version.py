import numpy as np
import random
import os
import pandas as pd
from tqdm import tqdm  # pip install tqdm
import datetime
import pickle
from utils import resumen_datos

# ==============================
# CONFIGURACIÓN
# ==============================
csv_folder = "csv_procesados"
features = ['MA_4', 'MA_8', 'MA_16', 'ATR', 'RSI', 'momentum', 'vol_rel']

population_size = 50     # Nº de individuos
epochs = 2              # Nº de generaciones
n_dfs_per_epoch = 14      # Nº de CSVs aleatorios por evaluación
mutation_sigma = 0.1
mutation_rate = 0.1
top_fraction = 0.1       # Top 10% sobrevivientes

layer_sizes = [len(features) + 1, 16, 3]  # input + hidden + output

#best_file_path = "results_20250901_192458/epoch07_fitness0.0781.pkl" 
best_file_path = None

# ==============================
# ID TRAINING
# ==============================
# ID único del entrenamiento
train_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_folder = f"results_{train_id}"
os.makedirs(results_folder, exist_ok=True)
print(f"Resultados se guardarán en: {results_folder}")

# ==============================
# CARGAR CSVS
# ==============================
all_dfs = []
for file in os.listdir(csv_folder):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(csv_folder, file))
        all_dfs.append(df)

print(f"Se han cargado {len(all_dfs)} CSVs procesados.")

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

def mutate(individual, sigma=mutation_sigma, limit=2.0, mutation_rate=mutation_rate):
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
# FUNCIONES DE EVALUACIÓN
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

def masked_softmax(logits, mask):
    masked_logits = np.where(mask, logits, -1e9)
    exps = np.exp(masked_logits - np.max(masked_logits))
    return exps / np.sum(exps)

def evaluate(individuo, data, features, commission=0.001):
    closes = data["close"].to_numpy(dtype=np.float32)
    X = data[features].to_numpy(dtype=np.float32)

    position_open = False
    entry_price = 0.0
    total_pnl = 0.0

    for i in range(len(closes)):
        price = closes[i]
        feat = X[i]

        if position_open:
            if entry_price != 0:
                unrealized_pnl = (price - entry_price) / entry_price
            else:
                unrealized_pnl = 0.0  # previene división por cero
        else:
            unrealized_pnl = 1.0

        # Aseguramos que x sea 1D float32
        x = np.hstack((feat, np.array([unrealized_pnl], dtype=np.float32)))

        logits = forward(individuo, x)
        mask = np.array([1, 0, 1]) if position_open else np.array([1, 1, 0])
        probs = masked_softmax(logits, mask)
        action = np.argmax(probs)

        if action == 1 and not position_open:
            position_open = True
            entry_price = price
        elif action == 2 and position_open:
            pnl = (price - entry_price) / entry_price - commission
            total_pnl += pnl
            position_open = False
            entry_price = 0.0

    return total_pnl


def evaluate_on_multiple(individuo, all_dfs, features, n_samples=8):
    sampled_dfs = random.sample(all_dfs, min(n_samples, len(all_dfs)))
    fitness_values = [evaluate(individuo, df, features) for df in sampled_dfs]
    return np.mean(fitness_values)

def evaluate_on_multiple_not_random(individuo, dfs, features):
    fitness_values = [evaluate(individuo, df, features) for df in dfs]
    return np.mean(fitness_values)

def evaluate_on_all(individuo, all_dfs, features):
    """
    Evalúa un individuo contra todos los CSVs cargados y devuelve:
    - fitness promedio
    - lista de fitness por CSV
    """
    fitness_list = [evaluate(individuo, df, features) for df in all_dfs]
    mean_fitness = np.mean(fitness_list)
    return mean_fitness, fitness_list

# ==============================
# INICIALIZACIÓN DE POBLACIÓN
# ==============================
population = [init_individual(layer_sizes) for _ in range(population_size)]

if best_file_path:
    with open(best_file_path, "rb") as f:
        best_individual_loaded = pickle.load(f)
    population[0] = best_individual_loaded

#print("Mejor individuo final evaluado:", evaluate_on_multiple(population[0], all_dfs, features, n_samples=n_dfs_per_epoch))
#mean_fitness_best, fitness_list = evaluate_on_all(population[0], all_dfs, features)
#print(f"Mean 7 years fitness: {mean_fitness_best}")
#resumen_datos(fitness_list, plot=True)

# ==============================
# BUCLE DE EVOLUCIÓN
# ==============================
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    # ==============================
    # Pre-selección de CSVs para toda la epoch
    # ==============================
    sampled_dfs = random.sample(all_dfs, min(n_dfs_per_epoch, len(all_dfs)))

    # ==============================
    # Evaluar población
    # ==============================
    fitness_list = []
    for ind in tqdm(population, desc="Evaluando individuos"):
        fit = evaluate_on_multiple_not_random(ind, sampled_dfs, features)
        fitness_list.append(fit)

    # ==============================
    # Ordenar por fitness descendente
    # ==============================
    sorted_indices = np.argsort(fitness_list)[::-1]
    population = [population[i] for i in sorted_indices]
    fitness_list = [fitness_list[i] for i in sorted_indices]

    # ==============================
    # Mostrar info
    # ==============================
    top_n = max(1, int(population_size * top_fraction))
    print(f"Mejor fitness: {fitness_list[0]:.4f} | "
          f"Promedio top 10%: {np.mean(fitness_list[:top_n]):.4f}")

    # ==============================
    # Guardar mejor individuo de la epoch
    # ==============================
    best_epoch_individual = population[0]
    mean_fitness_all = fitness_list[0]
    filename = f"epoch{epoch+1:02d}_fitness{mean_fitness_all:.4f}.pkl"
    filepath = os.path.join(results_folder, filename)
    with open(filepath, "wb") as f:
        pickle.dump(best_epoch_individual, f)
    print(f"Guardado mejor individuo de la epoch {epoch+1} en {filename}")

    # ==============================
    # Selección y mutación
    # ==============================
    top_individuals = population[:top_n]

    new_population = []
    while len(new_population) < population_size:
        parent = random.choice(top_individuals)
        child = mutate(parent)
        new_population.append(child)

    population = new_population


# ==============================
# Resultado final
# ==============================
best_individual = population[0]
print("Mejor individuo final evaluado:", evaluate_on_multiple(best_individual, all_dfs, features, n_samples=n_dfs_per_epoch))


mean_fitness_best = evaluate_on_all(best_individual, all_dfs, features)[0]
filename_best = f"final_best_fitness{mean_fitness_best:.4f}.pkl"
filepath_best = os.path.join(results_folder, filename_best)
with open(filepath_best, "wb") as f:
    pickle.dump(best_individual, f)
print(f"Guardado mejor individuo final en {filename_best}")
