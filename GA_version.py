import numpy as np
import random
import os
import pandas as pd
from tqdm import tqdm  # pip install tqdm
import datetime
import pickle
from utils import resumen_datos
import glob
from concurrent.futures import ProcessPoolExecutor


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


def evaluate_on_multiple(individuo, all_dfs, features, n_samples=8, random=False):
    if random:
        sampled_dfs = random.sample(all_dfs, min(n_samples, len(all_dfs)))
    else:
        sampled_dfs = all_dfs[:n_samples]  # toma los primeros n_samples de la lista

    fitness_values = [evaluate(individuo, df, features) for df in sampled_dfs]
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

def _evaluate_individual(args):
    ind, sampled_dfs, features = args
    # aquí usamos evaluate directamente en todos los dfs
    return np.mean([evaluate(ind, df, features) for df in sampled_dfs])

def evaluate_population(population, sampled_dfs, features, n_jobs=None):
    tasks = [(ind, sampled_dfs, features) for ind in population]
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        fitness_list = list(executor.map(_evaluate_individual, tasks))
    return fitness_list

if __name__ == "__main__":
    # ==============================
    # CONFIGURACIÓN
    # ==============================
    csv_folder = "training_csv"
    #csv_folder = "dias_estables"
    features = ['MA_4', 'MA_8', 'MA_16', 'ATR', 'RSI', 'momentum', 'vol_rel']

    population_size = 100     # Nº de individuos
    epochs = 60              # Nº de generaciones
    n_dfs_per_epoch = 100      # Nº de CSVs aleatorios por evaluación
    mutation_sigma = 0.1
    mutation_rate = 0.1
    top_fraction = 0.3       # Top 10% sobrevivientes

    layer_sizes = [len(features) + 1, 16, 3]  # input + hidden + output

    best_file_path = None
    #best_file_path = "results_20250902_125247_60epocs_0056/last_best_epoch02.pkl" 
    #best_file_path = "results_20250905_172330_trained_from_normal/best_0.0143.pkl"
    #best_file_path = "results_20250905_175715/final_best_fitness0.0072.pkl"
    
    previous_top_folder = None
    #previous_top_folder = "results_20250902_125247_60epocs_0056/top_10"
    #previous_top_folder = "results_20250905_175715/top10"
    
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
    # INICIALIZACIÓN DE POBLACIÓN
    # ==============================
    population = [init_individual(layer_sizes) for _ in range(population_size)]

    if best_file_path:
        with open(best_file_path, "rb") as f:
            best_individual_loaded = pickle.load(f)
        population[0] = best_individual_loaded

    if previous_top_folder:
        import glob
        top_files = sorted(glob.glob(os.path.join(previous_top_folder, "*.pkl")))
        top_individuals_loaded = []

        for file in top_files:
            with open(file, "rb") as f:
                ind = pickle.load(f)
                top_individuals_loaded.append(ind)

        # Sustituir los 10 primeros individuos
        for i, ind in enumerate(top_individuals_loaded[:10]):
            population[i] = ind

        print(f"Sobrescritos {len(top_individuals_loaded[:10])} individuos con el top 10 previo")

    """mean_fitness_best, fitness_list = evaluate_on_all(population[0], all_dfs, features)
    print(f"Mean 7 years fitness: {mean_fitness_best}")
    resumen_datos(fitness_list, plot=True)"""

    # ==============================
    # BUCLE DE EVOLUCIÓN CON NUEVOS CHECKPOINTS
    # ==============================
    historical_best_individuals = []  # guardaremos tuplas (fitness, individuo)

    best_fitness_overall = -np.inf
    best_individual_overall = None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Pre-selección de CSVs para toda la epoch
        sampled_dfs = random.sample(all_dfs, min(n_dfs_per_epoch, len(all_dfs)))

        # Evaluar población
        fitness_list = []

        print("Evaluando individuos en paralelo...")
        fitness_list = evaluate_population(population, sampled_dfs, features, n_jobs=None)

        # Ordenar por fitness descendente
        sorted_indices = np.argsort(fitness_list)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_list = [fitness_list[i] for i in sorted_indices]

        # Mostrar info
        top_n = max(1, int(population_size * top_fraction))
        print(f"Mejor fitness epoch: {fitness_list[0]:.4f} | "
            f"Promedio top 10%: {np.mean(fitness_list[:top_n]):.4f}")

        # ==============================
        # Guardar checkpoints
        # ==============================
        # 1️⃣ Mejor individuo de la epoch (sobreescribe cada epoch)
        last_best_path = os.path.join(results_folder, f"last_best.pkl")
        with open(last_best_path, "wb") as f:
            pickle.dump(population[0], f)

        # 2️⃣ Mejor individuo histórico hasta ahora
        if fitness_list[0] > best_fitness_overall:
            best_fitness_overall = fitness_list[0]
            best_individual_overall = population[0]
            best_path = os.path.join(results_folder, f"best_{best_fitness_overall:.4f}.pkl")
            with open(best_path, "wb") as f:
                pickle.dump(best_individual_overall, f)

        # Guardar todos los mejores históricos para top 10 final
        historical_best_individuals.append((fitness_list[0], population[0]))

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
    # Guardar top 10 de la última generación
    # ==============================
    top_10_folder = os.path.join(results_folder, "top_10")
    os.makedirs(top_10_folder, exist_ok=True)

    # Ordenar la población final por fitness
    fitness_list_final = []
    for ind in population[:10]:
        fit = evaluate_on_all(ind, all_dfs, features)[0]
        fitness_list_final.append(fit)

    sorted_indices = np.argsort(fitness_list_final)[::-1]
    population_sorted = [population[i] for i in sorted_indices]
    fitness_sorted = [fitness_list_final[i] for i in sorted_indices]

    # Guardar los 10 mejores individuos
    for idx in range(min(10, len(population_sorted))):
        ind = population_sorted[idx]
        fit = fitness_sorted[idx]
        filepath = os.path.join(top_10_folder, f"top{idx+1}_fitness{fit:.4f}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(ind, f)

    print(f"Guardados los 10 mejores individuos de la última generación en {top_10_folder}")


    # ==============================
    # Resultado final
    # ==============================
    best_individual = population[0]

    mean_fitness_best, fitness_list = evaluate_on_all(best_individual, all_dfs, features)
    print(f"Mean 7 years fitness: {mean_fitness_best}")
    resumen_datos(fitness_list, plot=True)


    filename_best = f"final_best_fitness{mean_fitness_best:.4f}.pkl"
    filepath_best = os.path.join(results_folder, filename_best)
    with open(filepath_best, "wb") as f:
        pickle.dump(best_individual, f)
    print(f"Guardado mejor individuo final en {filename_best}")
