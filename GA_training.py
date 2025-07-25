from day_visor import compute_RSI, compute_SMA_normalized, compute_ATR_normalized
from GA_population import get_code_from_candle, generate_all_codes, generate_random_individuals
import pandas as pd
import random

def fitness(individuo, df, param_bins, dinero_invertido):
  
    estado = 0
    precio_entrada = None
    unidades = 0
    beneficios = []

    for idx, row in df.iterrows():
        code = get_code_from_candle(row, estado, param_bins)
        accion = individuo.get(code, 0)  # por defecto no hacer nada
        
        precio_actual = row['close']
        
        if estado == 0:
            if accion == 1:  # ENTRAR
                unidades = dinero_invertido / precio_actual
                precio_entrada = precio_actual
                estado = 1
        elif estado == 1:
            if accion == 3:  # SALIR
                dinero_final = unidades * precio_actual
                beneficio = dinero_final - dinero_invertido
                beneficios.append(beneficio)
                estado = 0
                unidades = 0
                precio_entrada = None
        # Si la acción es 0 o 2, simplemente mantener o no hacer nada

    return sum(beneficios)


# Execution
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

df = pd.read_csv("fake_crypto_day.csv", parse_dates=['timestamp'])
# Calcular indicadores
df['rsi'] = compute_RSI(df['close'])
df['sma'] = compute_SMA_normalized(df['close'], window=14)
_, df['atr'] = compute_ATR_normalized(df, window=14)
df.dropna(inplace=True)

individuo = population[0]

# 5. Evalúa su fitness con tu df y capital
dinero_invertido = 1000  # por ejemplo
fitness_score = fitness(individuo, df, param_bins, dinero_invertido)

print("Fitness del individuo 0:", fitness_score)