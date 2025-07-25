from day_visor import compute_RSI, compute_SMA_normalized, compute_ATR_normalized
from GA_population import get_code_from_candle, get_semicode_from_candle, get_code_from_semicode_and_state, Population
import pandas as pd

def fitness(individuo, df, param_bins, dinero_invertido):
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

        elif estado == 1 and accion == 3:  # SALIR
            dinero_final = unidades * precio_actual
            beneficio = dinero_final - dinero_invertido
            beneficios.append(beneficio)
            estado = 0
            unidades = 0
            precio_entrada = None

    return sum(beneficios)



if __name__ == "__main__":
    acciones = {0: "NO HACER NADA", 1: "ENTRAR", 2: "MANTENER", 3: "SALIR"}

    param_bins = {
        'rsi': [20, 40, 60, 80],
        'sma': [0.95, 1.0, 1.05, 1.1],
        'atr': [0.005, 0.01, 0.02, 0.05]
    }

    # 1. Crear población
    population = Population(param_bins=param_bins, num_individuals=5)

    print("Número total de genes:", population.num_genes)

    # 2. Cargar datos y calcular indicadores
    df = pd.read_csv("fake_crypto_day.csv", parse_dates=['timestamp'])
    df['rsi'] = compute_RSI(df['close'])
    df['sma'] = compute_SMA_normalized(df['close'], window=14)
    _, df['atr'] = compute_ATR_normalized(df, window=14)
    df.dropna(inplace=True)
    df['semicode'] = df.apply(lambda row: get_semicode_from_candle(row, param_bins), axis=1)

    # 3. Evaluar fitness
    individuo = population[0]
    dinero_invertido = 1000
    fitness_score = fitness(individuo, df, param_bins, dinero_invertido)

    print("Fitness del individuo 0:", fitness_score)

