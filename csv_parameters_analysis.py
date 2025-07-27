import os
import pandas as pd
import numpy as np

def analizar_indicadores(carpeta_csv):
    # Listar todos los archivos CSV en la carpeta
    archivos = [f for f in os.listdir(carpeta_csv) if f.endswith('.csv')]

    # Listas para acumular los datos
    todos_rsi = []
    todos_sma = []
    todos_atr = []

    # Leer cada archivo y acumular los valores de RSI, SMA y ATR
    for archivo in archivos:
        ruta = os.path.join(carpeta_csv, archivo)
        try:
            df = pd.read_csv(ruta)
            todos_rsi.extend(df['rsi'].dropna().tolist())
            todos_sma.extend(df['sma'].dropna().tolist())
            todos_atr.extend(df['atr'].dropna().tolist())
        except Exception as e:
            print(f"⚠️ Error leyendo {archivo}: {e}")

    # Convertir a arrays para análisis estadístico
    todos_rsi = np.array(todos_rsi)
    todos_sma = np.array(todos_sma)
    todos_atr = np.array(todos_atr)

    # Calcular cuartiles
    print("📊 Cuartiles RSI:")
    print(np.percentile(todos_rsi, [25, 50, 75]))
    print("\n📊 Cuartiles SMA:")
    print(np.percentile(todos_sma, [25, 50, 75]))
    print("\n📊 Cuartiles ATR:")
    print(np.percentile(todos_atr, [25, 50, 75]))

analizar_indicadores('data_small_processed')


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analizar_indicadores(carpeta_csv):
    archivos = [f for f in os.listdir(carpeta_csv) if f.endswith('.csv')]

    todos_rsi = []
    todos_sma = []
    todos_atr = []

    for archivo in archivos:
        ruta = os.path.join(carpeta_csv, archivo)
        try:
            df = pd.read_csv(ruta)
            todos_rsi.extend(df['rsi'].dropna().tolist())
            todos_sma.extend(df['sma'].dropna().tolist())
            todos_atr.extend(df['atr'].dropna().tolist())
        except Exception as e:
            print(f"⚠️ Error leyendo {archivo}: {e}")

    # Convertir a arrays
    todos_rsi = np.array(todos_rsi)
    todos_sma = np.array(todos_sma)
    todos_atr = np.array(todos_atr)

    # Crear histogramas
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    axs[0].hist(todos_rsi, bins=50, color='skyblue', edgecolor='black')
    axs[0].set_title('Histograma RSI')
    axs[0].set_xlabel('RSI')
    axs[0].set_ylabel('Frecuencia')

    axs[1].hist(todos_sma, bins=50, color='lightgreen', edgecolor='black')
    axs[1].set_title('Histograma SMA')
    axs[1].set_xlabel('SMA')
    axs[1].set_ylabel('Frecuencia')

    axs[2].hist(todos_atr, bins=50, color='salmon', edgecolor='black')
    axs[2].set_title('Histograma ATR')
    axs[2].set_xlabel('ATR')
    axs[2].set_ylabel('Frecuencia')

    plt.tight_layout()
    plt.show()

# Llama a la función
analizar_indicadores('data_small_processed')