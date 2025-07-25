import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




def compute_SMA_normalized(series, window):
    sma = series.rolling(window=window).mean()
    sma_normalized = series / sma
    return sma_normalized

def compute_RSI(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # evitar división por cero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_ATR_normalized(df, window=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    # ATR normalizado (dividido por el precio de cierre actual)
    atr_normalized = atr / df['close']
    
    return atr, atr_normalized


if __name__ == "__main__":
    # Carga el CSV generado previamente
    df = pd.read_csv("fake_crypto_day.csv", parse_dates=['timestamp'])

    # Crear el gráfico de precios
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['close'], label='Precio de Cierre', color='dodgerblue')
    plt.title('Precio Simulado de Criptomoneda (1 Día)')
    plt.xlabel('Hora')
    plt.ylabel('Precio (USD)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Asumiendo que ya tienes el DataFrame df con las columnas open, high, low, close, volume
    df['SMA_20'] = compute_SMA_normalized(df['close'], 20)
    df['RSI_14'] = compute_RSI(df['close'], 14)
    atr, atr_norm = compute_ATR_normalized(df, 14)
    #df['ATR_14'] = atr
    df['ATR_14'] = atr_norm

    # Mostrar primeras filas para verificar
    print(df[['timestamp', 'close', 'SMA_20', 'RSI_14', 'ATR_14']].head(24))
