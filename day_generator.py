import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parámetros de la simulación
np.random.seed(41)
n_intervals = 288  # 24h con datos cada 5 minutos
start_price = 10000  # Precio inicial
mu = 0.0002  # Tendencia diaria esperada (positiva)
sigma = 0.002  # Volatilidad (ajustable)
start_time = datetime(2024, 1, 1, 0, 0)

# Generar timestamps
timestamps = [start_time + timedelta(minutes=5*i) for i in range(n_intervals)]

# Generar precios usando un proceso estocástico tipo GBM discreto
returns = np.random.normal(loc=mu, scale=sigma, size=n_intervals)
log_prices = np.log(start_price) + np.cumsum(returns)
prices = np.exp(log_prices)

# Generar OHLC
df = pd.DataFrame({
    'timestamp': timestamps,
    'close': prices
})
df['open'] = df['close'].shift(1).fillna(df['close'][0])
df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 0.002, size=n_intervals) * df['close']
df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 0.002, size=n_intervals) * df['close']

# Volumen simulado basado en movimiento de precio
volatility = (df['high'] - df['low']) / df['open']
df['volume'] = (np.random.rand(n_intervals) + volatility * 50) * 100

# Reordenar columnas y guardar como CSV
df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
df.to_csv("fake_crypto_day.csv", index=False)

print("CSV generado: fake_crypto_day.csv")
