import ccxt
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm  # Para la barra de progreso

# 1. Configuración
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '5m'
limit = 1000  # Máximo por petición (Binance)
output_dir = 'data'
n_days = 365 * 10

# Crear carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# 2. Fechas
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=n_days)
since = int(start_date.timestamp() * 1000)
end_ts = int(end_date.timestamp() * 1000)

# 3. Descarga
all_data = []

print(f"Descargando velas de {symbol} desde {start_date.strftime('%Y-%m-%d')} hasta {end_date.strftime('%Y-%m-%d')}...\n")

with tqdm(total=(end_ts - since) // (5 * 60 * 1000), desc="Progreso (velas)") as pbar:
    while since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

            if not ohlcv:
                break

            all_data.extend(ohlcv)

            # Avanzar al siguiente bloque (después de la última vela)
            since = ohlcv[-1][0] + 1
            pbar.update(len(ohlcv))

            time.sleep(exchange.rateLimit / 1000)  # Respetar rate limit

        except Exception as e:
            print(f"Error: {e}. Esperando 5 segundos antes de continuar...")
            time.sleep(5)

# 4. Crear DataFrame
df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df['week'] = df['datetime'].dt.strftime('%Y-%W')

# 5. Guardar por semanas
print("\nGuardando CSVs semanales...")

for week, group in df.groupby('week'):
    filename = os.path.join(output_dir, f'{symbol.replace("/", "")}_week_{week}.csv')
    group[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_csv(filename, index=False)

print("✅ Datos guardados correctamente.")
