import ccxt
import pandas as pd
import datetime

# Set up the exchange
exchange = ccxt.binance()

# Set the symbol and timeframe
symbol = 'BTC/USDT'
timeframe = '5m'

# Set start and end times (UTC)
start_time = datetime.datetime(2024, 5, 31, 0, 0, 0)  # Replace with your desired start time
# end_time = datetime.datetime(2023, 10, 26, 12, 0, 0)  # Replace with your desired end time

# Convert timestamps to milliseconds
start_time_ms = int(start_time.timestamp() * 1000)
# end_time_ms = int(end_time.timestamp() * 1000)

# Define the timeframe (5 minutes)
timeframe = '5m'

# Fetch historical data
data = exchange.fetch_ohlcv('BTC/USDT', timeframe, since=start_time_ms, limit=1000)

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])

# Convert the 'open_time' column to datetime
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

# Set the 'open_time' column as the index
df.set_index('open_time', inplace=True)

print(df)

df.to_csv("BTC-5m-500-OHLCV.csv",index=True)
