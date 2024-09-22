import requests
import csv

# MEXC API URL for perpetual futures symbols
url = "https://contract.mexc.com/api/v1/contract/detail"

def fetch_perpetual_usdt_pairs():
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        pairs = []
        for contract in data['data']:
            # Correct keys and active state check
            if contract['quoteCoin'] == 'USDT' and contract['state'] == 0:
                # Remove underscore and add ".P" for perpetual contracts
                formatted_pair = contract['symbol'].replace('_', '').upper() + '.P'
                pairs.append(formatted_pair)
        return pairs
    else:
        print("Error fetching data from MEXC.")
        return []

# Fetch perpetual USDT pairs
pairs = fetch_perpetual_usdt_pairs()

# Generate a CSV file for TradingView import
def create_tradingview_watchlist(pairs, filename="mexc_perpetual_watchlist.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Symbol"])
        for pair in pairs:
            # Assuming the format required by TradingView is `EXCHANGE:SYMBOL`
            writer.writerow([f"MEXC:{pair}"])

if pairs:
    create_tradingview_watchlist(pairs)
    print(f"Watchlist generated: {len(pairs)} pairs saved to 'mexc_perpetual_watchlist.txt'")
else:
    print("No pairs to save.")
