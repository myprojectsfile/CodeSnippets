import requests
from bs4 import BeautifulSoup
import csv

# URL for the NASDAQ Composite stock list on NASDAQ's website
nasdaq_composite_url = "https://www.nasdaq.com/market-activity/stocks/screener"

def fetch_nasdaq_stocks():
    # Send a request to fetch the NASDAQ Composite page
    response = requests.get(nasdaq_composite_url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Find the table containing the NASDAQ stocks
    stocks = []
    for row in soup.findAll("tr")[1:]:  # Skip the header row
        columns = row.findAll("td")
        if columns and len(columns) > 1:
            ticker = columns[0].text.strip()
            stocks.append(ticker)
    
    return stocks

# Generate a CSV file for TradingView import
def create_tradingview_watchlist(stocks, filename="nasdaq_full_watchlist.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Symbol"])
        for stock in stocks:
            # Format for TradingView: 'NASDAQ:STOCK_SYMBOL'
            writer.writerow([f"NASDAQ:{stock}"])

# Fetch NASDAQ stocks and generate watchlist
nasdaq_stocks = fetch_nasdaq_stocks()
create_tradingview_watchlist(nasdaq_stocks)

print(f"NASDAQ Composite watchlist generated with {len(nasdaq_stocks)} symbols.")
