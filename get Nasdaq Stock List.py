import requests
from bs4 import BeautifulSoup
import csv

# URL for the Wikipedia page listing NASDAQ-100 stocks
nasdaq_100_url = "https://en.wikipedia.org/wiki/NASDAQ-100"

def fetch_nasdaq_100_stocks():
    # Send a request to fetch the NASDAQ-100 page
    response = requests.get(nasdaq_100_url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Find the table containing the NASDAQ-100 stocks
    table = soup.find("table", {"class": "wikitable sortable"})
    tickers = []

    # Loop through the rows of the table to extract stock symbols
    for row in table.findAll("tr")[1:]:  # Skip the header row
        ticker = row.findAll("td")[1].text.strip()
        tickers.append(ticker)
    
    return tickers

# Generate a CSV file for TradingView import
def create_tradingview_watchlist(stocks, filename="nasdaq_watchlist.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Symbol"])
        for stock in stocks:
            # Format for TradingView: 'NASDAQ:STOCK_SYMBOL'
            writer.writerow([f"NASDAQ:{stock}"])

# Fetch NASDAQ-100 stocks and generate watchlist
nasdaq_stocks = fetch_nasdaq_100_stocks()
create_tradingview_watchlist(nasdaq_stocks)

print(f"NASDAQ-100 watchlist generated with {len(nasdaq_stocks)} symbols.")
