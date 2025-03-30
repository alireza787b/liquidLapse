import requests

url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
response = requests.get(url)
data = response.json()
btc_price = data['bitcoin']['usd']
print("Bitcoin Price (USD):", btc_price)
