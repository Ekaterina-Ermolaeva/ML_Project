import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'Titanic'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'House_prices'))

from Titanic_main import run_titanic_main
from House_prices_main import run_house_prices_main

titanic_results = run_titanic_main()

house_results = run_house_prices_main()

print("TITANIC RESULTS")
print(titanic_results)

print("HOUSE PRICES RESULTS")
print(house_results)

titanic_results.to_csv('titanic_results.csv', index=False)
house_results.to_csv('house_prices_results.csv', index=False)