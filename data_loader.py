import pandas as pd

def load_data(path='Housing.csv'):
    return pd.read_csv(path)
