import pandas as pd

def load_csv_file(filepath, label=None):
    df = pd.read_csv(filepath, header=None)
    df.columns = ['Time', 'X', 'Y', 'Z', 'Absolute']
    df = df[['X', 'Y', 'Z']]
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    if label:
        df['label'] = label
    return df