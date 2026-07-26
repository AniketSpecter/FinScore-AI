import pandas as pd
from sklearn.datasets import fetch_openml
import os

def fetch_and_save_dataset(save_path="german_credit.csv"):
    print("Fetching German Credit Risk dataset from OpenML...")
    # 'credit-g' is dataset ID 31 on OpenML
    data = fetch_openml(name='credit-g', version=1, as_frame=True, parser='auto')
    df = data.frame
    
    print(f"Dataset fetched successfully. Shape: {df.shape}")
    
    # Check for target column
    if 'class' in df.columns:
        # Standardize target column name
        df.rename(columns={'class': 'target'}, inplace=True)
        # Convert 'good'/'bad' to 1/0 where 1 means default/bad (High Risk)
        # In German credit data, 'good' means no default, 'bad' means default
        df['target'] = df['target'].map({'good': 0, 'bad': 1})
    
    df.to_csv(save_path, index=False)
    print(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "german_credit.csv")
    fetch_and_save_dataset(csv_path)
