import os
import pandas as pd
import shutil
import random
import re
from tqdm import tqdm

# Simple variation dictionary for semantic changes
variation_map = {
    "The Great Gatsby": ["Gatsby", "Great Gatsby by Fitzgerald", "F. Scott's Gatsby"],
    "Google": ["Alphabet Inc.", "GGL", "Google LLC"],
    "New York": ["NYC", "New York City", "Big Apple"],
    "Harvard University": ["Harvard", "Harvard College", "Hvd Univ"],
    "Elon Musk": ["E. Musk", "Elon", "Tesla CEO"],
    "Amazon": ["AMZN", "Amazon.com", "Amazon Inc."],
    "San Francisco": ["SF", "San Fran", "Bay Area"]
}

def get_semantic_variant(val):
    val = str(val)
    for key in variation_map:
        if val == key:
            return random.choice(variation_map[key])
    return val  # unchanged

def apply_variation(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        for i in range(len(df_copy)):
            val = df_copy.at[i, col]
            df_copy.at[i, col] = get_semantic_variant(val)
    return df_copy

def main():
    source_dir = os.path.join("datasets", "tptr_small", "datalake")
    dest_dir = os.path.join("datasets", "tptr_small_bert", "datalake")

    if os.path.exists(dest_dir):
        print(f"Removing existing BERT dataset directory: {dest_dir}")
        shutil.rmtree(dest_dir)

    os.makedirs(dest_dir)
    print(f"Copying and modifying datasets from {source_dir} → {dest_dir} ...")

    for filename in tqdm(os.listdir(source_dir)):
        if filename.endswith(".csv"):
            path = os.path.join(source_dir, filename)
            df = pd.read_csv(path)
            modified_df = apply_variation(df)
            modified_df.to_csv(os.path.join(dest_dir, filename), index=False)

    print(f"\n✅ BERT-variant dataset created at: {dest_dir}")

if __name__ == "__main__":
    main()
