import pandas as pd
from sklearn.model_selection import train_test_split
import os

# -------------------------------
# Load raw data
# -------------------------------
def load_data(path):
    df = pd.read_csv(path)
    print("✨ First few rows of the raw data：")
    print(df.head())
    return df

# -------------------------------
# Basic text cleaning (extendable)
# -------------------------------
def clean_text(df):
    df["text"] = df["text"].astype(str).str.strip()
    return df

# -------------------------------
# Convert stereotype / non-stereotype to 1 / 0
# -------------------------------
def encode_labels(df):
    label_map = {
        "stereotype": 1,
        "non-stereotype": 0
    }
    df["label"] = df["label"].map(label_map)
    return df

# -------------------------------
# Split into train, validation, and test sets
# -------------------------------
def split_data(df):
    train, temp = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"]
    )
    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=42,
        stratify=temp["label"]
    )

    print(f"📊 Dataset split results：")
    print(f"Train: {len(train)} samples")
    print(f"Val:   {len(val)} samples")
    print(f"Test:  {len(test)} samples")

    return train, val, test

# -------------------------------
# Save datasets
# -------------------------------
def save_splits(train, val, test, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    train.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(save_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(save_dir, "test.csv"), index=False)

    print(f"✅ Saved to: {save_dir}")

# -------------------------------
# Main execution flow
# -------------------------------
if __name__ == "__main__":
    raw_path = "local_context/data/chinese_stereotypes.csv"
    save_dir = "local_context/data/processed/"

    df = load_data(raw_path)
    df = clean_text(df)
    df = encode_labels(df)

    train, val, test = split_data(df)
    save_splits(train, val, test, save_dir)

    print("🎉 Preprocessing completed！")
