import pandas as pd
import os

# Define file paths
RAW_PATH = "../data/raw/"
PROCESSED_PATH = "../data/processed/"

TRUE_FILE = os.path.join(RAW_PATH, "True.csv")
FAKE_FILE = os.path.join(RAW_PATH, "Fake.csv")

OUTPUT_FILE = os.path.join(PROCESSED_PATH, "cleaned_data.csv")


def load_data():
    print("Loading datasets...")

    true = pd.read_csv(TRUE_FILE, encoding="utf-8", quoting=1)
    fake = pd.read_csv(FAKE_FILE, encoding="utf-8", quoting=1)
    print(true.columns)
    print(fake.columns)

    print("Datasets loaded successfully.")

    return true, fake


def preprocess():
    true, fake = load_data()

    # Add labels
    true["label"] = 1   # Real news
    fake["label"] = 0   # Fake news

    # Combine datasets
    data = pd.concat([true, fake], axis=0)
    data = pd.concat([true, fake], axis=0)

    print("Before dropna:")
    print(data["label"].value_counts())
    print("Missing text values:")
    print(data["text"].isna().sum())

    # Shuffle data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Keep only relevant columns
    data = data[["text", "label"]]
    # Remove rows where text is missing
    data = data.dropna(subset=["text"]) 

    print("Data preprocessing complete.")

    return data


def save_data(data):
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    data.to_csv(OUTPUT_FILE, index=False)
    print(f"Cleaned data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    cleaned_data = preprocess()
    save_data(cleaned_data)