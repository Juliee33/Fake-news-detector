import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Locate dataset
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

df = pd.read_csv(data_path)

# Train/test split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]  # keeps class balance
)

print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

# Save files
train_path = os.path.join(BASE_DIR, "data", "processed", "train_data.csv")
test_path = os.path.join(BASE_DIR, "data", "processed", "test_data.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("Train and test datasets saved successfully.")