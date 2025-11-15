import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_CSV_DIR = "CIC-IDS2017-CSV"  # change to your CSV folder
OUT_DIR = "processed_cicids2017"
os.makedirs(OUT_DIR, exist_ok=True)

all_files = glob.glob(os.path.join(DATA_CSV_DIR, "*.csv"))
dfs = [pd.read_csv(f) for f in all_files]
data = pd.concat(dfs, ignore_index=True)

# Convert object columns to numeric labels
for c in data.columns:
    if data[c].dtype == 'object':
        data[c] = LabelEncoder().fit_transform(data[c].astype(str))

# Features & labels
X = data.drop("Label", axis=1).values.astype(np.float32)
y = LabelEncoder().fit_transform(data["Label"].values)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)

print(f"Saved processed files in {OUT_DIR}")