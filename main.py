import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score


# When debug is True, only load a subset of rows to iterate quickly
debug = True
sample_nrows = 100000 if debug else None

data_path = Path(__file__).parent / "GUIDE_Train.csv"
print(f"Loading data from {data_path.name} (debug={debug}, nrows={sample_nrows})...")
df = pd.read_csv(data_path, nrows=sample_nrows)

# Drop alerts with unannotated IncidentGrade (missing target)
df = df.dropna(subset=["IncidentGrade"]).copy()

df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
df = df.sort_values("Timestamp").reset_index(drop=True)


keep_columns = [
    "Timestamp", # timestamp of the alert which will be used for engineering 'TimeLastAlert', 'HourOfDay', 'DayOfWeek', and 'IsWeekend' features.
    "Category", # category of the alert.
    "MitreTechniques", #MITRE ATT&CK techniques used in the alert. Converted to 0 or 1 for binary classification.
    "EntityType", # Type of the entity that the alert is associated with.
    "DetectorId", # unique identifier for the detector that generated the alert.
    "AccountSid", # unique identifier for the account that the alert is associated with.
    "DeviceId", # unique identifier for the device that the alert is associated with.
    "IpAddress", # IP address of the device that the alert is associated with.
    "ApplicationId", # unique identifier for the application that the alert is associated with.
]

X_train = df[keep_columns].copy()
y_train = df["IncidentGrade"].copy() # customer-provided triage label.

# ======================================================================================
# Engineer the features

## Map categorical features to numerical features
### Get unique values of 'Category' and map to integers
category_mapping = {cat: i for i, cat in enumerate(X_train["Category"].dropna().unique())}
X_train["Category"] = X_train["Category"].map(category_mapping).fillna(-1).astype(int)

### Get unique values of 'EntityType' and map to integers
entity_type_mapping = {et: i for i, et in enumerate(X_train["EntityType"].dropna().unique())}
X_train["EntityType"] = X_train["EntityType"].map(entity_type_mapping).fillna(-1).astype(int)

### Map 'MitreTechniques': 0 if missing/empty, 1 if present
X_train["MitreTechniques"] = X_train["MitreTechniques"].notna().astype(int)

## ======================================================================================
## Time-related features (We will characterize the patterns of alerts over time according to normal business operating hours.)
### TimeLastAlert (Time since last alert in seconds for each DetectorId; fill first alert with -1)
X_train["TimeLastAlert"] = (
    X_train.groupby("DetectorId")["Timestamp"].diff().dt.total_seconds().fillna(-1)
)
### HourOfDay
X_train["HourOfDay"] = X_train["Timestamp"].dt.hour.astype(int)
### DayOfWeek
X_train["DayOfWeek"] = X_train["Timestamp"].dt.dayofweek.astype(int)
### IsWeekend
X_train["IsWeekend"] = X_train["Timestamp"].dt.dayofweek.isin([5, 6]).astype(int)

## ======================================================================================
## Frequency-related features (Count occurrences of each unique value in the past X hours until the alert)
## We will be using a rolling window of 24 hours to characterize the daily operational patterns of the entities.
time_window = "24h"
frequency_columns = [
    "DetectorId",
    "AccountSid",
    "DeviceId",
    "IpAddress",
    "ApplicationId",
]

for column in frequency_columns:
    print(f"Computing 24h rolling count for {column}...")
    rolling_counts = (
        X_train[[column, "Timestamp"]]
        .assign(_alert=1)
        .groupby(column, dropna=False, sort=False)
        .rolling(window=time_window, on="Timestamp", closed="both")
        .sum()["_alert"]
        .reset_index(level=0, drop=True)
    )
    X_train[f"{column}AlertFrequency"] = rolling_counts.fillna(0).astype("int64")

## ======================================================================================
## Final Processing - Drop the raw identifier and timestamp features
X_train = X_train.drop(
    columns=[
        "Timestamp",
        "DetectorId",
        "AccountSid",
        "DeviceId",
        "IpAddress",
        "ApplicationId",
    ]
)

print("\n--- Features Sample ---")
print(X_train.head(10))

print("\n--- Dataset Info ---")
print(X_train.info())

print("\n--- Unique values per feature ---")
print(X_train.nunique())


# =======================================================================================
# Plot feature distributions (sample up to 50k to ensure fast rendering)
feature_columns = X_train.select_dtypes(include=np.number).columns
n_columns = 3
n_rows = int(np.ceil(len(feature_columns) / n_columns))

fig, axes = plt.subplots(n_rows, n_columns, figsize=(5 * n_columns, 4 * n_rows))
axes = np.atleast_1d(axes).ravel()

plot_sample = X_train.sample(n=min(len(X_train), 50000), random_state=42)

for ax, feature in zip(axes, feature_columns):
    values = plot_sample[feature].dropna()
    sns.histplot(values, bins=30, ax=ax)
    ax.set_title(feature)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")

for ax in axes[len(feature_columns):]:
    ax.remove()

fig.suptitle("Feature distributions (Sampled)", fontsize=16)
fig.tight_layout()
plt.show()