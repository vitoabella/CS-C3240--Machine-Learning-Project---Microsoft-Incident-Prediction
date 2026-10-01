import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score


# When debug is True, only load a subset of rows to iterate quickly
debug = False
sample_nrows = 10000 if debug else None

data_path = Path(__file__).parent / "GUIDE_Test.csv"
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
    "IncidentGrade", # customer-provided triage label (target).
]

X_train = df[keep_columns].copy()

# ======================================================================================
# Engineer the features

## Map categorical features to numerical features
### Map 'MitreTechniques': 0 if missing/empty, 1 if present
X_train["MitreTechniques"] = X_train["MitreTechniques"].notna().astype(int)

## Time-related features (We will characterize the patterns of alerts over time according to normal business operating hours.)
### TimeLastAlert (Time since last alert in seconds for each DetectorId)
X_train["TimeLastAlert"] = (
    X_train.groupby("DetectorId")["Timestamp"].diff().dt.total_seconds()
)
### HourOfDay
X_train["HourOfDay"] = X_train["Timestamp"].dt.hour.astype(int)
### DayOfWeek
X_train["DayOfWeek"] = X_train["Timestamp"].dt.dayofweek.astype(int)
### IsWeekend
X_train["IsWeekend"] = X_train["Timestamp"].dt.dayofweek.isin([5, 6]).astype(int)

## Frequency-related features (Count occurrences of each unique value in the past <time_window> hours until the alert)
## We will be using a rolling window of 24 hours to characterize the daily operational patterns of the features.
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

## Final Processing - Drop first alert per detector (no prior alert) and drop raw identifier features
X_train = X_train.dropna(subset=["TimeLastAlert"]).reset_index(drop=True)
y_train = X_train.pop("IncidentGrade")

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

print("\n--- Target (IncidentGrade) Distribution ---")
print(y_train.value_counts())

# =======================================================================================
# Plot feature distributions
# Plots display distinct category/value distributions as bar graphs (without one-hot explosion)
feature_columns = list(X_train.columns)
n_columns = 3
n_rows = int(np.ceil(len(feature_columns) / n_columns))

fig, axes = plt.subplots(n_rows, n_columns, figsize=(5.5 * n_columns, 4.5 * n_rows))
axes = np.atleast_1d(axes).ravel()

plot_sample = X_train

for ax, feature in zip(axes, feature_columns):
    values = plot_sample[feature].dropna()
    if feature == "TimeLastAlert":
        # TimeLastAlert ranges from seconds to months; bin into intuitive operational recency buckets
        bins = [-np.inf, 60, 300, 1800, 3600, 21600, 86400, 604800, np.inf]
        labels = ["< 1m", "1-5m", "5-30m", "30m-1h", "1-6h", "6-24h", "1-7d", "> 7d"]
        binned = pd.cut(values, bins=bins, labels=labels)
        counts = binned.value_counts(sort=False)
        ax.bar(range(len(counts)), counts.values, color="steelblue", edgecolor="black", alpha=0.8)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right", fontsize=8)
        ax.set_title("TimeLastAlert (Recency Buckets)")
        ax.set_xlabel("Elapsed Time")
        ax.set_ylabel("Count")
    elif not pd.api.types.is_numeric_dtype(values) or values.nunique() <= 24:
        # Discrete / Categorical: bar graph showing frequency of each distinct value/category
        counts = values.value_counts().sort_index() if pd.api.types.is_numeric_dtype(values) else values.value_counts()
        ax.bar(range(len(counts)), counts.values, color="steelblue", edgecolor="black", alpha=0.8)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index.astype(str), rotation=45 if values.nunique() > 6 else 0, ha="right", fontsize=8)
        ax.set_title(feature)
        ax.set_xlabel("Value / Category")
        ax.set_ylabel("Count")
    else:
        # Continuous / High-cardinality counts: binned bar chart (histogram)
        sns.histplot(values, bins=25, ax=ax, color="steelblue", edgecolor="black", alpha=0.8)
        ax.set_title(feature)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

for ax in axes[len(feature_columns):]:
    ax.remove()

fig.suptitle("Feature distributions (Sampled)", fontsize=16)
fig.tight_layout(h_pad=3.0, w_pad=1.5)

# =======================================================================================
# Plot target (y_train / IncidentGrade) distribution
fig_target, ax_target = plt.subplots(figsize=(6, 4.5))
target_counts = y_train.value_counts()
bars = ax_target.bar(
    target_counts.index.astype(str),
    target_counts.values,
    color="steelblue",
    edgecolor="black",
    alpha=0.8,
)
ax_target.set_title("Target Distribution: IncidentGrade (y_train)", fontsize=13)
ax_target.set_xlabel("Incident Grade")
ax_target.set_ylabel("Count")

# Add count and percentage labels on top of each bar
total_samples = len(y_train)
for bar in bars:
    height = bar.get_height()
    pct = (height / total_samples) * 100
    ax_target.annotate(
        f"{int(height):,}\n({pct:.1f}%)",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax_target.set_ylim(0, max(target_counts.values) * 1.15)
fig_target.tight_layout()

plt.show()

# =======================================================================================
# One-hot encode categorical features for Decision Tree model
X_train = pd.get_dummies(X_train, columns=["Category", "EntityType"], dtype=int)
print(f"\nFinal one-hot encoded features shape for model: {X_train.shape}")