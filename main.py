"""
Microsoft Security Incident Prediction - Feature Preprocessing Pipeline

Workflow:
1. Load dataset (train data)
2. Pre-process hand-picked features (one-hot as main features, categorical kept for plotting)
3. Perform a correlation matrix (output in console)
4. Drop unneeded features (based on correlation matrix threshold)
5. Plot the graphs (feature and target distributions)
6. Train Decision Tree model
7. Predict on test data & evaluate performance
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier


# ==============================================================================
# Configuration
# ==============================================================================
DEBUG = False
SAMPLE_NROWS = 10000 if DEBUG else None

TRAIN_DATA_PATH = Path(__file__).parent / "GUIDE_Train.csv"
TEST_DATA_PATH = Path(__file__).parent / "GUIDE_Test.csv"
DATA_PATH = TRAIN_DATA_PATH  # Alias for backward compatibility

# Threshold for dropping multicollinear features in Step 4
CORRELATION_THRESHOLD = 0.80

# 24-hour rolling window for alert entity frequencies
TIME_WINDOW = "24h"

# Feature sets
CATEGORICAL_COLUMNS = ["Category", "EntityType"]
FREQUENCY_COLUMNS = [
    "DetectorId",
    "AccountSid",
    "DeviceId",
    "IpAddress",
    "ApplicationId",
]
KEEP_COLUMNS = [
    "Timestamp",        # For engineering 'TimeLastAlert', 'HourOfDay', 'DayOfWeek', 'IsWeekend'
    "Category",         # Alert category (categorical)
    "MitreTechniques",  # MITRE ATT&CK techniques (binary 0/1 indicator)
    "EntityType",       # Entity type (categorical)
    "DetectorId",       # Unique detector identifier (rolling frequency + time diff)
    "AccountSid",       # Account identifier (rolling frequency)
    "DeviceId",         # Device identifier (rolling frequency)
    "IpAddress",        # IP address (rolling frequency)
    "ApplicationId",    # Application identifier (rolling frequency)
    "IncidentGrade",    # Customer-provided triage label (target)
]


# ==============================================================================
# 1. Load Dataset
# ==============================================================================
def load_dataset(data_path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Load raw dataset, drop missing target labels, and sort chronologically."""
    print(f"      Loading data from {data_path.name} (debug={DEBUG}, nrows={nrows})...")
    df = pd.read_csv(data_path, nrows=nrows)

    # Drop alerts with unannotated IncidentGrade if column is present
    if "IncidentGrade" in df.columns:
        df = df.dropna(subset=["IncidentGrade"]).copy()

    # Parse timestamps and sort chronologically
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df = df.sort_values("Timestamp").reset_index(drop=True)
    print(f"      Loaded {len(df):,} alerts.")
    return df


# ==============================================================================
# 2. Pre-process Hand-Picked Features
# ==============================================================================
def preprocess_features(
    df: pd.DataFrame, is_train: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
    """
    Engineer hand-picked features.
    
    Returns:
        X: One-hot encoded feature matrix (main feature set for ML modeling).
        X_cat: Feature matrix retaining raw categorical columns (for plotting).
        y: Target labels (IncidentGrade), or None if not present.
    """
    phase_label = "train" if is_train else "test"
    print(f"      Pre-processing {phase_label} features...")
    cols_to_use = [col for col in KEEP_COLUMNS if col in df.columns]
    df_feat = df[cols_to_use].copy()

    # Binary mapping: 1 if MITRE ATT&CK techniques present, else 0
    if "MitreTechniques" in df_feat.columns:
        df_feat["MitreTechniques"] = df_feat["MitreTechniques"].notna().astype(int)

    # Temporal features (characterize alert timing relative to detector and business hours)
    df_feat["TimeLastAlert"] = df_feat.groupby("DetectorId")["Timestamp"].diff().dt.total_seconds()
    df_feat["HourOfDay"] = df_feat["Timestamp"].dt.hour.astype(int)
    df_feat["DayOfWeek"] = df_feat["Timestamp"].dt.dayofweek.astype(int)
    df_feat["IsWeekend"] = df_feat["Timestamp"].dt.dayofweek.isin([5, 6]).astype(int)

    # Entity frequency features (24h rolling count of alerts per entity)
    for column in FREQUENCY_COLUMNS:
        if column in df_feat.columns:
            rolling_counts = (
                df_feat[[column, "Timestamp"]]
                .assign(_alert=1)
                .groupby(column, dropna=False, sort=False)
                .rolling(window=TIME_WINDOW, on="Timestamp", closed="both")
                .sum()["_alert"]
                .reset_index(level=0, drop=True)
            )
            df_feat[f"{column}AlertFrequency"] = rolling_counts.fillna(0).astype("int64")

    # Drop first alert per detector (no prior alert -> TimeLastAlert is NaN)
    df_feat = df_feat.dropna(subset=["TimeLastAlert"]).reset_index(drop=True)

    # Extract target label if present
    y = df_feat.pop("IncidentGrade") if "IncidentGrade" in df_feat.columns else None

    # Drop raw timestamp and identifier columns
    raw_identifiers = ["Timestamp"] + [c for c in FREQUENCY_COLUMNS if c in df_feat.columns]
    X_cat = df_feat.drop(columns=raw_identifiers)

    # Main ML feature set: One-hot encode categorical features
    available_cats = [c for c in CATEGORICAL_COLUMNS if c in X_cat.columns]
    X = pd.get_dummies(X_cat, columns=available_cats, dtype=int)

    print(f"      Processed {len(X):,} {phase_label} samples.")
    print(f"      Categorical feature shape: {X_cat.shape}")
    print(f"      One-hot encoded shape:     {X.shape}")
    return X, X_cat, y


# ==============================================================================
# 3. Perform Correlation Matrix
# ==============================================================================
def compute_correlation_matrix(X: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Compute feature correlation matrix and display top correlated pairs in console."""
    print("\n[3/7] Computing correlation matrix...")
    corr_matrix = X.corr()

    # Extract unique feature pairs from upper triangle
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            pairs.append((col1, col2, corr_matrix.iloc[i, j]))

    # Sort by absolute correlation magnitude
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)

    print(f"\n--- Top {top_n} Correlated Feature Pairs ---")
    print(f"{'Feature 1':<35} {'Feature 2':<35} {'Correlation':>12}")
    print("-" * 84)
    for col1, col2, val in pairs[:top_n]:
        print(f"{col1:<35} {col2:<35} {val:>12.4f}")

    return corr_matrix


# ==============================================================================
# 4. Drop Unneeded Features (Based on Correlation Matrix)
# ==============================================================================
def drop_correlated_features(
    X_train: pd.DataFrame,
    X_train_cat: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Identify and drop redundant features exceeding the correlation threshold."""
    print(f"\n[4/7] Checking for features with |correlation| > {threshold:.2f}...")

    # Mask lower triangle and diagonal
    upper_tri = corr_matrix.abs().where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find columns to drop
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]

    if to_drop:
        print(f"      Found {len(to_drop)} redundant feature(s) to drop:")
        for col in to_drop:
            collinear_with = upper_tri.index[upper_tri[col] > threshold].tolist()
            print(f"      - Drop '{col}' (correlated with: {', '.join(collinear_with)})")

        X_train = X_train.drop(columns=to_drop)
        # Also drop from the categorical representation if the feature is present
        cat_drop = [col for col in to_drop if col in X_train_cat.columns]
        if cat_drop:
            X_train_cat = X_train_cat.drop(columns=cat_drop)
    else:
        print("      No features exceeded the correlation threshold.")

    print(f"      Updated one-hot feature shape: {X_train.shape}")
    return X_train, X_train_cat, to_drop


# ==============================================================================
# 5. Plot Graphs
# ==============================================================================
def plot_feature_distributions(X_cat: pd.DataFrame) -> None:
    """
    Plot feature distributions using the categorical dataset.
    Preserves categories as clean individual bar charts instead of one-hot explosion.
    """
    feature_columns = list(X_cat.columns)
    n_columns = 3
    n_rows = int(np.ceil(len(feature_columns) / n_columns))

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(5.5 * n_columns, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, feature_columns):
        values = X_cat[feature].dropna()

        if feature == "TimeLastAlert":
            # Bin wide-range seconds into intuitive operational recency buckets
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
            # Discrete / Categorical: bar graph showing frequency of each distinct value
            counts = values.value_counts().sort_index() if pd.api.types.is_numeric_dtype(values) else values.value_counts()
            ax.bar(range(len(counts)), counts.values, color="steelblue", edgecolor="black", alpha=0.8)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index.astype(str), rotation=45 if values.nunique() > 6 else 0, ha="right", fontsize=8)
            ax.set_title(feature)
            ax.set_xlabel("Value / Category")
            ax.set_ylabel("Count")

        else:
            # Continuous / High-cardinality counts: histogram
            sns.histplot(values, bins=25, ax=ax, color="steelblue", edgecolor="black", alpha=0.8)
            ax.set_title(feature)
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")

    # Hide unused subplot grids
    for ax in axes[len(feature_columns):]:
        ax.remove()

    fig.suptitle("Feature Distributions (Categorical & Engineered Features)", fontsize=16)
    fig.tight_layout(h_pad=3.0, w_pad=1.5)


def plot_target_distribution(y: pd.Series) -> None:
    """Plot target (IncidentGrade) class distribution with counts and percentages."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    target_counts = y.value_counts()
    bars = ax.bar(
        target_counts.index.astype(str),
        target_counts.values,
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
    )
    ax.set_title("Target Distribution: IncidentGrade", fontsize=13)
    ax.set_xlabel("Incident Grade")
    ax.set_ylabel("Count")

    # Add count and percentage labels above each bar
    total_samples = len(y)
    for bar in bars:
        height = bar.get_height()
        pct = (height / total_samples) * 100
        ax.annotate(
            f"{int(height):,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, max(target_counts.values) * 1.15)
    fig.tight_layout()


def plot_graphs(X_cat: pd.DataFrame, y: pd.Series) -> None:
    """[5/7] Render all feature and target distribution plots."""
    print("\n[5/7] Generating plots... (Close the plot window to continue)")
    plot_feature_distributions(X_cat)
    plot_target_distribution(y)
    plt.show()


# ==============================================================================
# 6. Train Decision Tree Model
# ==============================================================================
def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int | None = None,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Train a DecisionTreeClassifier on preprocessed features."""
    print("\n[6/7] Training Decision Tree Classifier...")
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    print(f"      Model training complete. Training Accuracy: {train_score:.4f}")

    # Display top 10 feature importances
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top_importances = importances.sort_values(ascending=False).head(10)
    print("\n--- Top 10 Feature Importances ---")
    for feat, imp in top_importances.items():
        print(f"{feat:<40} {imp:>8.4f}")

    return model


# ==============================================================================
# 7. Predict Test Data
# ==============================================================================
def predict_test_data(
    model: DecisionTreeClassifier,
    feature_columns: list[str],
    test_path: Path = TEST_DATA_PATH,
    nrows: int | None = SAMPLE_NROWS,
) -> tuple[np.ndarray, pd.Series | None]:
    """Load test dataset, apply identical preprocessing, align features, and generate predictions."""
    print(f"\n[7/7] Predicting on test data from {test_path.name}...")
    df_test = load_dataset(test_path, nrows=nrows)
    X_test, _, y_test = preprocess_features(df_test, is_train=False)

    # Align columns to match X_train exactly (same order, fill missing one-hot dummies with 0)
    X_test_aligned = X_test.reindex(columns=feature_columns, fill_value=0)

    # Generate predictions
    y_pred = model.predict(X_test_aligned)

    print("\n--- Test Predictions Distribution ---")
    pred_counts = pd.Series(y_pred).value_counts()
    for label, count in pred_counts.items():
        pct = (count / len(y_pred)) * 100
        print(f"{label:<20} {count:>8,} ({pct:>5.1f}%)")

    # Evaluate if ground truth labels are present
    if y_test is not None and not y_test.empty:
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        print("\n--- Test Evaluation Results ---")
        print(f"Macro-F1 Score: {macro_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    return y_pred, y_test


# ==============================================================================
# Main
# ==============================================================================
def main() -> None:
    # 1. Load dataset (train data)
    print("[1/7] Loading training dataset...")
    df_train = load_dataset(TRAIN_DATA_PATH, nrows=SAMPLE_NROWS)

    # 2. Pre-process hand-picked features
    print("\n[2/7] Pre-processing training features...")
    X_train, X_train_cat, y_train = preprocess_features(df_train, is_train=True)

    # 3. Perform correlation matrix (output in console)
    corr_matrix = compute_correlation_matrix(X_train)

    # 4. Drop unneeded features (based on correlation matrix)
    X_train, X_train_cat, dropped_features = drop_correlated_features(
        X_train, X_train_cat, corr_matrix, threshold=CORRELATION_THRESHOLD
    )

    # Summary inspection
    print("\n--- Final Training Dataset Summary ---")
    print(f"X_train (model features) shape: {X_train.shape}")
    print(f"X_train_cat (plot features) shape: {X_train_cat.shape}")
    print(f"y_train (target) count:\n{y_train.value_counts().to_string()}")

    # 5. Plot the graphs
    plot_graphs(X_train_cat, y_train)

    # 6. Train Decision Tree model
    model = train_decision_tree(X_train, y_train)

    # 7. Predict on test data & evaluate
    y_pred, y_test = predict_test_data(
        model=model,
        feature_columns=list(X_train.columns),
        test_path=TEST_DATA_PATH,
        nrows=SAMPLE_NROWS,
    )


if __name__ == "__main__":
    main()