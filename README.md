# Microsoft Security Incident Prediction

Machine learning project for the **CS-C3240 Machine Learning** course at Aalto University.

This project implements machine learning models (decision trees and baseline classifiers) to predict incident triage grades on the [Microsoft Security Incident Prediction (GUIDE)](https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction) dataset.

## Features & Preprocessing

- **Categorical Encoding:** Integer mapping for `Category` and `EntityType`.
- **Technique Detection:** Binary encoding of MITRE ATT&CK techniques (`MitreTechniques`).
- **Temporal Features:** Time difference from previous detector alert (`TimeLastAlert`), hour of day (`HourOfDay`), day of week (`DayOfWeek`), and weekend flag (`IsWeekend`).
- **Entity Operational Frequency:** 24-hour rolling alert counts across key entity identifiers (`DetectorId`, `AccountSid`, `DeviceId`, `IpAddress`, `ApplicationId`).
- **Evaluation:** Macro-F1 score across triage labels (`BenignPositive`, `FalsePositive`, `TruePositive`) with temporal train/validation splitting.

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Dataset

Download `GUIDE_Train.csv` and `GUIDE_Test.csv` from [Kaggle](https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction) and place them in the project root directory.

### Running

To run the data preprocessing and view feature distributions:

```bash
python main.py
```
