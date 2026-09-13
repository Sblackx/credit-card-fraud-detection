# Credit Card Fraud Detection

A classification project detecting fraudulent credit card transactions in a highly imbalanced dataset — and a deep dive into why accuracy is the wrong metric to trust when one class is rare.

## Dataset

The [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group) — 284,807 transactions made by European cardholders over two days, of which only **492 (0.17%) are fraudulent**. Features `V1`–`V28` are PCA-transformed and anonymized by the dataset provider for confidentiality; only `Time` and `Amount` retain their original meaning.

> Not included in this repo due to file size (~150MB) — download `creditcard.csv` from the link above and place it in the project folder before running.

## Exploratory Analysis

- **Fraud rate by hour of day:** fraud rate spikes sharply between **2–4 AM**, reaching roughly 10x the overall average fraud rate during those hours — likely reflecting lower transaction volume and less active monitoring during low-traffic overnight hours.
- This insight was used to engineer a new `Hour` feature (extracted from the raw `Time` column, which only represents seconds elapsed since the first transaction) as an additional model input.

## Approach

**1. Feature Engineering**
- Derived `Hour` (0–23) from the raw `Time` column
- Dropped `Time` after extracting `Hour`

**2. Train/Test Split**
- Used `stratify=y` — critical for this dataset, since a plain random split risks an uneven distribution of the very small fraud class between train and test

**3. Scaling**
- Applied `StandardScaler` only to `Amount` and `Hour` — the `V1`–`V28` features arrive already PCA-scaled by the dataset provider

**4. Model Comparison & Tuning**
- Compared `LogisticRegression` and `RandomForestClassifier`, each tuned with `GridSearchCV`
- **Best model: Random Forest** (`max_depth=15`, `min_samples_leaf=4`)

**5. The Accuracy Trap**
At the default 0.5 threshold, the model achieves **99.95% accuracy** — but this number is dangerously misleading. A model that predicted "not fraud" for every single transaction would still score ~99.8% accuracy while catching zero fraud. The real story is only visible through precision, recall, and F1-score for the fraud class specifically.

**6. Threshold Tuning**
Rather than accepting the default 0.5 classification threshold, tested multiple thresholds and measured the resulting precision/recall/F1 tradeoff for the fraud class:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.60 | 0.98 | 0.72 | 0.82 |
| **0.45** | **0.90** | **0.77** | **0.83** |
| 0.25 | 0.80 | 0.84 | 0.82 |
| 0.15 | 0.78 | 0.85 | 0.81 |

**Selected threshold: 0.45**, maximizing F1-score. Lower thresholds catch more fraud (higher recall) at the cost of more false alarms (lower precision); higher thresholds do the reverse. F1-optimization assumes both error types are equally costly — in a real production system, this threshold would likely be tuned lower, since missing real fraud is typically far more costly to a business than a false alarm. This project doesn't have real cost data to make that tradeoff precisely, so F1 is used as a principled, neutral default.

## Results (final model, threshold = 0.45)

```
              precision    recall  f1-score   support
      Normal       1.00      1.00      1.00     71079
       Fraud       0.90      0.77      0.83       123
```

Confusion matrix:
```
[[71069    10]
 [   28    95]]
```
<img width="636" height="551" alt="WhatsApp Image 2026-09-13 at 18 46 02" src="https://github.com/user-attachments/assets/01d885ee-b5c0-44bb-9247-2d42a568928a" />


Out of 123 real fraud cases in the test set, the model correctly identified **95** while missing **28**, with only **10 false alarms** among over 71,000 legitimate transactions.

## Tools

Python, pandas, scikit-learn (`LogisticRegression`, `RandomForestClassifier`, `GridSearchCV`, `StandardScaler`, `classification_report`), matplotlib

## Key Takeaways

- Accuracy is a misleading metric on imbalanced data — a model can score 99%+ while being functionally useless for the task it's meant to solve.
- `stratify=y` is essential when splitting data with a rare minority class, to avoid an unrepresentative test set.
- Adjusting the classification threshold (not just the model itself) is a direct, practical lever for controlling the precision-recall tradeoff.
- The "best" threshold depends on real business costs, which weren't available here — F1-optimization is a reasonable default, but a production system would tune this differently based on the actual cost of missed fraud vs. false alarms.

## How to Run

```bash
pip install pandas scikit-learn matplotlib
python fraud_detection.py
```
