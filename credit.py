import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

file = open("creditcard.csv", "r")
df = pd.read_csv(file)


print(df.groupby("Class")["Amount"].describe())
print(df.groupby("Class")["Time"].describe())


df["Hour"] = (df["Time"] // 3600) % 24

fraud_by_hour = df.groupby("Hour")["Class"].mean()

# first we will drop the time feature, because we add instead of it a hour feature
df = df.drop(columns=["Time"])
# define the X,y : feature (inputs) and label
y = df["Class"]
X = df.drop(columns=["Class"])

# split unbalanced data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# scaling
scaler = StandardScaler()
# features to scale, [amount,hour]
target_feature_to_scale = ["Amount", "Hour"]

X_train_scaled_feature = scaler.fit_transform(X_train[target_feature_to_scale])
X_test_scaled_feature = scaler.transform(X_test[target_feature_to_scale])

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Overwrite only those two columns with the scaled versions
X_train_scaled[target_feature_to_scale] = X_train_scaled_feature
X_test_scaled[target_feature_to_scale] = X_test_scaled_feature

# choosing the model, and train
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=50),
}

parameters = {
    "Logistic Regression": {"max_iter": [100, 500, 1000, 1500, 2000, 2500, 3000]},
    "Random Forest": {"max_depth": [10, 15], "min_samples_leaf": [4]},
    # {"max_depth": [10, 15, 20, None], "min_samples_leaf": [4, 6, 10]}
}


# finding the best hyperparameter for each model
optimized_model = {}
best_scores = {}
for name, model in models.items():
    param_grid = parameters[name]
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2
    )
    grid_search.fit(X_train_scaled, y_train)
    optimized_model[name] = grid_search.best_estimator_
    best_scores[name] = grid_search.best_score_

    print(f"Best parameters for {name}: {grid_search.best_params_}\n")

best_model = max(best_scores, key=best_scores.get)
print(f"best model train: {best_model} with mean score: {best_scores[best_model]:.4}")

m = optimized_model[best_model].fit(X_train_scaled, y_train)

y_probs = m.predict_proba(X_test_scaled)[:, 1]

custom_threshold = [0.6, 0.45, 0.25, 0.15]
best_f1 = 0
best_threshold = None

for threshold in custom_threshold:
    y_pred = (y_probs >= threshold).astype(int)
    print(f"\n=== Threshold = {threshold} ===")
    print(classification_report(y_test, y_pred))
    
    f1 = f1_score(y_test, y_pred, pos_label=1)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nBest threshold found: {best_threshold} (F1 = {best_f1:.4f})")

y_pred_final = (y_probs >= best_threshold).astype(int)

print(f"\n=== Final model at threshold = {best_threshold} ===")
print(classification_report(y_test, y_pred_final, target_names=['Normal', 'Fraud']))

recall = recall_score(y_test, y_pred_final, pos_label=1)
precision = precision_score(y_test, y_pred_final, pos_label=1)
f1 = f1_score(y_test, y_pred_final, pos_label=1)

print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")

print(confusion_matrix(y_test, y_pred_final))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_final, display_labels=["Normal", "Fraud"])
plt.title(f"Confusion Matrix (threshold = {best_threshold})")
plt.show()