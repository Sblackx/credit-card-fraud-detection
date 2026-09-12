import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
)

file = open("creditcard.csv", "r")
df = pd.read_csv(file)

# print(df.shape)
# print(df.head())
# print(df['Class'].value_counts())

print(df.groupby("Class")["Amount"].describe())
print(df.groupby("Class")["Time"].describe())


# plt.figure(figsize=(8, 5))
# df.boxplot(column='Amount', by='Class')
# plt.title('Transaction Amount by Class (0=Normal, 1=Fraud)')
# plt.suptitle('')  # removes the default ugly auto-title pandas adds
# plt.xlabel('Class')
# plt.ylabel('Amount')
# plt.yscale('log')
# plt.show()


df["Hour"] = (df["Time"] // 3600) % 24

fraud_by_hour = df.groupby("Hour")["Class"].mean()

# from visuilizing and reading the data, we can see that most of
# fraud trans happens between 2 to 4 (am)
# plt.figure(figsize=(10, 5))
# fraud_by_hour.plot(kind='bar')
# plt.title('Fraud Rate by Hour of Day')
# plt.xlabel('Hour')
# plt.ylabel('Fraud Rate')
# plt.show()


# #compare each one of the classes visualization
# fraud = df[df['Class'] == 1]
# normal = df[df['Class'] == 0]

# fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

# fraud['Hour'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='red')
# axes[0].set_title('Fraud Transactions by Hour')
# axes[0].set_xlabel('Hour')
# axes[0].set_ylabel('Count')

# normal['Hour'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='blue')
# axes[1].set_title('Normal Transactions by Hour')
# axes[1].set_xlabel('Hour')
# axes[1].set_ylabel('Count')

# plt.tight_layout()
# plt.show()

# organize the data
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
print(f"best model train: {best_model} with accuracy {best_scores[best_model]:.4}")

# fiting the data
m = optimized_model[best_model].fit(X_train_scaled, y_train)

predict = m.predict(X_test_scaled)

# print("prediction: \n", predict)
# print("real: \n", y_test)

# imbalanced data,using accuracy will give us wrong probability!
print("-----------------\n")
print("accuracy:", accuracy_score(y_test, predict))
print("-----------------\n")

print(confusion_matrix(y_test, predict))
ConfusionMatrixDisplay.from_predictions(
    y_test, predict, display_labels=["Normal", "Fraud"]
)
plt.show()
