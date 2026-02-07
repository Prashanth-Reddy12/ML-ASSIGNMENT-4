# ---------------------------------------------------------
# LAB 04 : kNN Classification Experiments
# Subject Code : 22AIE213
# Dataset      : All-India-Production.csv
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


# -------------------- A1 FUNCTIONS -----------------------

def train_knn(X_train, X_test, y_train, y_test, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    return train_pred, test_pred

def classification_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return cm, precision, recall, f1


# -------------------- A3 FUNCTIONS -----------------------

def generate_training_points():
    np.random.seed(10)
    X = np.random.randint(1, 11, size=(20, 2))
    y = np.array([0 if (x[0] + x[1]) < 12 else 1 for x in X])
    return X, y

def plot_training_points(X, y):
    colors = ['blue' if label == 0 else 'red' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.xlabel("X Feature")
    plt.ylabel("Y Feature")
    plt.title("Synthetic Training Data")
    plt.show()


# -------------------- A4 & A5 FUNCTIONS ------------------

def generate_test_points():
    x = np.arange(0, 10, 0.1)
    y = np.arange(0, 10, 0.1)
    gx, gy = np.meshgrid(x, y)
    return np.c_[gx.ravel(), gy.ravel()]

def classify_and_plot(X_train, y_train, test_points, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    preds = model.predict(test_points)

    colors = ['blue' if p == 0 else 'red' for p in preds]
    plt.scatter(test_points[:, 0], test_points[:, 1], c=colors, s=1)
    plt.title(f"kNN Decision Boundary (k = {k})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


# -------------------- A6 FUNCTIONS -----------------------

def load_dataset(path):
    data = pd.read_csv(path)
    data = data.dropna()

    # Two numeric feature columns from CSV
    X = data[['Production-2022-23', 'Production-2023-24']]

    # Binary target using median (classification requirement)
    y = (
        data['Production-2024-25']
        > data['Production-2024-25'].median()
    ).astype(int)

    return X, y


# -------------------- A7 FUNCTIONS -----------------------

def find_best_k(X_train, y_train):
    param_grid = {'n_neighbors': range(1, 21)}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    return grid.best_params_, grid.best_score_


# ==================== MAIN PROGRAM ======================

# -------- A6 : Load dataset --------
X, y = load_dataset(
    "C:/amrita/sem4/ml/ML_Project_Datasets/All-India-Production.csv"
)

# Train–Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- A1 : Confusion Matrix & Metrics --------
train_pred, test_pred = train_knn(X_train, X_test, y_train, y_test, k=3)

train_cm, train_prec, train_rec, train_f1 = classification_metrics(
    y_train, train_pred
)
test_cm, test_prec, test_rec, test_f1 = classification_metrics(
    y_test, test_pred
)

print("Training Confusion Matrix:\n", train_cm)
print("Training Precision:", train_prec)
print("Training Recall:", train_rec)
print("Training F1 Score:", train_f1)

print("\nTest Confusion Matrix:\n", test_cm)
print("Test Precision:", test_prec)
print("Test Recall:", test_rec)
print("Test F1 Score:", test_f1)


# -------- A3 : Synthetic Data --------
X_syn, y_syn = generate_training_points()
plot_training_points(X_syn, y_syn)


# -------- A4 & A5 : Decision Boundaries --------
test_points = generate_test_points()
for k in [1, 3, 5, 7]:
    classify_and_plot(X_syn, y_syn, test_points, k)


# -------- A7 : Hyperparameter Tuning --------
best_k, best_score = find_best_k(X_train, y_train)
print("\nBest k value:", best_k)
print("Best Cross-Validation Score:", best_score)
