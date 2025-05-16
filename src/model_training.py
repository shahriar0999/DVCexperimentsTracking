import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import dvclive
from dvclive import Live
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("./data/iris.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=2, stratify=y)

# Hyperparameters
n_neighbors = 3
weights = 'distance'
algorithm = 'kd_tree'

#apply KNN
knn = KNeighborsClassifier(n_neighbors = n_neighbors, weights= weights ,algorithm=algorithm)


# Make predictions
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred_proba = knn.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')


print(f"Accuracy: {accuracy:.4f}")


with Live(save_dvc_exp=True) as live:
    # Log scalar metrics
    live.log_metric("accuracy", accuracy)
    live.log_metric("roc_auc", roc_auc)
    live.log_metric("precision", precision)
    live.log_metric("recall", recall)
    live.log_metric("f1", f1)
    
    live.log_param("n_estimators", n_estimators)
    live.log_param("max_depth", max_depth)
