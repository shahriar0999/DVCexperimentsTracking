import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import dvclive
from dvclive import Live

df = pd.read_csv("./data/iris.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=2, stratify=y)

n_estimators=2
max_depth=2

# Initialize and train the Gradient Boosting model
model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
    


with Live(save_dvc_exp=True) as live:
    # Log scalar metrics
    live.log_metric("accuracy", accuracy)
    live.log_metric("classification_report", report)
    
    live.log_param("n_estimators", n_estimators)
    live.log_param("max_depth", max_depth)
