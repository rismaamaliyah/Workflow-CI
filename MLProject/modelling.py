import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--max_iter", type=int, default=100)
args = parser.parse_args()

mlflow.autolog(log_models=True)

train_df = pd.read_csv("adult_income_dataset_preprocessing.csv")
X_train = train_df.drop(columns=["income"])
y_train = train_df["income"]

model = LogisticRegression(max_iter=args.max_iter)
model.fit(X_train, y_train)
acc = accuracy_score(y_train, model.predict(X_train))
print(f"Training accuracy: {acc}")