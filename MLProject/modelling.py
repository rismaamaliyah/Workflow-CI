import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# load dataset
train_df = pd.read_csv("MLProject/adult_income_dataset_preprocessing.csv")
X_train = train_df.drop(columns=["income"])
y_train = train_df["income"]

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# log metric
acc = accuracy_score(y_train, model.predict(X_train))
mlflow.log_metric("accuracy", acc)

# save model
joblib.dump(model, "adult_income_model.pkl")
mlflow.log_artifact("adult_income_model.pkl", artifact_path="models")