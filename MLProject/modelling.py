import dagshub
import mlflow
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

dagshub.init(repo_owner="rismaamaliyah", repo_name="Eksperimen_SML_Risma", mlflow=True)
mlflow.set_experiment("Adult Income Prediction")

train_df = pd.read_csv("Membangun_model/adult_income_dataset_preprocessing/adult_income_dataset_preprocessing.csv")
test_df = pd.read_csv("Membangun_model/adult_income_dataset_preprocessing/adult_income_dataset_preprocessing_test.csv")

X_train = train_df.drop(columns=["income"])
y_train = train_df["income"]

X_test = test_df.drop(columns=["income"])
y_test = test_df["income"]

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    
    # Log parameter & metrics
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)
    
    # Save the model
    joblib.dump(model, "adul_income_model.pkl")
    mlflow.log_artifact("adul_income_model.pkl", artifact_path="models")
    
    # Log preprocessing data
    mlflow.log_artifact("Membangun_model/adult_income_dataset_preprocessing/adult_income_dataset_preprocessing.csv", artifact_path="data")
    
    # Addition artifact logs
    mlflow.log_artifact("Membangun_model/screenshoot_dashboard.jpg", artifact_path="screenshoot")
    mlflow.log_artifact("Membangun_model/screenshoot_artifact.jpg", artifact_path="screenshoot")
    mlflow.log_artifact("Membangun_model/screenshoot_artifact_2.jpg", artifact_path="screenshoot")
    mlflow.log_artifact("Membangun_model/screenshoot_artifact_charts.jpg", artifact_path="screenshoot")