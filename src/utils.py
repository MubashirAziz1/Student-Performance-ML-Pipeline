import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

import mlflow
import mlflow.sklearn


from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train,X_test,y_test,models,params):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]

            with mlflow.start_run(run_name=model_name) as run:

                gs = GridSearchCV(model,para,cv=3)
                gs.fit(X_train,y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)
                
                mlflow.log_param("model_type", model_name)
                mlflow.log_params(gs.best_params_)  
                mlflow.log_metric("train_r2_score", train_model_score)  
                mlflow.log_metric("test_r2_score", test_model_score)

                mlflow.sklearn.log_model(model, "model",
                                        input_example=X_test[:5])

                print(f"{model_name} → Run ID: {run.info.run_id} | "
                      f"Test R2: {test_model_score:.4f}")


            report[list(models.keys())[i]] = test_model_score


        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)