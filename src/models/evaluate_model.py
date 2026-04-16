
import json

import pickle

import os

import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



INPUT_DIR = "data/processed_data"

MODEL_PATH = "models/model.pkl"

PREDICTIONS_PATH = "data/predictions.csv"

METRICS_PATH = "metrics/scores.json"



def main():

    os.makedirs("metrics", exist_ok=True)

    os.makedirs("data", exist_ok=True)



    X_test_scaled = pd.read_csv(f"{INPUT_DIR}/X_test_scaled.csv")

    y_test = pd.read_csv(f"{INPUT_DIR}/y_test.csv").squeeze()



    with open(MODEL_PATH, "rb") as f:

        model = pickle.load(f)



    predictions = model.predict(X_test_scaled)



    mse = mean_squared_error(y_test, predictions)

    rmse = mse ** 0.5

    mae = mean_absolute_error(y_test, predictions)

    r2 = r2_score(y_test, predictions)



    scores = {

        "mse": float(mse),

        "rmse": float(rmse),

        "mae": float(mae),

        "r2": float(r2)

    }



    predictions_df = pd.DataFrame({

        "y_true": y_test,

        "y_pred": predictions

    })

    predictions_df.to_csv(PREDICTIONS_PATH, index=False)



    with open(METRICS_PATH, "w") as f:

        json.dump(scores, f, indent=4)



    print("Évaluation terminée.")

    print("Scores :", scores)

    print(f"Prédictions sauvegardées dans : {PREDICTIONS_PATH}")

    print(f"Métriques sauvegardées dans : {METRICS_PATH}")



if __name__ == "__main__":

    main()

