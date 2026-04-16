
import os

import pickle

import pandas as pd

from sklearn.linear_model import Ridge



INPUT_DIR = "data/processed_data"

BEST_PARAMS_PATH = "models/best_params.pkl"

MODEL_PATH = "models/model.pkl"



def main():

    os.makedirs("models", exist_ok=True)



    X_train_scaled = pd.read_csv(f"{INPUT_DIR}/X_train_scaled.csv")

    y_train = pd.read_csv(f"{INPUT_DIR}/y_train.csv").squeeze()



    with open(BEST_PARAMS_PATH, "rb") as f:

        best_params = pickle.load(f)



    model = Ridge(**best_params)

    model.fit(X_train_scaled, y_train)



    with open(MODEL_PATH, "wb") as f:

        pickle.dump(model, f)



    print("Entraînement terminé.")

    print(f"Modèle sauvegardé dans : {MODEL_PATH}")



if __name__ == "__main__":

    main()

