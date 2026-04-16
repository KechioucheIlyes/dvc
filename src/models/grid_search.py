
import os

import pickle

import pandas as pd

from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV



INPUT_DIR = "data/processed_data"

OUTPUT_PATH = "models/best_params.pkl"



def main():

    os.makedirs("models", exist_ok=True)



    X_train_scaled = pd.read_csv(f"{INPUT_DIR}/X_train_scaled.csv")

    y_train = pd.read_csv(f"{INPUT_DIR}/y_train.csv").squeeze()



    model = Ridge()



    param_grid = {

        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],

        "solver": ["auto", "svd", "cholesky", "lsqr"]

    }



    grid_search = GridSearchCV(

        estimator=model,

        param_grid=param_grid,

        cv=5,

        scoring="neg_mean_squared_error",

        n_jobs=-1

    )



    grid_search.fit(X_train_scaled, y_train)



    best_params = grid_search.best_params_



    with open(OUTPUT_PATH, "wb") as f:

        pickle.dump(best_params, f)



    print("Grid search terminée.")

    print("Meilleurs paramètres :", best_params)



if __name__ == "__main__":

    main()

