
import os

import pandas as pd

import joblib

from sklearn.preprocessing import StandardScaler



INPUT_DIR = "data/processed_data"

OUTPUT_DIR = "data/processed_data"

SCALER_PATH = "models/scaler.pkl"



def main():

    os.makedirs("models", exist_ok=True)



    X_train = pd.read_csv(f"{INPUT_DIR}/X_train.csv")

    X_test = pd.read_csv(f"{INPUT_DIR}/X_test.csv")



    scaler = StandardScaler()



    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)



    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)



    X_train_scaled.to_csv(f"{OUTPUT_DIR}/X_train_scaled.csv", index=False)

    X_test_scaled.to_csv(f"{OUTPUT_DIR}/X_test_scaled.csv", index=False)



    joblib.dump(scaler, SCALER_PATH)



    print("Normalisation terminée.")

    print(f"Scaler sauvegardé dans : {SCALER_PATH}")



if __name__ == "__main__":

    main()

