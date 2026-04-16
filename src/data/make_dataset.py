
import os

import pandas as pd

from sklearn.model_selection import train_test_split



RANDOM_STATE = 42

TEST_SIZE = 0.2



INPUT_PATH = "data/raw_data/raw.csv"

OUTPUT_DIR = "data/processed_data"



def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)



    df = pd.read_csv(INPUT_PATH)



    # On enlève la colonne date pour garder uniquement les variables utiles

    if "date" in df.columns:

        df = df.drop(columns=["date"])



    target_col = "silica_concentrate"



    X = df.drop(columns=[target_col])

    y = df[target_col]



    X_train, X_test, y_train, y_test = train_test_split(

        X, y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE

    )



    X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)

    X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)

    y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)

    y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)



    print("Split terminé.")

    print(f"X_train: {X_train.shape}")

    print(f"X_test: {X_test.shape}")

    print(f"y_train: {y_train.shape}")

    print(f"y_test: {y_test.shape}")



if __name__ == "__main__":

    main()

