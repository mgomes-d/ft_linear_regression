import numpy as np
import pandas as pd

def load_file(path: str) -> pd.DataFrame:
    try:
        assert path.lower().endswith(".csv"), "Path needs to end with .csv"
        df = pd.read_csv(path)
        return df
    except FileNotFoundError as msg:
        print(msg)
        return None
    except pd.errors.EmptyDataError as msg:
        print(msg)
        return None
    except AssertionError as msg:
        print(msg)
        return None

def calculate_rmse(df, theta):
    # Calcul des valeurs prédites
    predicted_values = theta["theta0"].values.astype(float) + theta["theta1"].values.astype(float) * df["km"]

    # Calcul du RMSE(root-mean-square-error)
    rmse = np.sqrt(((predicted_values - df["price"])**2).mean())
    return rmse

def main():
    theta = load_file("parameters.csv")
    df = load_file("data.csv")

    if theta is not None and df is not None:
        rmse = calculate_rmse(df, theta)
        accuracy = 1 - (rmse / (df["price"].max() - df["price"].min()))
        print("Accuracy Percentage: {:.0f}%".format(accuracy * 100))
    else:
        print("Error loading files.")

if __name__ == "__main__":
    main()
