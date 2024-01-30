import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_file(path: str) -> pd.DataFrame:
    try:
        assert path.lower().endswith(".csv"), "Path need to end with .csv"
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


def training_algo(df, trainingSet=1000, learningRate=0.0000000000001):
    m = len(df)
    theta = np.array([0.0, 0.0])
    for _ in range(trainingSet):
        df_estimate_price = theta[0] + theta[1] * df["km"]
        df_sum = df_estimate_price - df["price"]
        tmptheta0 = learningRate * ((1 / m) * df_sum.sum())
        tmptheta1 = learningRate * ((1 / m) * (df_sum * df["km"]).sum())
        theta[0] -= float(tmptheta0)
        theta[1] -= float(tmptheta1)
    return theta


def main():
    try:
        df_params = load_file("data.csv")
        assert df_params is not None, "Error when load csv file"
        print(df_params)

        min_vals = df_params["price"].min()
        max_vals = df_params["price"].max()

        # normalized_df = (df_params - min_vals) / (max_vals - min_vals)
        normalized_df = df_params
        print(normalized_df)
        theta = np.array(training_algo(normalized_df))
        # denormalized_theta = (theta * (max_vals - min_vals)) + min_vals
        denormalized_theta = theta
        predictvalue = denormalized_theta[0] + (denormalized_theta[1] * 240000)
        print(f"Theta values after training: {denormalized_theta} , predict = {predictvalue}")

        min_X = df_params['km'].min()
        max_X = df_params['km'].max()
        x_line = np.linspace(min_X, max_X, 100)
        y_line =  denormalized_theta[0] + denormalized_theta[1] * x_line

        plt.scatter(df_params["km"], df_params["price"])
        plt.plot(x_line, y_line, color='red')
        plt.xlabel("Kilometers")
        plt.ylabel("Price")
        plt.title("Linear Regression")
        plt.show()
        # print(f'km:\n{km}, price:\n{price}')
    except AssertionError as msg:
        print(msg)


if __name__ == "__main__":
    main()
