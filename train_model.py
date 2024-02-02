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


def training_algo(df, trainingSet=1000, learningRate=0.0000000001):
    m = len(df)
    theta = np.array([0.0, 0.0])
    for _ in range(trainingSet):
        df_estimate_price = theta[0] + (theta[1] * df["km"])
        df_sum = df_estimate_price - df["price"]
        print(df_sum.values, " and ", df_sum.sum(), "and", sum(df_sum))
        gradient_theta0 = (1 / m) * df_sum.sum()
        tmptheta0 = learningRate * gradient_theta0
        gradient_theta1 = (1 / m) * (df_sum * df["km"]).sum()
        tmptheta1 = learningRate * gradient_theta1
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

        mileage = np.array(df_params['km'])
        price = np.array(df_params['price'])
        mileage_norm = (mileage - np.mean(mileage)) / np.std(mileage)

        normalized_df = df_params
        normalized_df["km"] = mileage_norm
        print(normalized_df)
        theta = np.array(training_algo(normalized_df))
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
    except AssertionError as msg:
        print(msg)


if __name__ == "__main__":
    main()
