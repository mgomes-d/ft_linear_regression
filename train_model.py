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

class LinearRegression:
    def __init__(self, df):
        self.df = self.standarisation(df)
        self.theta = np.array([0.0, 0.0])

    def standarisation(self, df):
        mean_x = (1 / df["km"].len()) * df["km"].sum()
        self.standard_x = (df["km"].apply(lambda x: (x - mean_x) **0.5).sum() / df["km"].len())
        mean_y = (1 / df["price"].len()) * df["price"].sum()
        self.standard_y = (df["price"].apply(lambda x: (x - mean_x) **0.5).sum() / df["price"].len())
        std_df = df.copy()
        std_df["km"].apply(lambda x: (x - mean_x) / self.standard_x)
        std_df["price"].apply(lambda y: (y - mean_y) / self.standard_y)
        return std_df
    def training(self, training_set=1000, learning_rate=0.1):
        m = len(self.df)
        theta = np.array([0.0, 0.0])
        for _ in range(training_set):
            estimate_price = theta[0] + (theta[1] * self.df["km"])
            price_diff = estimate_price - df["price"]
            gradient_theta0 = (1 / m) * price_diff.sum()
            tmp_theta0 = learning_rate * gradient_theta0
            gradient_theta1 = (1 / m) * (price_diff * self.df["km"]).sum()
            tmp_theta1 = learning_rate * gradient_theta1
            theta -= [float(tmp_theta0), float(tmp_theta1)]
        self.theta = theta
        self.save_theta(theta)
    def 
    def save_theta(self, theta):
        try:
            theta_df = pd.DataFrame(theta.reshape(1, -1), columns=['theta0', 'theta1'])
            theta_df.to_csv("parameters.csv", index=False)
        except Exception as e:
            print("Error occurred while saving theta values:", e)
    def show_graph(self, df):
        plt.scatter(df["km"], df["price"])
        
        plt.xlabel("Kilometers")
        plt.ylabel("Price")
        plt.title("Linear Regression")


def training_algo(df, trainingSet=1000, learningRate=0.1):
    m = len(df)
    theta = np.array([0.0, 0.0])
    for _ in range(trainingSet):
        df_estimate_price = theta[0] + (theta[1] * df["km"])
        df_sum = df_estimate_price - df["price"]
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
