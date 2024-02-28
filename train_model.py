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
        self.df = self.__standarisation(df)
        self.theta = np.array([0.0, 0.0])

    def __standarisation(self, df):
        self.mean_x = (1 / df["km"].size) * df["km"].sum()
        self.standard_x = ((1 / len(df['km'])) * (df['km'].apply(lambda x: (x - self.mean_x)**2).sum()))**0.5
        self.mean_y = (1 / df["price"].size) * df["price"].sum()
        self.standard_y = ((1 / len(df['price'])) * (df['price'].apply(lambda y: (y - self.mean_y)**2).sum()))**0.5
        std_df = df.copy()
        std_df["km"] = (std_df["km"] - self.mean_x) / self.standard_x
        std_df["price"] = (std_df["price"] - self.mean_y) / self.standard_y
        return std_df

    def __destandarisation(self, theta):
        new_theta = theta * self.standard_y / self.standard_x
        new_theta[0] = new_theta[0] - new_theta[1] * self.mean_x + self.mean_y
        return new_theta

    def training(self, training_set=600, learning_rate=0.01):
        m = len(self.df)
        theta = np.array([0.0, 0.0])
        for _ in range(training_set):
            estimate_price = theta[0] + (theta[1] * self.df["km"])
            price_diff = estimate_price - self.df["price"]
            gradient_theta0 = (1 / m) * price_diff.sum()
            tmp_theta0 = learning_rate * gradient_theta0
            gradient_theta1 = (1 / m) * (price_diff * self.df["km"]).sum()
            tmp_theta1 = learning_rate * gradient_theta1
            theta[0] -= float(tmp_theta0)
            theta[1] -= float(tmp_theta1)
        theta = self.__destandarisation(theta)
        self.theta = theta
        self.save_theta(theta)

    def save_theta(self, theta):
        try:
            theta_df = pd.DataFrame(theta.reshape(1, -1), columns=['theta0', 'theta1'])
            theta_df.to_csv("parameters.csv", index=False)
        except Exception as e:
            print("Error occurred while saving theta values:", e)
    def show_graph(self, df):
        min_X = df["km"].min()
        max_X = df["km"].max()
        x_line = np.linspace(min_X, max_X, 100)
        y_line = self.theta[0] + (self.theta[1] * x_line)
        plt.scatter(df["km"], df["price"])
        plt.plot(x_line, y_line, color="red", label="Regression Line")
        
        plt.legend()
        plt.xlabel("Kilometers")
        plt.ylabel("Price")
        plt.title("Linear Regression")
        plt.show()

def main():
    try:
        df_params = load_file("data.csv")
        assert df_params is not None, "Error when load csv file"
        linear_regression = LinearRegression(df_params)
        linear_regression.training()
        linear_regression.show_graph(df_params)

    except AssertionError as msg:
        print(msg)

if __name__ == "__main__":
    main()
