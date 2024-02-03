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

def save_theta(theta):
    try:
        #1 column et -1 permet de faire automatiquement
        theta_df = pd.DataFrame(theta.reshape(1, -1), columns=['theta0', 'theta1'])
        theta_df.to_csv("parameters.csv", index=False)
    except Exception as e:
        print("Error occurred while saving theta values:", e)

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
        
        mean_x = 1 / len(df_params['km']) * df_params['km'].sum()

        standard_x = ((1 / len(df_params['km'])) * (df_params['km'].apply(lambda x: (x - mean_x)**2).sum()))**0.5

        scaled_x = df_params['km'].apply(lambda x: (x - mean_x) / standard_x)

        mean_y = 1 / len(df_params['price']) * df_params['price'].sum()

        standard_y = ((1 / len(df_params['price'])) * (df_params['price'].apply(lambda y: (y - mean_y)**2).sum()))**0.5

        scaled_y = df_params['price'].apply(lambda y: (y - mean_y) / standard_y)

        normalized_df = df_params.copy()
        normalized_df["km"] = scaled_x
        normalized_df["price"] = scaled_y

        theta = np.array(training_algo(normalized_df))

        denormalized_theta = theta * standard_y / standard_x
        denormalized_theta[0] = denormalized_theta[0] - denormalized_theta[1] * mean_x + mean_y
        save_theta(denormalized_theta)
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
