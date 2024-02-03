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


def training_algo(df, trainingSet=1000, learningRate=0.01):
    m = len(df)
    theta = np.array([0.0, 0.0])
    for _ in range(trainingSet):
        df_estimate_price = theta[0] + (theta[1] * df["km"])
        df_sum = df_estimate_price - df["price"]
        # print(df_sum.values, " and ", df_sum.sum(), "and", sum(df_sum))
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
        
        # Calcul de la moyenne
        mean_x = 1 / len(df_params['km']) * df_params['km'].sum()
        print("Mean:", mean_x)

        # Calcul de l'écart-type
        standard_x = ((1 / len(df_params['km'])) * (df_params['km'].apply(lambda x: (x - mean_x)**2).sum()))**0.5
        print("Standard deviation:", standard_x)


        # Normalisation des données
        scaled_x = df_params['km'].apply(lambda x: (x - mean_x) / standard_x)
        print("Scaled values:", scaled_x)

        # Calcul de la moyenne
        mean_y = 1 / len(df_params['price']) * df_params['price'].sum()
        print("Mean:", mean_y)

        # Calcul de l'écart-type
        standard_y = ((1 / len(df_params['price'])) * (df_params['price'].apply(lambda y: (y - mean_y)**2).sum()))**0.5
        print("Standard deviation:", standard_y)

        # Normalisation des données
        scaled_y = df_params['price'].apply(lambda y: (y - mean_y) / standard_y)
        # print("Scaled values:", scaled_y)


        # min_val_x = df_params['km'].min()
        # max_val_x = df_params['km'].max()
        # min_val_y = df_params['price'].min()
        # max_val_y = df_params['price'].max()
        # scaled_values_x = (scaled_x - min_val_x) / (max_val_x - min_val_x)
        # scaled_values_y = (scaled_y - min_val_y) / (max_val_y - min_val_y)

        normalized_df = df_params
        # normalized_df["km"] = scaled_values_x
        # normalized_df["price"] = scaled_values_y
        min_val = df_params['km'].min()
        max_val = df_params['km'].max()
        min_price = df_params['price'].min()
        max_price = df_params['price'].max()

        # Appliquer la formule de mise à l'échelle Min-Max pour normaliser les données
        scaled_values = (df_params['km'] - min_val) / (max_val - min_val)

        normalized_df["km"] = scaled_values
        normalized_df["price"] = (df_params['price'] - min_price) / (max_price - min_price)

        print(normalized_df)
        theta = np.array(training_algo(normalized_df))


        denormalized_theta = theta
        # denormalized_theta = np.zeros(2)

        mean_y = df_params['price'].mean()
        standard_y = df_params['price'].std()
        # Désnormalisation de theta0
        # denormalized_theta[0] = denormalized_theta[0] * standard_y + mean_y

        # Désnormalisation de theta1
        # denormalized_theta[1] = denormalized_theta[1] * standard_y / standard_x

        
        predictvalue = denormalized_theta[0] + (denormalized_theta[1] * 0.93)
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
