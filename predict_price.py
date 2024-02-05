import pandas as pd


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

def main():
    df = load_file("parameters.csv")
    if df is not None:
        try:
            value = input("give a certain mileage:\n")
            predict = df["theta0"].values.astype(float) + (df["theta1"].values.astype(float) * float(value))
            print("the predict =",predict[0])
        except Exception as msg:
            print(msg)


if __name__ == "__main__":
    main()
