import pandas as pd


def load_file(path: str) -> pd.DataFrame:
    assert path.lower().endswith(".csv"), "Path need to end with .csv"
    df = pd.read_csv(path)
    return df

def main():
    try:
        df = load_file("parameters.csv")
        value = input("give a certain mileage:\n")
        predict = df["theta0"].values.astype(float) + (df["theta1"].values.astype(float) * float(value))
        print("Estimate price =",predict[0])
    except Exception as msg:
        print(msg)


if __name__ == "__main__":
    main()
