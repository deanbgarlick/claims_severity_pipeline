from pandas import read_csv
from sklearn.model_selection import train_test_split


def main():
    """Load training daata from data directory. Separate features from response.
    Make row partitions of the data and save the partitions to disk."""

    data = read_csv("data/train.csv", index_col="id")
    X = data[data.columns.drop("loss")]
    y = data["loss"]

    X_train, X_not_train,  y_train, y_not_train = train_test_split(
        X, y, test_size=0.4, random_state=1
    )
    X_train.to_csv("data/X_train.csv")
    y_train.to_csv("data/y_train.csv", header=True)

    X_test, X_holdout_and_blind, y_test, y_holdout_and_blind = train_test_split(
        X_not_train, y_not_train, test_size=0.5, random_state=43
    )
    X_test.to_csv("data/X_test.csv")
    y_test.to_csv("data/y_test.csv", header=True)

    X_holdout, X_blind, y_holdout, y_blind = train_test_split(
        X_holdout_and_blind, y_holdout_and_blind, test_size=0.5, random_state=43
    )
    X_holdout.to_csv("data/X_holdout.csv")
    y_holdout.to_csv("data/y_holdout.csv", header=True)
    X_blind.to_csv("data/X_blind.csv")
    y_blind.to_csv("data/y_blind.csv", header=True)


if __name__ == "__main__":
    main()