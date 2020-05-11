from pandas import read_csv
from sklearn.model_selection import train_test_split


def partition_data_and_save_to_disk(X_data, y_data, X_save_name, y_save_name, save_size, random_state):
    X_partition_to_save, X_remaining,  y_partition_to_save, y_remaining = train_test_split(
        X_data, y_data, test_size=1-save_size, random_state=random_state
    )
    X_partition_to_save.to_csv(X_save_name)
    y_partition_to_save.to_csv(y_save_name, header=True)
    return X_remaining, y_remaining


def main():
    """Load training data from data directory. Separate features from response.
    Partition the data along its rows and the partitions to disk."""

    data = read_csv("data/train.csv", index_col="id")
    X = data[data.columns.drop("loss")]
    y = data["loss"]

    X_not_train, y_not_train = partition_data_and_save_to_disk(X, y, "data/X_train.csv", "data/y_train.csv", 0.6, 1)
    X_holdout_and_blind, y_holdout_and_blind = partition_data_and_save_to_disk(X_not_train, y_not_train, "data/X_test.csv", "data/y_test.csv", 0.5, 2)
    X_blind, y_blind = partition_data_and_save_to_disk(X_holdout_and_blind, y_holdout_and_blind, "data/X_holdout.csv", "data/y_holdout.csv", 0.5, 3)
    X_blind.to_csv("data/X_blind.csv")
    y_blind.to_csv("data/y_blind.csv", header=True)


if __name__ == "__main__":
    main()
