from category_encoders.james_stein import JamesSteinEncoder
from pandas import read_csv, Series
from scipy.stats import boxcox


def tukey_outlier_test(pd_series):
    Q1 = round(pd_series.shape[0] / 4)
    Q3 = Q1 * 3
    pd_series_sorted = pd_series.sort_values()
    IQR = pd_series_sorted.iloc[Q3] - pd_series_sorted.iloc[Q1]
    lb = pd_series_sorted.iloc[Q1] - 1.5 * IQR
    ub = pd_series_sorted.iloc[Q3] + 1.5 * IQR
    outliers = pd_series_sorted[(lb > pd_series_sorted) | (pd_series_sorted > ub)].index
    return outliers


def fit_transform_normalize_y(y_train):
    y_train_bx, _ = boxcox(y_train)
    y_train_bx = Series(y_train_bx, y_train.index)
    outliers = tukey_outlier_test(y_train_bx)
    y_train_outliers_removed = y_train.drop(outliers)
    y_train_bx, bx_lambda = boxcox(y_train_outliers_removed)
    y_train_bx = Series(y_train_bx, y_train_outliers_removed.index)
    y_train_bx_whitened = (y_train_bx - y_train_bx.mean()) / y_train_bx.std()
    return y_train_bx_whitened, bx_lambda


def fit_X_encoder():
    X_train = read_csv("data/X_train.csv", index_col="id")
    X_cat_features = X_train.dtypes[X_train.dtypes == 'object'].index
    y_train = read_csv("data/y_train.csv", index_col="id").squeeze()
    y_train_normalized, _ = fit_transform_normalize_y(y_train)
    js_encoder = JamesSteinEncoder(cols=list(X_cat_features), randomized=True, sigma=0.0001)
    X_train_outliers_removed = X_train.loc[y_train_normalized.index]
    X_train_encoded = js_encoder.fit_transform(X_train_outliers_removed, y_train_normalized)
    X_train_encoded.to_csv("data/X_train_encoded.csv")
    return js_encoder


def transform_X(partition_name, js_encoder):
    X = read_csv("data/X_"+partition_name+".csv", index_col="id")
    X_encoded = js_encoder.transform(X)
    X_encoded.to_csv("data/X_"+partition_name+"_encoded.csv")


def main():
    js_encoder = fit_X_encoder()
    for partition in ["test", "holdout", "blind"]:
        transform_X(partition, js_encoder)


if __name__ == "__main__":
    main()
