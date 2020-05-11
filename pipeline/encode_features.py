from category_encoders.james_stein import JamesSteinEncoder
from pandas import read_csv
from scipy.stats import boxcox


def tukey_outlier_test(pd_series):
    Q1 = round(pd_series.shape[0] / 4)
    Q3 = Q1 * 3
    pd_series_sorted = pd_series.sort_values()
    IQR = pd_series_sorted[Q3] - pd_series_sorted[Q1]
    lb = pd_series_sorted[Q1] - 1.5 * IQR
    ub = pd_series_sorted[Q3] + 1.5 * IQR
    outliers = pd_series_sorted[(lb > pd_series_sorted) | (pd_series_sorted > ub)].index
    return outliers


def normalize_y():
    y_train = read_csv("data/y_train.csv").squeeze()
    y_train_bx, _ = boxcox(y_train)
    outliers = tukey_outlier_test(y_train_bx)
    y_train_outliers_removed = y_train.drop(outliers)
    y_train_bx, bx_lambda = boxcox(y_train_outliers_removed)
    y_train_bx_whitened = (y_train_bx - y_train_bx.mean()) / y_train_bx.std()
    return y_train_bx_whitened, bx_lambda
