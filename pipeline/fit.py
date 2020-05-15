from copy import deepcopy
import json

from pandas import read_csv
from xgboost import XGBRegressor


def score_gbm_configuration(data, fixed_gbm_params, variable_gbm_params):
    gbm = fit_gbm(data, fixed_gbm_params, variable_gbm_params)
    results = gbm.evals_result()
    return min(results['validation_0']['mae'])


def fit_gbm(data, fixed_gbm_params, variable_gbm_params):

    gbm_parameters = deepcopy(variable_gbm_params)
    gbm_parameters.update(fixed_gbm_params)
    gbm = XGBRegressor(objective="reg:gamma")
    gbm.set_params(**gbm_parameters)
    gbm.fit(
        data["X_train_encoded"],
        data["y_train"],
        early_stopping_rounds=30,
        eval_metric="mae",
        eval_set=[
            (data["X_test_encoded"], data["y_test"]),
            (data["X_holdout_encoded"], data["y_holdout"])
        ],
    )
    return gbm


def main(variable_gbm_params=None):

    data = {
        data_name: read_csv("data/" + data_name + ".csv", index_col="id")
        for data_name in [
            "X_train_encoded",
            "y_train",
            "X_test_encoded",
            "y_test",
            "X_holdout_encoded",
            "y_holdout",
            "X_blind_encoded",
            "y_blind",
        ]
    }
    if variable_gbm_params is None:
        with open("params/benchmark_gbm_params.json", "r") as f:
            variable_gbm_params = json.load(f)
    with open("params/fixed_gbm_params.json", "r") as f:
        fixed_gbm_params = json.load(f)
    data["y_train"] = data["y_train"].loc[data["X_train_encoded"].index]

    data["X_train_encoded"] = data["X_train_encoded"].append(data["X_test_encoded"])
    data["y_train"] = data["y_train"].append(data["y_test"])

    data["X_test_encoded"] = data["X_holdout_encoded"]
    data["y_test"] = data["y_holdout"]

    data["X_holdout_encoded"] = data["X_blind_encoded"]
    data["y_holdout"] = data["y_blind"]

    del data["X_blind_encoded"]
    del data["y_blind"]

    gbm = fit_gbm(data, fixed_gbm_params, variable_gbm_params)

    return gbm

if __name__ == "__main__":
    main()
