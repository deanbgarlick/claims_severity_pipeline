from copy import deepcopy
import json

from pandas import read_csv
from xgboost import XGBRegressor


def fit_gbm(data, variable_gbm_params, fixed_gbm_params):
    gbm_parameters = deepcopy(variable_gbm_params)
    gbm_parameters.update(fixed_gbm_params)
    gbm = XGBRegressor(objective="reg:gamma")
    gbm.set_params(**gbm_parameters)
    gbm.fit(
        data["X_train_encoded"],
        data["y_train"],
        #early_stopping_rounds=30,
        #eval_metric="mae",
        #eval_set=[(data["X_test_encoded"], data["y_test"])],
    )

def main():
    data = {
        data_name: read_csv("data/" + data_name + ".csv", index_col="id")
        for data_name in [
            "X_train_encoded",
            "y_train",
            "X_test_encoded",
            "y_test",
            "X_holdout_encoded",
            "y_holdout",
        ]
    }
    with open('params/variable_gbm_params.json', 'r') as f:
        variable_gbm_params = json.load(f)
    with open('params/fixed_gbm_params.json', 'r') as f:
        fixed_gbm_params = json.load(f)
    data["y_train"] = data["y_train"].loc[data["X_train_encoded"].index]
    fit_gbm(data, {}, fixed_gbm_params)


if __name__ == "__main__":
    main()