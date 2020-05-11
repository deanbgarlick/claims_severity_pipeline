from functools import partial
import json

from ax import Data, Metric, Models, Objective, OptimizationConfig, Runner
from ax.service.utils.instantiation import make_experiment
from ax.models.torch.botorch_defaults import get_NEI
from pandas import read_csv, DataFrame
from torch.cuda import is_available as torch_is_available
from torch import device

from .fit import score_gbm_configuration


if torch_is_available():
    DEVICE = device("cuda")
else:
    DEVICE = "cpu"


class SimpleRunner(Runner):
    __name__ = "ax_simple_runner"

    def run(self, trial):
        return {"name": str(trial.index)}


def construct_ax_metric(evaluation_func, name):

    class ResultsCachingAxMetric(Metric):

        def __init__(self, name):
            super().__init__(name=name, lower_is_better=True)
            self.trial_cache = {}

        def fetch_trial_data(self, trial):
            records = []
            if str(trial.index) not in self.trial_cache.keys():
                self.trial_cache[str(trial.index)] = {}
            for arm_name, arm in trial.arms_by_name.items():
                if arm_name not in self.trial_cache[str(trial.index)].keys():
                    params = arm.parameters
                    record = {
                        "arm_name": arm_name,
                        "metric_name": self.name,
                        "mean": evaluation_func(params),
                        "sem": 0.0,
                        "trial_index": trial.index,
                    }
                    self.trial_cache[str(trial.index)][str(arm_name)] = record
                else:
                    record = self.trial_cache[str(trial.index)][str(arm_name)]
                records.append(record)
            return Data(df=DataFrame.from_records(records))

    return ResultsCachingAxMetric(name)


def setup_ax_experiment_optimizer(data, ax_search_domain, fixed_gbm_params, control_arm_params):

    ax_experiment = make_experiment(
        name="GBM param optimisation",
        parameters=ax_search_domain,
        minimize=True,
        status_quo=control_arm_params,
    )

    ax_trial_evaluation_func = partial(
        score_gbm_configuration,
        data,
        fixed_gbm_params
    )

    gbm_mae = construct_ax_metric(ax_trial_evaluation_func, 'gbm_mae')

    ax_experiment.optimization_config = OptimizationConfig(
        objective=Objective(metric=gbm_mae, minimize=True,),
    )

    ax_experiment.runner = SimpleRunner()

    return ax_experiment


def find_best_parameters(data, ax_search_domain, control_arm_params, fixed_gbm_params):

    ax_experiment = setup_ax_experiment_optimizer(data, ax_search_domain, control_arm_params, fixed_gbm_params)

    sobol = Models.SOBOL(
        search_space=ax_experiment.search_space, device=DEVICE
    )

    batch_size = 2
    for _ in range(0, batch_size):
        generator_run = sobol.gen(1)
        ax_experiment.new_batch_trial(generator_run=generator_run)
        ax_experiment.trials[len(ax_experiment.trials)-1].run()

    n_iter = 1
    for iter_id in range(0, n_iter):
        data = ax_experiment.fetch_data()
        botorch_model = Models.BOTORCH(
            experiment=ax_experiment,
            data=data,
            acqf_constructor=get_NEI,
            device=DEVICE,
        )
        generator_run = botorch_model.gen(1)
        ax_experiment.new_batch_trial(generator_run=generator_run)
        ax_experiment.trials[len(ax_experiment.trials)-1].run()

    ax_experiment_data = ax_experiment.fetch_data().df
    best_score = ax_experiment_data.min()['mean']
    best_arm_name = ax_experiment_data.loc[ax_experiment_data['mean']==best_score].arm_name.item()
    best_arm = ax_experiment.arms_by_name[best_arm_name]
    best_params = best_arm.parameters

    return best_params, best_score


def main():

    data_ = {
        data_name: read_csv("data/" + data_name + ".csv", index_col="id")
        for data_name in [
            "X_train_encoded",
            "y_train",
            "X_test_encoded",
            "y_test",
            "X_holdout_encoded",
            "y_holdout"
        ]
    }
    data_["y_train"] = data_["y_train"].loc[data_["X_train_encoded"].index]

    with open('params/variable_param_search_domain.json', 'r') as f:
        ax_search_domain_ = json.load(f)
    with open('params/benchmark_gbm_params.json', 'r') as f:
        control_arm_params_ = json.load(f)
    with open('params/fixed_gbm_params.json', 'r') as f:
        fixed_gbm_params_ = json.load(f)

    best_params, best_score = find_best_parameters(data_, ax_search_domain_, fixed_gbm_params_, control_arm_params_)

    return best_params, best_score


if __name__ == "__main__":
    main()
