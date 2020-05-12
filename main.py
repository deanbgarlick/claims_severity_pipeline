import pickle

import click

import pipeline


@click.command(help="Perform hyperparameter search with Ax library.")
@click.option("--data_path", type=click.STRING, default="data/train.csv",
              help="Maximum number of runs to evaluate.")
@click.option("--n_sobol", type=click.INT, default=10,
              help="Maximum number of runs to evaluate.")
@click.option("--n_ei", type=click.INT, default=500,
              help="Number of epochs")
def main(data_path, n_sobol, n_ei):

    pipeline.partition_data.main(data_path)
    js_encoder = pipeline.encode_features.main()
    best_params, best_score = pipeline.optimize.main(n_sobol, n_ei)

    print("\n Best model achieved mae of: " + str(best_score) + "in training" + "\n")
    gbm = pipeline.fit.main(best_params)
    final_blind_score = gbm.evals_result()['validation_1']['mae'][-1]
    print("\n Final model built with blind score: " + str(final_blind_score) + "\n")

    with open("artifacts/gbm", "wb") as f:
        pickle.dump(gbm, f)
    with open("artifacts/js_encoder", "wb") as f:
        pickle.dump(js_encoder, f)

    pipeline.save_model.main("artifacts/gbm", "artifacts/js_encoder")


if __name__ == "__main__":
    main()
