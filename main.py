import pickle

import pipeline


def main():

    pipeline.partition_data.main()
    js_encoder = pipeline.encode_features.main()
    best_params, best_score = pipeline.optimize.main()

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
