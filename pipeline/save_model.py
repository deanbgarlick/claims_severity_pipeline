import category_encoders as ce
import mlflow.pyfunc
import xgboost as xgb
import cloudpickle


class XGBWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        #import category_encoders as ce
        import xgboost as xgb
        self.js_encoder = context.artifacts["js_encoder"]
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(context.artifacts["xgb_model"])

    def predict(self, context, model_input):
        input_matrix = xgb.DMatrix(model_input.values)
        return self.xgb_model.predict(input_matrix)


def main(xgb_model_path, js_encoder_model_path, mlflow_pyfunc_model_path="xgb_mlflow_pyfunc"):

    conda_env = {
        "channels": ["defaults", "conda-forge"],
        "dependencies": [
            "xgboost={}".format(xgb.__version__),
            "category_encoders={}".format(ce.__version__),
            'cloudpickle={}'.format(cloudpickle.__version__),
        ],
        "name": "claims_severity_model_env"
    }

    artifacts = {
        "xgb_model": xgb_model_path,
        "js_encoder": js_encoder_model_path
    }

    # Save the MLflow Model
    mlflow.pyfunc.save_model(
        path=mlflow_pyfunc_model_path,
        python_model=XGBWrapper(),
        artifacts=artifacts,
        conda_env=conda_env)


if __name__ == "__main__":
    main("artifacts/gbm", "artifacts/js_encoder")
