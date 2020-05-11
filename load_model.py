import mlflow
import pandas as pd

x_test = pd.read_csv("data/test.csv", nrows=1, index_col="id")

# Load the model in `python_function` format
loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)

# Evaluate the model
test_predictions = loaded_model.predict(x_test)
print(test_predictions)
