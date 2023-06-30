import warnings
import sys
import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse

import runhouse as rh
import mlflow

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def fit_model_on_training_data(alpha, l1_ratio, train_x, train_y, random_state=42):
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    lr.fit(train_x, train_y)
    return lr


def split_data_into_training_and_test(table_obj):
    """Split the data into training and test sets. (0.75, 0.25) split."""
    # This code will run on a cluster, so let's load the data here
    data = table_obj.fetch()
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    return [train, test, train_x, train_y, test_x, test_y]


def make_test_predictions(lr, test_x):
    predicted_qualities = lr.predict(test_x)
    return predicted_qualities


def eval_model(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return [rmse, mae, r2]


def load_and_save_data():
    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        raise e

    # Save the data to a bucket in S3
    table_obj = rh.table(data=data, name="raw_data", system="s3").write().save()
    print(f"Saved table to s3 in path: {table_obj.path}")

    return table_obj


def run_training(alpha, l1_ratio):
    """Create a Run which splits the data, fits the model, makes predictions, and evaluates the model.
    Each step in this workflow is run by calling into a microservice which lives on a cluster. Note
    that we could just as easily provision a separate cluster (e.g. a GPU) for the model fitting or prediction steps,
    but for this example we'll just use a CPU cluster for everything."""
    table: rh.Table = load_and_save_data()

    # Initialize a CPU cluster to hold each of the functions / microservices needed for training our model
    cpu = rh.cluster("^rh-cpu").up_if_not()
    cpu.restart_server()

    # Create a microservice for splitting the data on the cpu cluster
    split_data = rh.function(split_data_into_training_and_test, name="split_data").to(cpu, reqs=['mlflow',
                                                                                                 'scikit-learn',
                                                                                                 'pandas<2.0.0',
                                                                                                 's3fs']).save()
    # Run the data splitting on the cluster
    train, test, train_x, train_y, test_x, test_y = split_data(table)

    # Create a microservice for fitting the model on the cpu cluster
    fit_model = rh.function(fn=fit_model_on_training_data, name="fit_model").to(cpu, reqs=['scikit-learn', 'mlflow',
                                                                                           'pandas<2.0.0']).save()

    # Run the model fitting on the cpu cluster
    lr = fit_model(alpha, l1_ratio, train_x, train_y)

    # Create a microservice for making model predictions on the cpu cluster
    make_predictions = rh.function(make_test_predictions, name="make_predictions").to(cpu).save()
    predicted_qualities = make_predictions(lr, test_x)

    # Create a microservice evaluating the model metrics, and have it live on the cpu cluster
    eval_metrics = rh.function(eval_model, name="eval_metrics").to(cpu, reqs=['scikit-learn']).save()
    rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("RMSE: %s" % rmse)
    print("MAE: %s" % mae)
    print("R2: %s" % r2)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
    else:
        mlflow.sklearn.log_model(lr, "model")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with rh.Run(name=f"train_run_{int(time.time())}") as r:
        run_training(alpha, l1_ratio)

    # save this Run locally to the `rh` folder of this working directory
    r.save()
