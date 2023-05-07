import sys
from urllib.parse import urlparse

import mlflow
import pandas as pd
import runhouse as rh


def re_run_training(alpha, l1_ratio):
    with rh.run(name="re-running_training"):
        table = rh.Table.from_name("raw_data")
        data: pd.DataFrame = table.fetch()

        fit_model = rh.Function.from_name("fit_model")

        split_data = rh.Function.from_name(name="split_data")
        train, test, train_x, train_y, test_x, test_y = split_data(data)

        # Run the model fitting on the cpu cluster
        lr = fit_model(alpha, l1_ratio, train_x, train_y)

        make_predictions = rh.Function.from_name(name="make_predictions")
        predicted_qualities = make_predictions(lr, test_x)

        eval_metrics = rh.Function.from_name(name="eval_metrics")
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

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
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    re_run_training(alpha, l1_ratio)
