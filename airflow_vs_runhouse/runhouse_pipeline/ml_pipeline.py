import pickle

import runhouse as rh

from airflow_vs_runhouse.runhouse_pipeline.helpers import preprocess_raw_data, split_data, fit_and_save_model, \
    predict_test_wt_arima, measure_accuracy, load_raw_data


# Based on Delivery Hero Airflow ML Workshop
# https://github.com/deliveryhero/pyconde2019-airflow-ml-workshop


def preprocessing_and_data_split(raw_df, cpu):
    # Send the function for loading the dataset to the cluster along with the requirements it needs to run
    preprocessed_data_on_cpu = rh.function(name="preprocess_data", fn=preprocess_raw_data, system=cpu,
                                           reqs=["pmdarima"]).save()

    # Run the preprocessing on the cluster, which returns a remote reference to the dataset saved on the cluster
    dataset_ref_on_cpu = preprocessed_data_on_cpu(raw_df)
    print(f"Saved dataset on cluster to path: {dataset_ref_on_cpu.path}")

    # Run the data splitting on the cluster, which returns a remote reference to the train + test data on the cluster
    split_data_on_cpu = rh.function(name="split_data", fn=split_data, system=cpu).save()
    train_data_ref, test_data_ref = split_data_on_cpu(preprocessed_dataset_ref=dataset_ref_on_cpu)

    print(f"Saved train data to path: {train_data_ref.path} on the cluster")
    print(f"Saved test data to path: {test_data_ref.path} on the cluster")

    return train_data_ref, test_data_ref


def model_training(gpu, train_data_ref, test_data_ref):
    train_model_on_gpu = rh.function(fn=fit_and_save_model, system=gpu, reqs=["pmdarima"]).save()

    # Run the training on the cluster
    model_ref = train_model_on_gpu(train_dataset_ref=train_data_ref)
    print(f"Saved model on cluster to path: {model_ref.path}")

    predict_test_on_gpu = rh.function(fn=predict_test_wt_arima, system=gpu, reqs=["pmdarima"]).save()
    test_predictions_ref = predict_test_on_gpu(test_dataset_ref=test_data_ref)
    print(f"Saved test data predictions on cluster to path: {test_predictions_ref.path}")

    return model_ref, test_predictions_ref


def run_pipeline():
    """
    The Runhouse pipeline consists of the same steps outlined in the Airflow DAG:
    preprocess >> split data >> fit and save model >> predict test >> measure accuracy

    We can easily deploy each of these stages as microservices, or Runhouse function objects containing the code
    and dependencies required to run. For the preprocessing stage, we provision a 32 CPU cluster to handle
    running the preprocessing and data splitting stages.

    For the model fitting and predict stages, we provision a GPU (in this case an A10G) for our Runhouse
    microservices to live.

    Notice how we pass object refs between each of the microservices - this is to prevent having to bounce around data
    between the local env and the cluster.
    """
    # Launch a new cluster (with 32 CPUs) to handle loading and processing of the dataset
    cpu = rh.cluster(name="^rh-32-cpu").up_if_not().save()

    raw_df = load_raw_data()
    train_data_ref, test_data_ref = preprocessing_and_data_split(raw_df, cpu)

    # Launch a new cluster (with a GPU) to handle model training
    gpu = rh.cluster(name='rh-a10x') if rh.exists('rh-a10x') else rh.cluster(name='rh-a10x',
                                                                             instance_type='A10:1').up_if_not()
    model_ref, test_predictions_ref = model_training(gpu, train_data_ref, test_data_ref)
    accuracy_on_gpu = rh.function(fn=measure_accuracy, system=gpu, reqs=["pmdarima"]).save()
    accuracy_ref = accuracy_on_gpu(test_dataset_ref=test_data_ref, predicted_test_ref=test_predictions_ref)
    print(f"Accuracy: {pickle.loads(accuracy_ref.data)}")


if __name__ == "__main__":
    run_pipeline()
