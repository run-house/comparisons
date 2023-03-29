import math
import os
import pickle
import pandas as pd
import numpy as np

import runhouse as rh

from pmdarima import auto_arima


def preprocess_raw_data(raw_df):
    raw_df['Date'] = list(map(lambda x: pd.to_datetime(x), raw_df['Date']))
    raw_df = raw_df.sort_values('Date')

    processed_df = raw_df.rename(index=str,
                                 columns={'Daily minimum temperatures in Melbourne, '
                                          'Australia, 1981-1990': 'y'})

    for sub in processed_df['y']:
        if '?' in sub:
            processed_df.loc[processed_df['y'] == sub, 'y'] = sub.split('?')[1]

    dataset_obj = rh.blob(data=pickle.dumps(processed_df), name="processed_dataset").write().save()
    return dataset_obj


def split_data(preprocessed_dataset_ref, n_weeks_to_test=2):
    """
    Reads preprocessed data from the cluster and splits it to test/train and saves it to
    the cluster.
    :param n_weeks_to_test: Number of weeks for the test data. Default is 2.
    """
    preprocessed_blob = rh.Blob.from_name(preprocessed_dataset_ref.name)
    preprocessed_data = pickle.loads(preprocessed_blob.data)

    n_days_for_test = n_weeks_to_test * 7

    test_df = preprocessed_data[-n_days_for_test:]
    train_df = preprocessed_data[:-n_days_for_test]

    # Save to blob storage
    train_blob = rh.blob(data=pickle.dumps(train_df), name="train_data").write().save()
    test_blob = rh.blob(data=pickle.dumps(test_df), name="test_data").write().save()

    return train_blob, test_blob


def fit_and_save_model(train_dataset_ref):
    """
    Runs Prophet for the train dataframe. It reads data from the cluster and saves the model
    object to the cluster.
    """
    train_blob = rh.Blob.from_name(train_dataset_ref.name)
    train_df = pickle.loads(train_blob.data)

    train_df['Date'] = list(map(lambda x: pd.to_datetime(x), train_df['Date']))
    train_df = train_df.set_index('Date')

    model = auto_arima(train_df, start_p=1, start_q=1,
                       test='adf',
                       max_p=1, max_q=1, m=12,
                       start_P=0, seasonal=True,
                       d=None, D=1, trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)

    model_obj = rh.blob(data=pickle.dumps(model), name="arima").write().save()
    return model_obj


def predict_test_wt_arima(test_dataset_ref):
    """
    Reads test dataframe and model object from the cluster and makes prediction.
    Data with predicted values for test dataframe will be saved to the cluster.
    """
    test_blob = rh.Blob.from_name(test_dataset_ref.name)
    test_df = pickle.loads(test_blob.data)

    model_blob = rh.Blob.from_name("arima")
    model = pickle.loads(model_blob.data)

    fitted, confint = model.predict(n_periods=len(test_df), return_conf_int=True)

    predicted_test = pd.merge(
        pd.DataFrame(fitted), pd.DataFrame(confint), right_index=True, left_index=True)

    predicted_test = predicted_test.rename(index=str,
                                           columns={'0_x': 'yhat',
                                                    '0_y': 'yhat_lower',
                                                    1: 'yhat_upper'})

    predicted_test_obj = rh.blob(data=pickle.dumps(predicted_test), name="predicted_test").write().save()

    return predicted_test_obj


def measure_accuracy(test_dataset_ref, predicted_test_ref):
    """
    Uses the above defined accuracy metrics and calculates accuracy for both test series in
    terms of MAPE and RMSE. Saves those results to local as a csv file on the cluster.
    :return: A dictionary with accuracy metrics for test dataset.
    """
    test_blob = rh.Blob.from_name(test_dataset_ref.name)
    test_df = pickle.loads(test_blob.data)

    predicted_test_blob = rh.Blob.from_name(predicted_test_ref.name)
    predicted_test = pickle.loads(predicted_test_blob.data)

    mape_test = calculate_mape(test_df['y'], predicted_test['yhat'])

    rmse_test = calculate_rmse(test_df['y'], predicted_test['yhat'])

    days_in_test = len(test_df)

    accuracy_dict = {'mape_test': [mape_test],
                     'rmse_test': [rmse_test],
                     'days_in_test': [days_in_test]}

    acc_df = pd.DataFrame(accuracy_dict)

    accuracy_obj = rh.blob(data=pickle.dumps(acc_df.to_dict('index')[0]), name="accuracy").write().save()
    return accuracy_obj


def forecast_wt_arima_for_date(input_date, model_ref, test_dataset_ref):
    """
    :param input_date: A date as a string in ISO format (yyyy-mm-dd).
    :return: Dictionary with the forecasted values.
            `yhat`: Forecasted value for given date.
            `yhat_upper`: Forecasted upper value for given date & confidence intervals.
            `yhat_lower`: Forecasted lower value for given date & confidence intervals.
    """
    print(f"Computing forecast for {input_date}")

    model_blob = rh.Blob.from_name(model_ref.name)
    model = pickle.loads(model_blob.data)

    test_blob = rh.Blob.from_name(test_dataset_ref.name)
    test_df = pickle.loads(test_blob.data)

    min_test_date = pd.to_datetime(test_df.index.min())

    date_diff = pd.to_datetime(input_date) - min_test_date

    fitted, confint = model.predict(n_periods=date_diff.days, return_conf_int=True)

    forecast_results = pd.merge(
        pd.DataFrame(fitted), pd.DataFrame(confint), right_index=True, left_index=True)

    forecast_results = forecast_results.rename(
        index=str, columns={'0_x': 'yhat', '0_y': 'yhat_upper', 1: 'yhat_lower'})

    final_forecast = forecast_results[-1:]
    final_forecast['Date'] = input_date
    final_forecast = final_forecast.set_index('Date')

    return final_forecast.to_dict('index')[input_date]


# ------------------------ Helper Functions ------------------------
def calculate_mape(y, yhat):
    """
    Calculates Mean Average Percentage Error.
    :param y: Actual values as series
    :param yhat: Predicted values as series
    :return: MAPE as percentage
    """

    y = y.replace(0, np.nan)

    error_daily = y - yhat
    abs_daily_error = list(map(abs, error_daily))
    relative_abs_daily_error = abs_daily_error / y

    mape = (np.nansum(relative_abs_daily_error) / np.sum(~np.isnan(y))) * 100

    return mape


def calculate_rmse(y, yhat):
    """
    Calculates Root Mean Square Error
    :param y: Actual values as series
    :param yhat: Predicted values as series
    :return: RMSE value
    """
    error_sqr = (y - yhat) ** 2
    error_sqr_rooted = list(map(lambda x: math.sqrt(x), error_sqr))
    rmse = sum(error_sqr_rooted) / len(error_sqr_rooted)

    return rmse


def load_raw_data():
    # Load in the raw data from local folder
    data_path = os.path.join(os.getcwd(), 'data', 'raw_data', 'daily_minimum_temp.csv')
    raw_df = pd.read_csv(data_path, error_bad_lines=False)
    return raw_df
