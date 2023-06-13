
import warnings
from datetime import datetime
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.filterwarnings('ignore')


def get_time_series_data(file):

    cigData = pd.read_csv(file, index_col=0).dropna()
    cigData.rename(columns={'Time': 'Month'}, inplace=True)

    cigData.Month = pd.to_datetime(cigData.Month, format='%Y-%m')
    cigData.rename(columns={'#CigSales': 'cig_sales'}, inplace=True)

    cigData.cig_sales = cigData.cig_sales.astype('int64')

    cigData.set_index('Month', inplace=True)
    return np.log(cigData)


def train_test_split(cigData):

    # Make train and test variables, with 'train, test'
    split = int(len(cigData) * 0.8)
    df_train, df_test = cigData[0:split], cigData[split:len(cigData)]
    return df_train, df_test


def run_arima(data, order):
    model = ARIMA(data, order=order)
    model_fit_ = model.fit()
    return model_fit_


def test_model(data, arima_order):
    # Needs to be an integer because it is later used as an index.
    # Use int()
    data.dropna(inplace=True)
    split = int(len(data) * 0.8)
    # Make train and test variables, with 'train, test'
    train, test = data[0:split], data[split:len(data)]
    past = [x for x in train.cig_sales]
    # make predictions
    predictions = list()
    for i in range(len(test)):
        test_model_fit = run_arima(data=past, order=arima_order)
        predict = test_model_fit.forecast()[0]
        predictions.append(predict)
        past.append(test.iloc[i].cig_sales)
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    # Return the error
    return error, test


# Make a function to evaluate different ARIMA models with several different p, d, and q values.
def evaluate_arima(data, p_values, q_values):
    test_results = []
    best =None
    for p_val in p_values:
        for q_val in q_values:
            order = (p_val, 1, q_val)
            loss, _ = test_model(data=data, arima_order=order)
            record = dict(order=order, loss=loss)
            if best is None or loss < best['loss']:
                best = record
            test_results.append( record )

    return sorted(test_results, key=itemgetter('loss')), best


def evaluate_parameters(df):

    p = [0, 1, 2, 12]
    q = [0, 1, 2, 3]
    return evaluate_arima(df, p, q)


def create_tail(df, forcast):

    # Declare a variable called forecast_period with the amount of months to forecast, and
    # create a range of future dates that is the length of the periods you've chosen to forecast



    date_range = forcast.index.to_series()
    # timestamp = np.max(dates)
    # datetime.fromtimestamp(timestamp, tz = None)

    # date_range = pd.date_range(last_date, periods = forecast_period,
    #                            freq='MS').strftime("%Y-%m-%d").tolist()

    merge=pd.merge(df, forcast, how='left', left_index=True, right_index=True)

    print(dates)
    # # Convert that range into a dataframe that includes your predictions
    # # First, call DataFrame on pd
    # future_months = pd._ _ _(date_range, columns = ['Month']
    # # Let's now convert the 'Month' column to a datetime object with to_datetime
    # future_months['Month'] = pd._ _ _(future_months['Month'])
    # future_months.set_index('Month', inplace = True)
    # future_months['Prediction'] = forecast[0]


def do_work():
    cigData = get_time_series_data('CowboyCigsData.csv')
    model_fit = run_arima(data=cigData, order=(2 ,1 ,1))
    forcast = model_fit.forecast(len(cigData))
    create_tail(cigData, forcast)
    results, best = evaluate_parameters(cigData)
    print(best)


if __name__ == '__main__':
    do_work()
    print("Bye")
