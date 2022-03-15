# First import the libraries that we need to use
import datetime
import json
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import requests
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


def fetch_daily_data(symbol, start, end, api_call_limit=300):
    # calculate number of iterations
    df = pd.DataFrame(columns=["one", "two"])

    df.one = [start]
    df.one = pd.to_datetime(df.one)

    df.two = [end]
    df.two = pd.to_datetime(df.two)
    
    difference = (df.two - df.one)
    difference = difference.astype('timedelta64[D]')

    iterations = difference/api_call_limit
    iterations = math.ceil(iterations)

    full_df = pd.DataFrame()
    pair_split = symbol.split('/')  # symbol must be in format XXX/XXX e.g., BTC/EUR
    symbol = pair_split[0] + '-' + pair_split[1]
    final_end = end
    for i in range(iterations):
        # update start + end
        start = pd.to_datetime(start)
        end = start + datetime.timedelta(days=api_call_limit)
        
        start = start.strftime("%Y-%m-%d")
        end = end.strftime("%Y-%m-%d")

        if i == (iterations - 1):
            end = final_end


        # extract data
        url = f'https://api.pro.coinbase.com/products/{symbol}/candles?granularity=86400&start={start}&end={end}'
        response = requests.get(url)
        
        if response.status_code == 200:  # check to make sure the response from server is good
            data = pd.DataFrame(json.loads(response.text), columns=['unix', 'low', 'high', 'open', 'close', 'volume'])
            data['date'] = pd.to_datetime(data['unix'], unit='s')  # convert to a readable date
            # data['vol_fiat'] = data['volume'] * data['close']      # multiply the BTC volume by closing price to approximate fiat volume

            # if we failed to get any data, print an error...otherwise write the file
            if data is None:
                print("Did not return any data from Coinbase for this symbol")
            else:
                full_df = pd.concat([full_df, data])

        else:
            print(response.status_code)
            print(response.text)
            print("Did not receieve OK response from Coinbase API")
            print("iteration #: " + str((iterations+1)))
        
        #move start forward
        start = pd.to_datetime(start)
        start = start + datetime.timedelta(days=api_call_limit)
        start = start.strftime("%Y-%m-%d")
    # save full dataframe
    full_df.to_csv(f'data/Coinbase_{pair_split[0] + pair_split[1]}_dailydata.csv', index=False)
    full_df.to_parquet(f'data/{pair_split[0]}-{pair_split[1]}.parquet', index=False)

    return full_df


def extract_data(token_pair, start, end):
    dt = fetch_daily_data(symbol=token_pair, start=start, end=end)
    return dt


def read_data(file_name, cols=['date','close','volume']):
    dt = pd.read_parquet('data/' + file_name)
    dt = dt[cols]
    dt['date'] = pd.to_datetime(dt.date)
    dt = dt.sort_values(['date'])
    return dt 

def create_merged_df(BAT_BTC, BAT_USD, BTC_USD, col):
    BAT_BTC = BAT_BTC[['date',col]]
    BAT_USD = BAT_USD[['date',col]]
    BTC_USD = BTC_USD[['date',col]]
    BAT_BTC = BAT_BTC.rename(columns={col: "BAT_BTC"})
    BAT_USD = BAT_USD.rename(columns={col: "BAT_USD"})
    BTC_USD = BTC_USD.rename(columns={col: "BTC_USD"})
    close_df = BAT_BTC.merge(BAT_USD, on='date')
    close_df = close_df.merge(BTC_USD, on='date')
    close_df.set_index('date',inplace=True)
    close_df = close_df.fillna(method="ffill") # forward fill two missing BAT/BTC values
    return close_df

def create_plot(df, cols, plt_show=False, plt_save=False, png_name='plot.png'):
    fig, ax = plt.subplots()
    plt.plot(df[cols])
    plt.title(str(cols[0]) + ' Price Plot')
    plt.xlabel('Date')
    plt.ylabel('Price')
    fmt = '${x:,.2f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick) 
    if plt_show == True:
        plt.show()
    if plt_save == True:
        plt.savefig(png_name)

def correlation_plot(df, plt_show=False, plt_save=False, png_name='plot.png'):
    fig, ax = plt.subplots()
    plt.plot(df)
    plt.title('Correlation Plot')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.legend(['BAT/USD & BAT/BTC','BAT/USD & BTC/USD','BAT/BTC & BTC/USD'])
    if plt_show == True:
        plt.show()
    if plt_save == True:
        plt.savefig(png_name)

def table_plot(df, plt_show=False, plt_save=False, png_name='table_plot.png'): # https://stackoverflow.com/questions/19726663/how-to-save-the-pandas-dataframe-series-data-as-a-figure
    fig =  ff.create_table(close_corr_table)
    fig.update_layout(
        autosize=False,
        width=500,
        height=200,
        font={'size':8})
    if plt_show == True:
        fig.show()
    if plt_save == True:
        fig.write_image(png_name, scale=2)

def rolling_correlation(df):
    roll_bat = df['BAT_USD'].rolling(180).corr(df['BAT_BTC'])
    roll_bat.name = 'roll_bat'
    roll_usd = df['BAT_USD'].rolling(180).corr(df['BTC_USD'])
    roll_usd.name = 'roll_usd'
    roll_bat_btc = df['BAT_BTC'].rolling(180).corr(df['BTC_USD'])
    roll_bat_btc.name = 'roll_bat_btc'
    roll_df = pd.concat([roll_bat, roll_usd, roll_bat_btc], axis=1)
    return roll_df

def create_holdings_portfolio(start_date, today, hold_only_columns = ['BAT','USD','BTC']):
        index = pd.date_range(start_date, today)
        port_df = pd.DataFrame(index=index,columns=hold_only_columns)

        trading_dates = pd.date_range(start_date, today, freq='1M')+pd.offsets.MonthBegin(-1)
        return port_df, trading_dates

def bat_acquisition(df, today, trade_dates, BAT_USD, BAT_BTC):
        for i, date in enumerate(trade_dates):
            date = date.strftime("%Y-%m-%d")
            if i == 0:
                # find conversion rates
                bat_conversion_rate = float(BAT_USD[BAT_USD['date'] == date]['close'])
                btc_conversion_rate = float(BAT_BTC[BAT_BTC['date'] == date]['close'])

                # add to portfolio
                df.loc[date]['BAT'] = 100 
                df.loc[date]['USD'] = 100 * bat_conversion_rate
                df.loc[date]['BTC'] = 100 * btc_conversion_rate
                df = df.ffill()
            else:
                # find conversion rates
                bat_conversion_rate = float(BAT_USD[BAT_USD['date'] == date]['close'])
                btc_conversion_rate = float(BAT_BTC[BAT_BTC['date'] == date]['close'])

                # add to portfolio
                df.loc[date, 'BAT'] += 100
                df.loc[date, 'USD'] += 100 * bat_conversion_rate
                df.loc[date, 'BTC'] += 100 * btc_conversion_rate
                
                # fill down
                df[date:today]['BAT'] = df.loc[date,'BAT']
                df[date:today]['USD'] = df.loc[date,'USD']
                df[date:today]['BTC'] = df.loc[date,'BTC']

        return df

def calc_hold_only_roi(df, bat_usd, btc_usd, start_date, today):
        df_roi = df.copy()
        BAT_USD_roi = bat_usd.set_index('date')
        BTC_USD_roi = btc_usd.set_index('date')

        BAT_USD_roi = BAT_USD_roi.loc[start_date:today]['close']
        BTC_USD_roi = BTC_USD_roi.loc[start_date:today]['close']
        
        df_roi['BAT_ROI'] = df_roi['BAT'] * BAT_USD_roi
        df_roi['BTC_ROI'] = df_roi['BTC'] * BTC_USD_roi
        df_roi = df_roi.rename(columns={'USD':'USD_ROI'})

        df_roi = df_roi[['USD_ROI','BAT_ROI','BTC_ROI']]

        return df_roi

def roi_plot(df, plt_show=False, plt_save=False, png_name='roi_plot.png'):
    fig, ax = plt.subplots()
    plt.plot(df)
    plt.title('ROI/Price Plot')
    plt.xlabel('Date')
    plt.ylabel('Price')
    fmt = '${x:,.2f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick) 
    plt.legend(['USD_ROI','BAT_ROI','BTC_ROI'])
    if plt_show == True:
        plt.show()
    if plt_save == True:
        plt.savefig(png_name)


    
if __name__ == "__main__":
    print('execution started')
    # extract data
    # BTC_USD = extract_data('BTC/USD', '2017-11-09', '2022-02-28')

    # read data
    BAT_BTC = read_data('BAT-BTC.parquet')
    BAT_USD = read_data('BAT-USD.parquet')
    BTC_USD = read_data('BTC-USD.parquet')

    # # create close dataframe
    # close_df = create_merged_df(BAT_BTC, BAT_USD, BTC_USD, 'close')

    # # print graph
    # create_plot(close_df, ['BAT_USD'], False, False, 'plots/BAT_USD_Price_Plot.png')
    # create_plot(close_df, ['BTC_USD'], False, False, 'plots/BTC_USD_Price_Plot.png')

    # # correlation test + table
    # close_corr_table = close_df.corr()
    # table_plot(close_corr_table, False, False, 'plots/correlation_table.png')

    # # rolling correlation
    # rolling_correlation_df = rolling_correlation(close_df)
    # correlation_plot(rolling_correlation_df, False, False, 'plots/rolling_correlation.png')


    # ### market holdings

    start_date = '2021-01-01'
    # # today = datetime.datetime.now().strftime("%Y-%m-%d")
    today = '2022-02-28'

    # hold_only_portfolio, trade_dates = create_holdings_portfolio(start_date, today, hold_only_columns = ['BAT','USD','BTC'])
    # hold_only_portfolio = bat_acquisition(hold_only_portfolio, today, trade_dates, BAT_USD, BAT_BTC)


    # ### ROI conversion/analysis (convert everything to USD for appropriate $ analysis)
    # hold_only_portfolio_roi = calc_hold_only_roi(hold_only_portfolio, BAT_USD, BTC_USD, start_date, today)
    # roi_plot(hold_only_portfolio_roi, False, False, 'plots/roi_plot.png')



    # predict the price of BAT (relative to USD) # https://machinelearningmastery.com/xgboost-for-time-series-forecasting/
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[0]
        df = pd.DataFrame(data)
        cols = list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
        # put it all together
        agg = pd.concat(cols, axis=1)
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg.values

    def train_test_split(data, n_test):
        return data[:-n_test, :], data[-n_test:, :]


    # walk-forward validation for univariate data
    def walk_forward_validation(data, n_test):
        predictions = list()
        # split dataset
        train, test = train_test_split(data, n_test)
        # seed history with training dataset
        history = [x for x in train]
        # step over each time-step in the test set
        for i in range(len(test)):
            # split test row into input and output columns
            testX, testy = test[i, :-1], test[i, -1]
            # fit model on history and make a prediction
            yhat = xgboost_forecast(history, testX)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
        # estimate prediction error
        error = mean_absolute_error(test[:, -1], predictions)
        return error, test[:, 1], predictions

    # fit an xgboost model and make a one step prediction
    def xgboost_forecast(train, testX):
        # transform list into array
        train = np.asarray(train)
        # split into input and output columns
        trainX, trainy = train[:, :-1], train[:, -1]
        # fit model
        model = XGBRegressor(objective='reg:squaredlogerror', n_estimators=1000) #try: squaredlogerror; squarederror
        model.fit(trainX, trainy)
        # make a one-step prediction
        yhat = model.predict(np.asarray([testX]))
        return yhat[0]

    


    #CV: Iterable parameters
        # objective: reg:squarederror reg:squaredlogerror
        # n_estimators: default = 1000
        # n_in (how far to look back)
        # n_test (# of days to estiamte)
        # eval metrics: MAE
    # from matplotlib import pyplot

    
    # for i in range(1,31):
    # data = series_to_supervised(BAT_USD['close'],n_in=30)
    # mae, y, yhat = walk_forward_validation(data, 31)
    # print(mae)
    # print(str(i) + ' - ' + str(mae))
        # print(y)
        # print(yhat)

    # print(y)
    # print(yhat)
    # pyplot.plot(y, label='Expected')
    # pyplot.plot(yhat, label='Predicted')
    # pyplot.legend()
    # pyplot.show()

    # # transform the time series data into supervised learning
    # train = series_to_supervised(BAT_USD['close'], n_in=30)
    # # split into input and output columns
    # trainX, trainy = train[:, :-1], train[:, -1]

    # # fit model
    # model = XGBRegressor(objective='reg:squaredlogerror', n_estimators=1000)
    # model.fit(trainX, trainy)

    # # construct an input for a new preduction
    # row = BAT_USD['close'].values[-30:]

    # # make a one-step prediction
    # yhat = model.predict(np.asarray([row]))
    # print('Input: %s, Predicted: %.3f' % (row, yhat[0]))


    trading_df = BAT_USD.set_index('date')
    trading_df = trading_df['close']

    trade_start_date = '2021-02-01'

####### uncomment #####################################

    trade_start_date = datetime.datetime.strptime(trade_start_date, '%Y-%m-%d')
    trade_end_date = datetime.datetime.strptime('2022-02-28', '%Y-%m-%d')
    # delta = trade_end_date - trade_start_date


    # price_predictions = []
    # for i in range(delta.days + 1):
    #     train_start = trade_start_date - datetime.timedelta(days=59) - pd.DateOffset(days=1) + datetime.timedelta(days=i)
    #     train_end = trade_start_date - pd.DateOffset(days=1) + datetime.timedelta(days=i)
    #     train_data = trading_df[train_start:train_end]

    #     train = series_to_supervised(train_data, n_in=30)
    #     trainX, trainy = train[:, :-1], train[:, -1]

    #     model = XGBRegressor(objective='reg:squaredlogerror', n_estimators=1000)
    #     model.fit(trainX, trainy)

    #     row = train_data.values[-30:]
    #     yhat = model.predict(np.asarray([row]))
    #     price_predictions.append(yhat[0])
    
    # trading_df = trading_df[trade_start_date:trade_end_date]
    # trading_df = pd.DataFrame(trading_df)
    # trading_df['predictions'] = price_predictions
########################################################


    trading_df = pd.read_csv('test.csv')

    trading_df['date'] = pd.to_datetime(trading_df['date'])
    trading_df.set_index('date',inplace=True)

    error = mean_absolute_error(trading_df['close'].values, trading_df['predictions'].values)
    
    trading_df['cash_holdings'] = np.nan
    trading_df['bat_holdings'] = np.nan
    

    trading_df.loc[trade_start_date,['cash_holdings']] = 0
    trading_df.loc[trade_start_date,['bat_holdings']] = 200

    trading_df['bat_trade'] = np.where(trading_df['predictions'].shift(-1) > trading_df['close'],'buy','sell')

    bat_income_dates = pd.date_range(start_date, today, freq='1M')+pd.offsets.MonthBegin(-1)

    for index, row in trading_df.iterrows():
        if index == trade_start_date:
            cur_action = row['bat_trade']
            cur_close = row['close']
            cur_cash = row['cash_holdings']
            cur_bat = row['bat_holdings']
        else:
            index = index.strftime("%Y-%m-%d")
            if index in bat_income_dates:
                print('hello')
                print(trading_df.loc[[index],['bat_holdings']])
                print(trading_df.loc[index]['bat_holdings'])
                print(trading_df.loc[index]['bat_holdings'] + 100)
                trading_df.loc[[index],['bat_holdings']] = trading_df.loc[index]['bat_holdings'] + 100
                print(trading_df.loc[[index],['bat_holdings']])
            if cur_action == 'sell':
                trading_df.loc[[index],['bat_holdings']] = 0
                if cur_bat > 0:
                    trading_df.loc[[index],['cash_holdings']] = cur_bat * cur_close
                else:
                    trading_df.loc[[index],['cash_holdings']] = cur_cash
            else:
                trading_df.loc[[index],['cash_holdings']] = 0
                if cur_cash > 0:
                    trading_df.loc[[index],['bat_holdings']] = cur_cash / cur_close
                else:
                    trading_df.loc[[index],['bat_holdings']] = cur_bat
            cur_action = trading_df.loc[index]['bat_trade']
            cur_close = trading_df.loc[index]['close']
            cur_cash = trading_df.loc[index]['cash_holdings']
            cur_bat = trading_df.loc[index]['bat_holdings']

    
    print(trading_df)
    # trading_df.to_csv('test2.csv')




    
    



    







    





