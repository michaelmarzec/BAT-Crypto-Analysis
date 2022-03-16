# First import the libraries that we need to use
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
# import plotly.figure_factory as ff
# import requests
# import json

trading_df['cash_holdings'] = trading_df['cash_holdings'].ffill()
    trading_df['bat_holdings'] = trading_df['bat_holdings'].ffill()

df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
# df['col3'] = np.nan
# df['col3'] = df[0]

# df.loc[0,['col3']] = 1


df['col4'] = df['col2'].shift(-1)

print(df)


# from xgboost import XGBRegressor
# #  start = start.strftime("%Y-%m-%d")


# # dates= pd.date_range('2021-01-01','2022-03-01', freq='1M')+pd.offsets.MonthBegin(1)
# # print(dates)
# # for x in dates:
# #     print(x.strftime("%Y-%m-%d"))





    # # transform the time series data into supervised learning
    # train = series_to_supervised(BAT_USD['close'], n_in=30)
    # print(pd.DataFrame(train))
    
    # # split into input and output columns
    # trainX, trainy = train[:, :-1], train[:, -1]
    # print(pd.DataFrame(trainX))
    # print(pd.DataFrame(trainy))
    # breakpoint()

    # # fit model
    # model = XGBRegressor(objective='reg:squaredlogerror', n_estimators=1000)
    # model.fit(trainX, trainy)

    # # construct an input for a new preduction
    # row = BAT_USD['close'].values[-30:]

    # # make a one-step prediction
    # yhat = model.predict(np.asarray([row]))
    # print('Input: %s, Predicted: %.3f' % (row, yhat[0]))
