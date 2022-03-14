# First import the libraries that we need to use
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import plotly.figure_factory as ff
import requests
import json


from xgboost import XGBRegressor
#  start = start.strftime("%Y-%m-%d")


# dates= pd.date_range('2021-01-01','2022-03-01', freq='1M')+pd.offsets.MonthBegin(1)
# print(dates)
# for x in dates:
#     print(x.strftime("%Y-%m-%d"))