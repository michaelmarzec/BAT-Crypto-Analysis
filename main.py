# First import the libraries that we need to use
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import plotly.figure_factory as ff
import requests
import json

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

def create_plot(df, cols, y_axis, dollar=False, plt_show=False, plt_save=False, png_name='plot.png'):
    fig, ax = plt.subplots()
    plt.plot(df[cols])
    plt.title(str(cols[0]) + ' Price Plot')
    plt.xlabel('Date')
    plt.ylabel(y_axis)
    # ax.yaxis.set_major_formatter('${x:1.2f}')
    if dollar == True:
        fmt = '${x:,.2f}'
        tick = mtick.StrMethodFormatter(fmt)
        ax.yaxis.set_major_formatter(tick) 
    if plt_show == True:
        plt.show()
    if plt_save == True:
        plt.savefig(png_name)

def correlation_plot(df, cols, y_axis, plt_show=False, plt_save=False, png_name='plot.png'):
    fig, ax = plt.subplots()
    plt.plot(df[cols])
    plt.title('Correlation Plot')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
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
    
if __name__ == "__main__":
    # extract data
    # BTC_USD = extract_data('BTC/USD', '2017-11-09', '2022-02-28')

    # read data
    BAT_BTC = read_data('BAT-BTC.parquet')
    BAT_USD = read_data('BAT-USD.parquet')
    BTC_USD = read_data('BTC-USD.parquet')

    # create close dataframe
    close_df = create_merged_df(BAT_BTC, BAT_USD, BTC_USD, 'close')

    # print graph
    create_plot(close_df, ['BAT_USD'], 'Price', True, False, False, 'plots/BAT_USD_Price_Plot.png')
    create_plot(close_df, ['BTC_USD'], 'Price', True, False, False, 'plots/BTC_USD_Price_Plot.png')

    # correlation test + table
    close_corr_table = close_df.corr()
    table_plot(close_corr_table, False, False, 'plots/correlation_table.png')

    # rolling correlation
    roll_bat = close_df['BAT_USD'].rolling(180).corr(close_df['BAT_BTC'])
    roll_bat.name = 'roll_bat'
    roll_usd = close_df['BAT_USD'].rolling(180).corr(close_df['BTC_USD'])
    roll_usd.name = 'roll_usd'
    roll_bat_btc = close_df['BAT_BTC'].rolling(180).corr(close_df['BTC_USD'])
    roll_bat_btc.name = 'roll_bat_btc'
    roll_df = pd.concat([roll_bat, roll_usd, roll_bat_btc], axis=1)
    correlation_plot(roll_df, ['roll_bat', 'roll_usd', 'roll_bat_btc'], True, True, 'plots/rolling_correlation.png')



    




