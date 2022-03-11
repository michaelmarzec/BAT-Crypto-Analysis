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

    # create close dataframe
    close_df = create_merged_df(BAT_BTC, BAT_USD, BTC_USD, 'close')

    # print graph
    create_plot(close_df, ['BAT_USD'], False, False, 'plots/BAT_USD_Price_Plot.png')
    create_plot(close_df, ['BTC_USD'], False, False, 'plots/BTC_USD_Price_Plot.png')

    # correlation test + table
    close_corr_table = close_df.corr()
    table_plot(close_corr_table, False, False, 'plots/correlation_table.png')

    # rolling correlation
    rolling_correlation_df = rolling_correlation(close_df)
    correlation_plot(rolling_correlation_df, False, False, 'plots/rolling_correlation.png')


    ### market holdings

    start_date = '2021-01-01'
    # today = datetime.datetime.now().strftime("%Y-%m-%d")
    today = '2022-02-28'

    hold_only_portfolio, trade_dates = create_holdings_portfolio(start_date, today, hold_only_columns = ['BAT','USD','BTC'])
    hold_only_portfolio = bat_acquisition(hold_only_portfolio, today, trade_dates, BAT_USD, BAT_BTC)


    ### ROI conversion/analysis (convert everything to USD for appropriate $ analysis)
    hold_only_portfolio_roi = calc_hold_only_roi(hold_only_portfolio, BAT_USD, BTC_USD, start_date, today)
    roi_plot(hold_only_portfolio_roi, False, False, 'plots/roi_plot.png')









    




