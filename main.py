# First import the libraries that we need to use
import datetime
import math
import pandas as pd
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




if __name__ == "__main__":
    # extract data
    # BTC_USD = extract_data('BTC/USD', '2017-11-09', '2022-02-28')

    # read data
    BAT_BTC = read_data('BAT-BTC.parquet')
    BAT_USD = read_data('BAT-USD.parquet')
    BTC_USD = read_data('BTC-USD.parquet')

    # correlation test

