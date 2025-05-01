'''
Binance Market Data Loader

This script provides a class to fetch and manage OHLCV and funding rate data (both spot and perpetual futures)
through Binance Python API.


Required Packages:
- pandas (>=1.5.0)
- python-binance (>=1.0.16)
- datetime (built-in)
- os (built-in)
- glob (built-in)

Example:
    crypto_data = CryptoMarketData(
        symbol="BTCUSDT",
        time_frame="1h",
        start_date="2023-01-01",
        end_date="2023-12-31",
        futures=True
    )
    df = crypto_data.getDataDf()
    crypto_data.writeOfflineCSV(df) # write the CSV data to the offline path
    crypto_data.updateOfflineCSV() # update and save the offline CSV file

Author: Haoqing Wu
Date: February 2024
'''

import pandas as pd
import os
import datetime as dt

from binance.client import Client
import glob


class CryptoMarketData:
    def __init__( self, symbol: str, time_frame: str, start_date: str = None, end_date = None, offline_path = None,
                 futures = False):
        """
        Inputs:
            - symbol:       a string of trading pair, e.g. "BTCUSDT"
            - time_frame:   a string of time_frame of data
            - start_date:   starting date of the data
            - end_date:     ending date of the data
            - offline_path: the path to the directory storing the offline data
            - futures:      a boolean to indicate if the data is from futures or spot market
        """
        __Time_Frame = {
                        "5m": Client.KLINE_INTERVAL_5MINUTE,
                        "15m": Client.KLINE_INTERVAL_15MINUTE,
                        "1h": Client.KLINE_INTERVAL_1HOUR,
                        "4h": Client.KLINE_INTERVAL_4HOUR,
                        "1d": Client.KLINE_INTERVAL_1DAY,
                        "1w": Client.KLINE_INTERVAL_1WEEK,
                        "1m": Client.KLINE_INTERVAL_1MONTH,
                        }

        self.symbol     = symbol
        self.start_date = start_date
        self.end_date   = end_date
        self.time_frame = __Time_Frame[ time_frame ]
        self.futures    = futures

        if not offline_path:
            self.OfflinePath = os.path.join(os.path.dirname(__file__), 'OfflineData')
            if not os.path.exists(self.OfflinePath):
                os.makedirs(self.OfflinePath)
                print(f"Created directory: {self.OfflinePath}")
        else:
            self.OfflinePath = offline_path

    def __getFundingRate(self) -> pd.DataFrame:
        """ pull historical funding rates """

        api_key = ''
        api_secret = ''
        client = Client(api_key, api_secret)

        if not self.start_date:
            raise ValueError("Please specify a start date to retrieve funding rate!")

        start_date_timestamp = int(pd.to_datetime(self.start_date).timestamp() * 1000)  # in ms
        cur_timestamp = int(pd.to_datetime(dt.datetime.now(dt.timezone.utc)).timestamp() * 1000)

        limit = 900  # the maximum amount of funding rates the API can pull
        eight_hrs_in_ms = 8 * 3600 * 1000
        df = pd.DataFrame()

        while start_date_timestamp < cur_timestamp:
            df_temp = client.futures_funding_rate(symbol=self.symbol,
                                                  startTime=start_date_timestamp,
                                                  endTime=cur_timestamp,
                                                  limit=limit)
            df_temp = pd.DataFrame(df_temp)

            df_temp['fundingTime'] = df_temp['fundingTime'].apply(
                lambda x: dt.datetime.fromtimestamp(x / 1000, tz=dt.timezone.utc).replace(tzinfo=None, microsecond=0))
            df = pd.concat([df, df_temp], axis=0, ignore_index=True)

            start_date_timestamp = int(df['fundingTime'].iloc[-1].timestamp() * 1000 + eight_hrs_in_ms)

        df.rename(columns={'fundingTime': 'Open Time'}, inplace=True)
        df.drop(['symbol', 'markPrice'], axis=1, inplace=True)
        df.set_index('Open Time', inplace=True)

        return df

    def _getMarketData(self) -> pd.DataFrame:
        """ pull historical klines from binance """

        api_key = ''
        api_secret = ''

        client = Client( api_key, api_secret )
        # Get klines
        if self.futures:
            klines = client.futures_historical_klines( self.symbol, self.time_frame, self.start_date, self.end_date )
        else:
            klines = client.get_historical_klines( self.symbol, self.time_frame, self.start_date, self.end_date )
        # Create DataFrame and set column names
        df = pd.DataFrame(klines, columns=['Open Time',
                                           'Open',
                                           'High',
                                           'Low',
                                           'Close',
                                           'Volume',
                                           # volume in based asset, e.g., for 'BTCUSDT', the volume means the number of BTC traded in that kline
                                           'Close Time',
                                           'Quote Asset Volume',
                                           'Number of Trades',
                                           'Taker Buy Base',
                                           # total base asset volume are contributed by the taker buy orders
                                           'Taker Buy Quote',
                                           # the total value of the taker buy trades in terms of USDT (the quote currency)
                                           'Ignore',
                                           ])

        # don't know what Ignore does
        df['Taker_Avg_Cost'] = df['Taker Buy Quote'].apply(eval) / df['Taker Buy Base'].apply(eval)
        df.drop(['Quote Asset Volume', 'Ignore', 'Taker Buy Quote', 'Close Time'], axis=1, inplace=True)
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df.set_index('Open Time', inplace=True)

        if not self.start_date:
            self.start_date = (df.index[0].date()).strftime('%Y-%m-%d')
        df_funding_rate = self.__getFundingRate()
        df = df.join( df_funding_rate, how = 'left' )
        df[ 'fundingRate' ].fillna( method = 'ffill', inplace = True )

        # each columns values to float32
        for col in df.columns:
            df[col] = df[col].astype('float32')

        return df

    def getDataDf(self) -> pd.DataFrame:
        """ load market data from binance and return a dataframe """

        dataDf = self._getMarketData()
        if self.start_date and self.end_date:
            print(
                f'{dt.datetime.now()}: Pulled {self.symbol} {self.time_frame} Data from {self.start_date} to {self.end_date}.')
        elif self.start_date and (not self.end_date):
            print(
                f'{dt.datetime.now()}: Pulled {self.symbol} {self.time_frame} Data from {dataDf.index[0]} to {dataDf.index[-1]}.')
        else:
            print(
                f'{dt.datetime.now()}: Pulled {self.symbol} {self.time_frame} Data from {dataDf.index[0]} to {dataDf.index[-1]}.')
        return dataDf

    def updateOfflineCSV( self ) -> pd.DataFrame:
        """
        Check if the path exists then update the offline data of each symbol.
        The file name would be like, "../BTCUSDT_4h_Main.csv".
        """

        path = self.OfflinePath
        if not os.path.exists( path ):
            raise FileNotFoundError(f"The file path {path} does not exist.")

        # Search for the CSV file with the format ticker_timeframe_Main
        search_pattern = os.path.join( path, f"{ self.symbol }_{ self.time_frame }_Main.csv")
        matching_files = glob.glob( search_pattern )

        if not matching_files:
            self.start_date = "2018-01-01"
            new_df = self._getMarketData()
            new_df.to_csv( os.path.join( path, f"{ self.symbol }_{ self.time_frame }_Main.csv") )
            print(f"Main CSV {self.symbol} data stored at {path}")

            return new_df

        # Use the first matching file
        path = matching_files[ 0 ]

        # Load existing data
        existing_df = pd.read_csv( path, index_col = 'Open Time', parse_dates = ['Open Time'])

        # Get the last date from the existing data
        last_date = existing_df.index[ -1 ]

        # Update the start_date to the last date of the existing data
        self.start_date = ( last_date.date() - dt.timedelta( days = 1 ) ).strftime('%Y-%m-%d' )

        # Get new data from the last date to now
        new_df = self._getMarketData()
        existing_df = existing_df.iloc[ :-1 ]
        # Combine the data, ensuring no duplicates
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates()

        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        # Save and overwrite the file in the same location
        combined_df.to_csv(path)

        print(f"{self.symbol} data updated and stored at {path}")

        return combined_df

    def writeOfflineCSV(self, df=None):
        """ write the data to the offline path """

        if df is None:
            df = self._getMarketData()

        today = dt.datetime.today().strftime('%Y%m%d')  # format as YYYYMMDD
        lastTimestamp = str(df.index[-1])

        if self.futures:
            path = self.OfflinePath + self.symbol + '_' + self.time_frame + '_' + lastTimestamp  # name unchanged for futures
        else:
            path = self.OfflinePath + self.symbol + '_' + self.time_frame + '_' + 'spot_' + lastTimestamp  # add "spot" in the name
        df.to_csv(path + '.csv', index=True)

        print(f"{self.symbol} stores at {self.OfflinePath}")
