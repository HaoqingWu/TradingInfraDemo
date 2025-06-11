'''
Binance Market Data Loader

This script provides a class to fetch and manage OHLCV and funding rate data (both spot and perpetual futures)
through Binance Python API.


Required Packages:
- pandas (>=1.5.0)
- polars (>=1.30.0)
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

import os
import pandas as pd
import polars as pl
import datetime as dt

from binance.client import Client
import glob


class CryptoMarketData:
    def __init__( self, symbol: str, 
                  time_frame: str, 
                  start_date: str = None, 
                  end_date = None, 
                  offline_path = None,
                  futures = False ):
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
        self.client     = Client( '', '' )  # API keys can be set here if needed

        if not offline_path:
            self.OfflinePath = os.path.join(os.path.dirname(__file__), 'OfflineData')
            if not os.path.exists(self.OfflinePath):
                os.makedirs(self.OfflinePath)
                print(f"Created directory: {self.OfflinePath}")
        else:
            self.OfflinePath = offline_path


    def __getFundingRate( self ) -> pl.DataFrame:
        """ Pull historical funding rates using Polars """

        if not self.start_date:
            raise ValueError("Please specify a start date to retrieve funding rate!")

        start_date_timestamp = int(pd.to_datetime(self.start_date).timestamp() * 1000)  # in ms
        cur_timestamp = int(pd.to_datetime(dt.datetime.now(dt.timezone.utc)).timestamp() * 1000)

        limit = 900  # the maximum amount of funding rates the API can pull
        eight_hrs_in_ms = 8 * 3600 * 1000
        
        # Collect all data first
        all_data = []

        while start_date_timestamp < cur_timestamp:
            df_temp = self.client.futures_funding_rate(symbol=self.symbol,
                                                startTime=start_date_timestamp,
                                                endTime=cur_timestamp,
                                                limit=limit)
            
            if df_temp:  # Check if data is returned
                all_data.extend(df_temp)
                
                # Get the last funding time for next iteration
                last_funding_time = df_temp[-1]['fundingTime']
                start_date_timestamp = last_funding_time + eight_hrs_in_ms
            else:
                break

        if not all_data:
            return pl.DataFrame()  # Return empty DataFrame if no data

        # Convert to Polars DataFrame
        df = pl.DataFrame(all_data)
        
        # Convert timestamp to datetime and process columns
        df = df.with_columns([
            # Convert fundingTime from milliseconds to datetime
            pl.from_epoch( pl.col( "fundingTime" ), time_unit = "ms" )
              .alias("Open Time"),
              
            # Convert fundingRate to float
            pl.col("fundingRate").cast( pl.Float64 )
        ]).select([
            "Open Time",
            "fundingRate"
        ])

        return df

    def _getMarketData(self) -> pl.DataFrame:
        """ pull historical klines from binance """

        # Get klines
        if self.futures:
            klines = self.client.futures_historical_klines( self.symbol, self.time_frame, self.start_date, self.end_date )
        else:
            klines = self.client.get_historical_klines( self.symbol, self.time_frame, self.start_date, self.end_date )
        
        if not klines:
            raise ValueError(f"No data found for {self.symbol} in the specified date range.")
        
        col_names = [ 'Open Time',
                      'Open',
                      'High',
                      'Low',
                      'Close',
                      'Volume',
                      # volume in based asset, e.g., for 'BTCUSDT', the volume means the number of BTC traded in that kline
                      'Close Time',
                      'Quote Asset Volume',
                      'Number of Trades',
                      'Taker Buy Base',  # total base asset volume are contributed by the taker buy orders
                      'Taker Buy Quote', # the total value of the taker buy trades in terms of USDT (the quote currency)
                      'Ignore',          # don't know what Ignore does
                    ]
        
        df = pl.DataFrame( klines, schema = col_names, orient = "row" )

        # Process the data using Polars expressions
        df = df.with_columns([
            # Convert Open Time from milliseconds to datetime
            pl.from_epoch( pl.col( "Open Time" ), time_unit = "ms" ),

            # Convert numeric columns to float
            pl.col( "Open" ).cast( pl.Float64 ),
            pl.col( "High" ).cast( pl.Float64 ),
            pl.col( "Low" ).cast( pl.Float64 ),
            pl.col( "Close" ).cast( pl.Float64 ),
            pl.col( "Volume" ).cast( pl.Float64 ),
            pl.col( "Number of Trades" ).cast( pl.Float64 ),
            pl.col( "Taker Buy Base" ).cast( pl.Float64 ),
            pl.col( "Taker Buy Quote" ).cast( pl.Float64 ),
            
            # Calculate Taker_Avg_Cost
            ( pl.col( "Taker Buy Quote" ) / pl.col( "Taker Buy Base" ) ).alias( "Taker_Avg_Cost" )
        ]).select([
            # Select only the columns we need
            "Open Time", "Open", "High", "Low", "Close", "Volume",
            "Number of Trades", "Taker Buy Base", "Taker_Avg_Cost"
        ])

        # Get funding rate data
        df_funding_rate = self.__getFundingRate()

        # Join with funding rate data
        if not df_funding_rate.is_empty():
            df = df.join( df_funding_rate, on = "Open Time", how = "left" )
            
            # Forward fill funding rate
            df = df.with_columns([
                pl.col("fundingRate").forward_fill().alias("fundingRate")
            ] )
        else:
            # Add empty fundingRate column if no funding data
            df = df.with_columns([
                pl.lit(None, dtype = pl.Float64 ).alias("fundingRate")
            ])

        # Set Open Time as index
        df = df.set_sorted( "Open Time" )

        return df

    def getDataDf(self) -> pl.DataFrame:
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

    def updateOfflineCSV( self ) -> pl.DataFrame:
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

            # Write using Polars
            new_df.write_csv(os.path.join(path, f"{self.symbol}_{self.time_frame}_Main.csv"))
            print( f"Main CSV {self.symbol} data stored at {path}" )

            return new_df

        # Use the first matching file
        file_path = matching_files[ 0 ]

        # Load existing data with Polars
        try:
            existing_df = pl.read_csv( file_path, try_parse_dates = True )
            
            # Ensure Open Time is datetime type
            if existing_df.get_column("Open Time").dtype != pl.Datetime:
                existing_df = existing_df.with_columns([
                    pl.col( "Open Time" ).str.to_datetime().alias( "Open Time" )
                ])
                
        except Exception as e:
            print( f"Error reading existing CSV: {e}" )
            # If there's an error reading, treat as new file
            self.start_date = "2018-01-01"
            new_df = self._getMarketData()
            new_df.write_csv( file_path )
            print(f"Main CSV { self.symbol } data recreated at { path }")
            return new_df

        # Get the last date from the existing data
        if existing_df.is_empty():
            self.start_date = "2018-01-01"
        else:
            last_date = existing_df.get_column("Open Time")[-1]
            
            # Convert to Python datetime if it's a Polars datetime
            if hasattr(last_date, 'date'):
                last_date_py = last_date.date()
            else:
                # Handle case where it might be a different datetime format
                last_date_py = pd.to_datetime( last_date ).date()
            
            # Update the start_date to the last date minus 1 day
            self.start_date = ( last_date_py - dt.timedelta( days = 1 )).strftime( '%Y-%m-%d' )

        # Get new data from the last date to now
        new_df = self._getMarketData()
        
        if new_df.is_empty():
            print(f"No new data available for {self.symbol}")
            return existing_df

        # Remove the last row from existing data to avoid overlap
        existing_df_trimmed = existing_df.slice(0, -1) if not existing_df.is_empty() else existing_df
        
        # Combine the data using Polars concat
        combined_df = pl.concat( [ existing_df_trimmed, new_df ] )
        
        # Remove duplicates based on Open Time and keep the first occurrence
        combined_df = combined_df.unique(subset=["Open Time"], keep="first")
        
        # Sort by Open Time to ensure proper ordering
        combined_df = combined_df.sort("Open Time")
        
        # Write the combined data back to the file
        combined_df.write_csv(file_path)
        
        print(f"{self.symbol} data updated and stored at {file_path}")
        
        return combined_df
