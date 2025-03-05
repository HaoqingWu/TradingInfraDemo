"""
Backtesting Engine for my crpto trading strategies
Author: Haoqing Wu
Date: 2025-01-26

Description:
This script implements a backtesting engine designed to simulate and evaluate trading strategies. 
It supports multiple tradable instruments, leverages, and order types (OPEN/CLOSE). The engine 
tracks positions, calculates PnL, and provides visualization tools for analyzing strategy performance.

Key Features:
1. **Order Management**: Supports market/limit orders with customizable leverage, size, and duration.
2. **Trade Tracking**: Trades are managed with detailed attributes (entry price, direction, size, etc.).
3. **Portfolio Management**: Tracks cash, margin, notional value, and leverage across multiple instruments.
4. **Performance Metrics**: Computes PnL, Sharpe ratio, max drawdown, and CVaR.
5. **Visualization**: Thanks to Plotly. Provides equity curves, trade entry/exit plots, and daily PnL charts.
"""
import pandas as pd
import numpy as np
import datetime as dt
from tqdm.auto import tqdm
import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go

from typing import Optional, Union
from enum import Enum
from dataclasses import dataclass, field
import copy

# Configure logging
logging.basicConfig( level = logging.INFO, 
                     format = "%(asctime)s - %(levelname)s - %(message)s")

# Global Constants
INITIAL_CASH    = 1e6
BPS             = 0.0001 # 1 bp
TRANSACTION_FEE = 2 * BPS

# Define enums for TradeStatus, Direction and OrderType
class TradeStatus( Enum ):
    OPEN = 'OPEN'
    CLOSED = 'CLOSED'

class Direction( Enum ):
    LONG = 'LONG'
    SHORT = 'SHORT'

class OrderType( Enum ):
    """ whether an order is to open or close a trade """
    OPEN  = 'OPEN'
    CLOSE = 'CLOSE'


@dataclass
class Order:
    """
    A class representing a trading order.
    Attributes:
        price (float): The price at which the order is placed.
        direction (Direction): The direction of the order, either 'LONG' or 'SHORT'.
        type(str): Open or Close a trade, either 'OPEN' or 'CLOSE'.
        leverage (int): The leverage applied to the order.
        size (float): The size of the order (non-negative float).
        duration (Union[int, None]): The duration of the order in time bars or None for GTC (Good-til-Cancelled).
    """
    imnt: str
    price: float
    direction: Direction
    type: OrderType
    leverage: int
    size: float
    cur_time: dt.datetime
    duration: Union[ int, None ] 

    @property
    def notional( self ) -> float:
        """
        Returns the notional value of the order.
        """
        return self.price * self.size

    @property
    def margin( self ) -> float:
        """
        Returns the margin of the trade based on the entry price, size and leverage.
        """
        return self.price * self.size / self.leverage
    
    @property
    def info(self) -> str:
        """
        Returns a string summarizing the order's details.
        """
        fields = [f"{field.name}: {getattr(self, field.name)}" for field in self.__dataclass_fields__.values()]
        return " | ".join(fields)
    
    def orderFilledToTrade( self, open_time: pd.Timestamp ) -> 'Trade':
        """ transform an order to a trade if filled """

        trade = Trade( imnt = self.imnt, 
                       open_time = open_time, 
                       entry_price = self.price, 
                       direction = self.direction, 
                       size = self.size, 
                       leverage = self.leverage,
                       trade_status = TradeStatus.OPEN )
        return trade

class OrderTicket( dict[str, list[Order]] ):
    """
    Manages trading orders as a dictionary.

    Inherits from:
        dict: where keys are tradable instruments and values are lists of `Order` objects.

    Attributes:
        Inherited from dict.
    """

    def __init__( self, *args, **kwargs ):
        """
        Initialize the OrderTicket. Accepts initial data if provided.
        """
        super().__init__(*args, **kwargs)
        # super( OrderTicket, self ).__init__( *args, **kwargs )

    def add_order( self, order: Order ):
            """
            Add an order to the order book for a given instrument.
            Parameters:
                imnt (str): The tradeable instrument identifier.
                order (Order): An Order object to be added.
            """
            if order.imnt not in self:
                self[ order.imnt ] = []
            self[ order.imnt ].append( order )
            print(f"{order.cur_time}: Added order: {order.info}")

    def remove_order( self, order: Order ):
        """
        Remove an order for a given instrument. If no price, direction, or size is given, remove all orders for that instrument.
        Parameters:
            order (Order): The Order object to be removed.
        """
        imnt = order.imnt
        if imnt not in self:
            print(f"No existing orders for {imnt} to remove.")
            return
        
        if order in self[ imnt ]:
            self[ imnt ].remove( order )
            print(f"{order.cur_time}: Order removed for {imnt}.")
        else:
            print(f"No matching orders found for {imnt}.")

        if not self[imnt]:
            del self[imnt]

    def update_orders( self, ) -> None:
        """
        Update the duration of all active orders.
        - Decrease the duration of orders that have a finite duration.
        - Remove orders with duration <= 0.
        - Leave GTC orders (duration=None) unchanged.
        """
        imnts_to_remove = []
        for imnt, orders in list( self.items() ):
            updated_orders = []
            for order in orders:
                if order.duration is None:
                    # GTC order remains active indefinitely
                    updated_orders.append( order )
                else:
                    # Reduce the duration for time-limited orders
                    new_duration = order.duration - 1
                    if new_duration > 0:
                        updated_orders.append( Order( order.imnt, order.price, order.direction, order.type, order.leverage, order.size, order.cur_time, new_duration) )
                    elif new_duration == 0:
                        print(f"{imnt} order: at Price {order.price} has expired and will be removed.")
            # If there are still active orders, update the list
            if updated_orders:
                self[ imnt ] = updated_orders
            else:
                imnts_to_remove.append( imnt )

        # Remove instruments with no active orders
        for imnt in imnts_to_remove:
            del self[imnt]
        
    def aggregate_orders( self, newOrderTicket: 'OrderTicket' ) :
        """ aggregate the new order ticket to the existing order ticket """

        for imnt in newOrderTicket:
            for order in newOrderTicket[ imnt ]:
                self.add_order( order = order )
        

@dataclass
class Trade:
    """
    A class to represent an opened trade.

    Attributes:
        imnt (str): The symbol of the instrument being traded (e.g., 'BTCUSDT').
        open_time (datetime): The timestamp when the trade was opened.
        entry_price (float): The average price at which the trade was entered.
        direction (str): The direction of the order, either 'LONG' or 'SHORT'.
        size (float): The size of the trade (non-negative float).
        leverage (int): The leverage applied to the trade.
        trade_status (str): The current status of the trade ('OPEN' or 'CLOSED').
        closed_pnl( float ): PnL of the closed position.
        open_pnl( float ): MtM PnL of open position.
        close_time( float ): trade closing time
    """

    imnt: str
    open_time: dt.datetime
    entry_price: float
    direction: Direction
    size: float                 # size is non-negative
    leverage: int
    trade_status: TradeStatus = TradeStatus.OPEN  # Defaults to 'open' when the trade is created
    closed_pnl: float = 0
    open_pnl: float = 0
    close_time: Optional[ dt.datetime ] = None

    def __post_init__(self):
        if self.size < 0:
            raise ValueError( "Size must be non-negative" )
        
    @property
    def notional( self ) -> float:
        """
        Returns the notional value of the trade.
        """
        return self.entry_price * self.size

    @property
    def margin( self ) -> float:
        """
        Returns the margin of the trade based on the entry price, size and leverage.
        """
        return self.entry_price * self.size / self.leverage
    
    @property
    def info(self) -> str:
        """
        Returns a string summarizing the trade's details.
        """
        fields = [ f"{field.name}: {getattr(self, field.name)}" for field in self.__dataclass_fields__.values() ]
        return " | ".join(fields)
    
    def add_position( self, open_price: float, open_size: float ) -> None:
        """
        add more positions to this exisiting trade; recalculate entry price

        Args:
            open_price (float): The price at which the position is added.
            open_size (float):  The size of the closing trade.
        """
        new_entry_price  = ( open_price * open_size + self.entry_price * self.size ) / ( open_size + self.size )
        self.entry_price = new_entry_price
        self.size       += open_size

    def close_position( self, close_price: float, close_size: float, close_time: dt.datetime ) -> None:
        """
        Close or partially clode a trade and calculate its pnl.

        Args:
            close_price (float): The price at which the trade is closed.
            close_size (float): The size of the closing trade.
            close_time (datetime): The timestamp when the trade is closed.

        Returns:
            float: The total profit or loss for the trade.
        """

        if self.size <= close_size:
            self.trade_status = TradeStatus.CLOSED
            close_size = self.size
            if self.direction == Direction.LONG:
                self.closed_pnl += ( close_price - self.entry_price ) * close_size
            else:
                self.closed_pnl += ( self.entry_price - close_price ) * close_size
            self.open_pnl = 0
            self.size     = 0
            self.close_time = close_time            
            print( f"{ close_time }: {self.imnt} trade closed with PnL: { self.closed_pnl }")
        else: # partial close
            if self.direction == Direction.LONG:
                self.closed_pnl += ( close_price - self.entry_price ) * close_size
            else:
                self.closed_pnl += ( self.entry_price - close_price ) * close_size
            if abs( self.open_pnl ) < abs( self.closed_pnl ):
                raise ValueError( "Something is wrong: Closed PnL > open PnL!" )
            self.open_pnl   -= self.closed_pnl
            self.size       -= close_size
            

    def update_trade( self, order: Order ) -> None:
        # TODO: this has never been used
        """ update the trade """
        if self.direction != order.direction:
            raise ValueError( "The direction of the trade and order do not match!" )
        if order.type == OrderType.OPEN:
            self.add_position( order.price, order.size )
        else:
            self.close_position( order.price, order.size, order.cur_time )

    def _update_unrealized_pnl( self, cur_price: float ) -> None:
        """ update the current MtM pnl of the trade """
        if self.direction == Direction.LONG:
            self.open_pnl = ( cur_price - self.entry_price ) * self.size  
        else:
            self.open_pnl = ( self.entry_price - cur_price ) * self.size
        print(f"Updated trade: {self.direction.value} {self.imnt}. Open PnL: {self.open_pnl}")


class PositionManager:
    """
    A class to manage multiple trades for ONE instrument.
    It ensures that at most one LONG and one SHORT trade is open per instrument.
    """
    def __init__( self, imnt: str, positions: list[ Trade ] = [] ):
        self.imnt: str = imnt
        self.positions: list[ Trade ] = positions if positions else []

    def process_new_trade( self, new_trade: Trade ) -> None:
        """
        Processes an order to either open, update, or close a position for the instrument.

        Args:
            new_trade (Trade): The incoming trade to process.
        """
        if self.positions:
            cur_position = self.positions[ -1 ] 
        else:
            cur_position = None
        
        # if there is an existing open position
        if cur_position and cur_position.trade_status == TradeStatus.OPEN:
            if cur_position.direction == new_trade.direction:
                cur_position.add_position( new_trade.entry_price, new_trade.size )
            else:
                cur_position.close_position( new_trade.entry_price, new_trade.size, new_trade.open_time )
        
        # open a new position
        elif ( not cur_position ) or ( cur_position.trade_status == TradeStatus.CLOSED ): # no open position
            self.positions.append( new_trade )


    def _update_position_unrealized_pnl(self, current_price: float) -> None:
        """
        Updates the unrealized PnL for both LONG and SHORT trades at the current market price.

        Args:
            current_price (float): The current price of the instrument.
        """
        try:
            cur_trade = self.positions[ -1 ]
        except IndexError:
            return None
        if cur_trade.trade_status == TradeStatus.OPEN:
            cur_trade._update_unrealized_pnl( current_price )


@dataclass
class BacktestStatus:
    """
    A dataclass to manage the state during backtesting, including positions, prices, PnLVector, cash, and order tickets.

    Attributes:
        tradables (list[str]): A list of tradable instruments.
        initial_cash (float): The initial cash available for trading.

        cur_positions (dict[str, PositionManager]): A dictionary that stores the current positions for each instrument.
        cur_priceVector (dict[str, float]): A dictionary that stores the current price for each instrument.
        cur_PnLVector (dict[str, float]): A dictionary that stores the current cumulative PnL for each instrument.
        cur_cash (float): The current cash available for trading.
        cur_portfolioMtMValue (float): The current market-to-market value of the portfolio.
        cur_orderTicket (OrderTicket): The current order ticket that stores the trading orders.
        cur_total_margin (float): The current margin used for open positions.
        cur_total_notional (float): The current notional value of open positions.
        cur_total_leverage (float): The current total leverage of the portfolio.
    """
    tradables: list[str]
    initial_cash: float

    cur_positions: dict[str, 'PositionManager'] = field(init=False)
    cur_priceVector: dict[str, float] = field(init=False)
    cur_PnLVector: dict[str, float] = field(init=False)
    cur_cash: float = field( init = False )
    cur_fundingCost: dict[str, float] = field( init=False )
    cur_portfolioMtMValue: float = field(init=False)
    cur_orderTicket: 'OrderTicket' = field(init=False)
    cur_total_margin: float = field(init=False)
    cur_total_notional: float = field(init=False)
    cur_total_leverage: float = field(init=False)

    def __post_init__(self):
        """Initializes the default state for positions, price vector, PnL vector, etc."""
        self.cur_positions = {imnt: PositionManager(imnt) for imnt in self.tradables}
        self.cur_priceVector = {imnt: 0.0 for imnt in self.tradables}
        self.cur_PnLVector = {imnt: 0.0 for imnt in self.tradables}
        self.cur_cash = self.initial_cash
        self.cur_fundingCost = { imnt: 0.0 for imnt in self.tradables }
        self.cur_portfolioMtMValue = self.initial_cash
        self.cur_orderTicket = OrderTicket()
        self.cur_total_margin = 0.0
        self.cur_total_notional = 0.0
        self.cur_total_leverage = 0.0

    def update_margin_and_notional( self ):
        """Updates the margin, notional value, and total leverage of the portfolio."""
        total_margin = 0.0
        total_notional = 0.0

        for imnt, position_manager in self.cur_positions.items():
            for trade in position_manager.positions:
                if trade.trade_status == TradeStatus.OPEN:
                    total_margin += trade.margin
                    total_notional += self.cur_priceVector[ imnt ] * trade.size

        self.cur_total_margin = total_margin
        self.cur_total_notional = total_notional
        self.cur_total_leverage = total_notional / total_margin if total_margin > 0 else 0.0

    def __str__(self):
        """
        Returns a string representation of the current status.
        """
        msg = (f"Current Positions: {self.cur_positions}, "
               f"Current Prices: {self.cur_priceVector}, "
               f"Current PnLs: {self.cur_PnLVector}, "
               f"Current Cash: {self.cur_cash}, "
               f"Current Funding Cost Paid: {self.cur_fundingCost}, "
               f"Current Portfolio MtM Value: {self.cur_portfolioMtMValue}, "
               f"Current Order Ticket: {self.cur_orderTicket}, "
               f"Current Margin: {self.cur_total_margin}, "
               f"Current Notional: {self.cur_total_notional}, "
               f"Current Total Leverage: {self.cur_total_leverage}.")
        return msg
    
    def __deepcopy__(self, memo ):
        """
        Returns a deep copy of the current status.
        """
        new_copy = BacktestStatus(
            tradables=self.tradables,
            initial_cash=self.initial_cash,
        )
        new_copy.cur_positions = {imnt: copy.deepcopy(pos) for imnt, pos in self.cur_positions.items()}
        new_copy.cur_priceVector = self.cur_priceVector.copy()
        new_copy.cur_PnLVector = self.cur_PnLVector.copy()
        new_copy.cur_cash = self.cur_cash
        new_copy.cur_fundingCost = self.cur_fundingCost.copy()
        new_copy.cur_portfolioMtMValue = self.cur_portfolioMtMValue
        new_copy.cur_orderTicket = copy.deepcopy(self.cur_orderTicket)
        new_copy.cur_total_margin = self.cur_total_margin
        new_copy.cur_total_notional = self.cur_total_notional
        new_copy.cur_total_leverage = self.cur_total_leverage

        return new_copy


class Backtest:
    def __init__( self, dataDict: dict, initial_cash: float = INITIAL_CASH ):
        """ 
        Initialize the BacktestingEngine with tradables, market data, and initial cash.

        Args:
            tradables (list[str]): A list of tradable instruments.
            dataDict (dict): A dictionary where keys are instrument names and values are dataframes containing market data for each instrument. Each dataframe should have the same shape.
            initial_cash (float, optional): Initial cash available for trading. Defaults to INITIAL_CASH.

        Raises:
            ValueError: If the market data for instruments do not have matching shapes.

        """
        if not isinstance( dataDict, dict ):
            raise ValueError( "dataDict must be a dictionary" )
        
        self._dataDict = dataDict
        self._tradables = dataDict.keys()
        self._initialCash = initial_cash 
        self._availableTimestamps = list(next(iter( self._dataDict.values())).index)
        self._totalTimeSteps = len( self._availableTimestamps )
        self._timeStampCounter = 0
        
        if not self.__checkShape():
            raise ValueError("Mismatch of instrument's market data")
        
    def __checkShape( self ) -> bool:
        """ check if the market data of each instrument has the same shapes """
        shapes = [ df.shape for df in self._dataDict.values() ]
        return all( shape == shapes[ 0 ] for shape in shapes )
    

    ########################
    ### run back-testing ###
    def getCurTime( self, ) -> pd.Timestamp:
        """ get current timestamp """
        return self._availableTimestamps[ self._timeStampCounter ]
    

    def getNextTime( self, ) -> pd.Timestamp:
        """ get next timestamp """
        if self._timeStampCounter + 1 == self._totalTimeSteps:
            return None
        else:
            return self._availableTimestamps[ self._timeStampCounter + 1 ]
    
    def __incrementTime( self ) -> None:
        """ increment the timestamp counter """
        self._timeStampCounter += 1
    
    def _getNextOpenPrice( self, imnt ) -> np.ndarray:
        nextTimeIndex = self._timeStampCounter + 1
        nextOpenPrice = self._dataDict[ imnt ][ 'Open' ].iloc[ nextTimeIndex ] 
        return nextOpenPrice
    
    def _getNextHighPrice( self, imnt ) -> np.ndarray:
        nextTimeIndex = self._timeStampCounter + 1
        nextHighPrice = self._dataDict[ imnt ][ 'High' ].iloc[ nextTimeIndex ] 
        return nextHighPrice
    
    def _getNextLowPrice( self, imnt ) -> np.ndarray:
        nextTimeIndex = self._timeStampCounter + 1
        nextLowPrice = self._dataDict[ imnt ][ 'Low' ].iloc[ nextTimeIndex ] 
        return nextLowPrice
    
    def _getNextClosePrice( self, imnt ) -> np.ndarray:
        nextTimeIndex = self._timeStampCounter + 1
        nextClosePrice = self._dataDict[ imnt ][ 'Close' ].iloc[ nextTimeIndex ] 
        return nextClosePrice 
    
    def _getNextVolume( self, imnt ) -> np.ndarray:
        nextTimeIndex = self._timeStampCounter + 1
        nextVolumePrice = self._dataDict[ imnt ][ 'Volume' ].iloc[ nextTimeIndex ] 
        return nextVolumePrice 
    
    def _getNextFundingRate( self, imnt ) -> float:
        nextTimeIndex   = self._timeStampCounter + 1
        nextFundingRate = self._dataDict[ imnt ][ 'fundingRate' ].iloc[ nextTimeIndex ]
        return nextFundingRate
    
    def _checkLiquidation( self ) -> bool:
        """ check if the current portfolio needs liquidation """
        cur_cash = self._status.cur_cash
        cur_total_margin = self._status.cur_total_margin    
        cur_positions = self._status.cur_positions
        cur_totalPnL = sum( [ trade.open_pnl for imnt in cur_positions for trade in cur_positions[ imnt ].positions ] )

        if cur_cash + cur_total_margin + cur_totalPnL < 0:
            return True
        else:
            return False
    
    def getCurrentAvailableData( self, lookBack: int = None ) -> dict:
        """ get observed data from t - `lookBack` to t """
        curTimeIndex = self._timeStampCounter
        curTime      = self.getCurTime()

        lookBackStartIndex = max(0, curTimeIndex - lookBack) if lookBack else 0
        
        marketDataDict = { imnt: self._dataDict[ imnt ].iloc[ lookBackStartIndex: curTimeIndex + 1 ] for imnt in self._tradables }

        resDict = { 
                "marketData": marketDataDict,
                "status": getattr(self, '_status', None)  # Check if self has attribute _status
            }
        return resDict
 
    def initializeStrategy( self ) -> Optional[ None ]:
        """ initialize the strategy """
        
        return None
    
    # Overwrite this function to implment your own strategy
    def myStrategy( self ) -> OrderTicket:
        """ return an `OrderTicket` object of the current strategy """

        return NotImplementedError( "You need to override the myStrategy function." )
    
    
    def runStrategy( self, startTime = None, endTime = None, progressBar = True, verbose = False ) -> dict:
        """
        Executes a backtesting process over the specified time range, optionally displaying
        a progress bar and providing verbose output. The function initializes the backtesting
        environment, optionally calls a user-defined strategy initialization method, and then
        iteratively applies the user-defined strategy to generate trades. It updates the internal
        status of the backtesting after each iteration and returns a dictionary of status snapshots.
        Args:
            startTime (datetime.date | pandas.Timestamp | None, optional):
                The start time for the backtesting. If a date is provided, it is converted
                to a pandas Timestamp. If None, uses the earliest available timestamp.
            endTime (datetime.date | pandas.Timestamp | None, optional):
                The end time for the backtesting. If a date is provided, it is converted
                to a pandas Timestamp. If None, uses the latest available timestamp.
            progressBar (bool, optional):
                Whether to display a progress bar during the backtesting. Defaults to True.
            verbose (bool, optional):
                Whether to display detailed logs for each iteration. Defaults to False.
        Returns:
            dict:
                A dictionary keyed by timestamps. Each entry contains a snapshot of the
                backtesting status, including holdings, cash balance, and any relevant
                performance metrics.
        """

        if ( isinstance( startTime, dt.date ) ) or ( isinstance( endTime, dt.date ) ):
            startTime = pd.Timestamp( startTime )
            endTime   = pd.Timestamp( endTime )

        # initialize the status dictionary
        self._status = BacktestStatus( tradables = self._tradables, initial_cash = self._initialCash )

        userInitializationFunc = getattr( self, "initializeStrategy", None )
        if callable( userInitializationFunc ):
            self.initializeStrategy()
            print( "User defined initialization method recognized: initiating strategy." )

        # if statTime and endTime are not in __availableTimestamps, then we find the closest timestamp
        startTime      = min( self._availableTimestamps, key = lambda x: abs( startTime - x ) )
        endTime        = min( self._availableTimestamps, key = lambda x: abs( endTime - x ) )
        startTimeIndex = self._availableTimestamps.index( startTime )
        endTimeIndex   = self._availableTimestamps.index( endTime )

        # move the timestamp counter to startTimeIndex
        self._timeStampCounter = startTimeIndex

        # initialize the cur_price vector in the status dictionary
        for imnt in self._tradables:
            self._status.cur_priceVector[ imnt ] = self._dataDict[ imnt ][ 'Close' ].iloc[ startTimeIndex ]
        status_dict = {}
        
        # backtesting starts here
        for i, timeStamp in tqdm( enumerate( range( startTimeIndex, endTimeIndex + 1 ) ), total = endTimeIndex - startTimeIndex + 1 ):
            # myStrategy function will be run in this for loop and return an OrderTicket object every time
            newOrderTicket = self.myStrategy()  
            try:
                # nextTimeIndex = self._timeStampCounter + 1
                self._updateStatus( newOrderTicket, verbose = verbose )
                status_temp = copy.deepcopy( self._status )
                status_dict[ self._availableTimestamps[ timeStamp ] ] = status_temp
                if timeStamp != endTimeIndex - 1:
                    self.__incrementTime()

            except Exception as e:
                logging.info( f"Error at {timeStamp}" )
                logging.error( e )
                break

        return status_dict
        
    def _executeOrders( self, 
                        orderTicket_t: OrderTicket, 
                        positions_t: dict, 
                        cash_t: float, 
                        curTime: pd.Timestamp ) -> None:
        """ 
        execute the orders in the order ticket at time t.
        return the updated positions and cash at time t+1.
        
        Args:
            orderTicket_t (OrderTicket): The order ticket at time t.
            positions_t (dict): The positions at time t.
            cash_t (float): The cash at time t.
            curTime (pd.Timestamp): The current timestamp.
        """

        # Execution
        # execute the order in the keys of `orderTicket`
        for imnt in orderTicket_t.copy():
            openPrice_tPlus1    = self._getNextOpenPrice( imnt )  
            highPrice_tPlus1    = self._getNextHighPrice( imnt )
            lowPrice_tPlus1     = self._getNextLowPrice( imnt )
            closePrice_tPlus1   = self._getNextClosePrice( imnt ) 

            # PnL for existing position = existing position * price change
            # curPnL[ imnt ]      = positions[ imnt ] * ( closePrice_tPlus1 - openPrice_tPlus1 ) 
            if imnt not in self._tradables:
                raise ValueError( f"{imnt} is not a tradable!" )
            
            for order in orderTicket_t[ imnt ]:
                leverage = order.leverage
                notional = order.notional
                margin   = order.margin
                order_type = order.type

                # check if size of an order is negative or zero
                if order.size <= 0:
                    raise ValueError( f"Size of the order {order} is negative or zero." )

                ## Execution Logic
                # 0. if execution price is between Low and High of next candle => filled
                # 1. make sure enough cash to execute OR
                # 2. enough position to exit
                # 3. Update funding

                # open a new trade
                if order_type == OrderType.OPEN and order.price <= highPrice_tPlus1 and order.price >= lowPrice_tPlus1:
                    if cash_t >= ( margin + notional * TRANSACTION_FEE ):
                        trade = order.orderFilledToTrade( curTime )
                        positions_t[ imnt ].process_new_trade( trade )

                        cash_t -= ( margin + notional * TRANSACTION_FEE )

                        orderTicket_t.remove_order( order )
                        logging.info( f" Order { order.info } executed." )
                    else:
                        logging.info( f"Not enough cash to execute the trade {order}." )
                        continue

                # close or close partially an existing trade
                elif order_type == OrderType.CLOSE: 
                    if not positions_t[ imnt ].positions: 
                        logging.info( f"No position on {imnt} to close!" )
                        orderTicket_t.remove_order( order )
                        continue

                    last_trade = positions_t[ imnt ].positions[ -1 ]
                    if last_trade.trade_status == TradeStatus.CLOSED:
                        logging.info( f"Position on {imnt} is already closed!" )
                        orderTicket_t.remove_order( order )
                        continue
                    elif order.direction == Direction.LONG and last_trade.direction == Direction.LONG:
                        logging.info( f"Cannot close a LONG position with a LONG order." )
                        orderTicket_t.remove_order( order )
                        continue
                    elif order.direction == Direction.SHORT and last_trade.direction == Direction.SHORT:
                        logging.info( f"Cannot close a SHORT position with a Short order." )
                        orderTicket_t.remove_order( order )
                        continue
                    elif order.leverage != last_trade.leverage:
                        logging.info( f"Cannot close a position with different leverage." )
                        orderTicket_t.remove_order( order )
                        continue
                    else: # now we can close the position
                        cur_size = last_trade.size
                        if order.size > abs( cur_size ):
                            logging.info( f"{order.size} is larger than the current position, it'll get truncated to current position." )
                            order.size = abs( cur_size )
                        if order.price <= highPrice_tPlus1 and order.price >= lowPrice_tPlus1:
                            trade = order.orderFilledToTrade( curTime )
                            positions_t[ imnt ].process_new_trade( trade )

                            # need to be careful handling the cash here
                            initial_trade_margin = last_trade.entry_price * trade.size / trade.leverage
                            if last_trade.direction == Direction.LONG:
                                closed_pnl = ( order.price - last_trade.entry_price ) * trade.size
                            else: # SHORT trade pnl
                                closed_pnl = ( last_trade.entry_price - order.price ) * trade.size
                            cash_t += ( initial_trade_margin - trade.notional * TRANSACTION_FEE + closed_pnl )
                            # Update the open PnL of the position
                            # We don't need to single out when status is closed, as its size will be 0
                            if last_trade.trade_status == TradeStatus.CLOSED:
                                assert last_trade.size == 0
                            
                            orderTicket_t.remove_order( order )

        self._status.cur_cash = cash_t
        self._status.cur_positions = positions_t
        self._status.cur_orderTicket = orderTicket_t


    def _updateFundingCost( self, ) -> None:
        """ update the funding cost of the open positions """

        for imnt in self._status.cur_positions:
            cur_price = self._status.cur_priceVector[ imnt ]
            fundingRate = self._getNextFundingRate( imnt )
            
            # if no open positions, continue
            if not self._status.cur_positions[ imnt ].positions or self._status.cur_positions[ imnt ].positions[ -1 ].trade_status == TradeStatus.CLOSED:
                continue
            
            cur_pos  = self._status.cur_positions[ imnt ].positions[ -1 ]
            notional = cur_pos.notional

            if cur_pos.direction == Direction.LONG:
                if fundingRate > 0:
                    if self._status.cur_cash > abs( notional * fundingRate ):
                        self._status.cur_cash -= notional * fundingRate
                    else:
                        # not enough cash to pay the funding cost, substract the margin which is an estimate
                        cur_pos.size -= notional * fundingRate / cur_price 
                else:
                    self._status.cur_cash -= notional * fundingRate
                
                self._status.cur_fundingCost[ imnt ] -= notional * fundingRate


            else: # SHORT trade
                if fundingRate < 0:
                    if self._status.cur_cash > abs( notional * fundingRate ):
                        self._status.cur_cash += notional * fundingRate
                    else:
                        cur_pos.size += notional * fundingRate / cur_price
                else:
                    self._status.cur_cash += notional * fundingRate

                self._status.cur_fundingCost[ imnt ] += notional * fundingRate

    def _updateStatus( self, newOrderTicket: OrderTicket, verbose: bool = True ) -> None:
        """ during the evolution of states, update the status dict """

        # gather input
        curTime              = self.getCurTime()
        positions_t          = self._status.cur_positions
        cash_t               = self._status.cur_cash
        orderTicket_t        = self._status.cur_orderTicket


        # Update the Order Ticket 
        orderTicket_t.update_orders() # clean up expired orders
        orderTicket_t.aggregate_orders( newOrderTicket )

        self._executeOrders( orderTicket_t, positions_t, cash_t, curTime )

        # Pay funding cost
        if curTime.hour in [0, 8, 16] and curTime.minute == 0:
            self._updateFundingCost()

        # Update the cumulative PnL for each instrument
        for imnt in self._tradables:
            # prev_cumPnL = self._status.cur_PnLVector[ imnt ]
            if positions_t[ imnt ].positions:
                # Update the unrealized/open PnL of the open trades
                positions_t[ imnt ]._update_position_unrealized_pnl( self._status.cur_priceVector[ imnt ] )
                # Update pnl vector
                last_position = positions_t[ imnt ].positions[ -1 ]
                if last_position.trade_status == TradeStatus.OPEN:
                    self._status.cur_PnLVector[ imnt ] = last_position.open_pnl + last_position.closed_pnl

        if cash_t < 0:
            raise ValueError( "Cash cannot be negative!" )        

        # Update the MtM value of the portfolio
        cur_open_trade = {imnt: positions_t[imnt].positions[-1] if positions_t[imnt].positions and positions_t[imnt].positions[-1].trade_status == TradeStatus.OPEN
                                                                else None for imnt in self._tradables}

        imntMtMValue = [ cur_open_trade[imnt].open_pnl + cur_open_trade[imnt].margin if cur_open_trade[imnt] else 0 for imnt in self._tradables]
        self._status.cur_portfolioMtMValue = self._status.cur_cash + np.sum( imntMtMValue )
        self._status.cur_orderTicket       = orderTicket_t
        self._status.update_margin_and_notional()
        self._status.cur_priceVector = { imnt: self._getNextClosePrice( imnt )  for imnt in self._tradables }

        # Check liquidation
        isLiquidated = self._checkLiquidation()
        if isLiquidated:
            raise ValueError( "Portfolio is liquidated!" )


    ##########################
    ### Postprocessing API ###
    def getPnL( self, status_dict: dict ) -> pd.DataFrame:
        """ return the PnL of each time bar from the backtest """
        cum_pnl_dict = { timeStamp: status.cur_PnLVector for timeStamp, status in status_dict.items() }
        cum_pnl_df = pd.DataFrame.from_dict( cum_pnl_dict, orient='index', columns=self._tradables)
        pnl_df = cum_pnl_df.diff().fillna( 0 )

        return pnl_df
    
    def getTradesHistoryDf( self, status_dict: dict ) -> dict[ str, pd.DataFrame ]:
        """ return the trades history of the backtest as a pandas DataFrame """
        lastTimeStamp = next( reversed( status_dict.keys() ) )
        res = {}
        for imnt in status_dict[ lastTimeStamp ].cur_positions: # loop instruments
            res[ imnt ] = {
                'Open Time': [],
                'Close Time': [],
                'Entry Price': [],
                'Direction': [],
                'Close PnL': [],
                'Open PnL': []
            }
            for trade in status_dict[ lastTimeStamp ].cur_positions[ imnt ].positions: # loop trades
                res[ imnt ][ 'Open Time' ].append( trade.open_time )
                res[ imnt ][ 'Entry Price' ].append( trade.entry_price )
                res[ imnt ][ 'Direction' ].append( trade.direction )
                res[ imnt ][ 'Close Time' ].append( trade.close_time )
                res[ imnt ][ 'Close PnL' ].append( trade.closed_pnl )
                res[ imnt ][ 'Open PnL' ].append( trade.open_pnl )
            
            res[ imnt ] = pd.DataFrame( res[ imnt ] )

        return res


    def getDailyPnL( self, status_dict: dict ) -> pd.DataFrame:
        """ return the daily PnL of the backtest as a pandas DataFrame """
        pnl_df = self.getPnL( status_dict )
        daily_pnl_df = pnl_df.resample( 'D' ).sum()
        daily_pnl_df.name = 'Daily PnL'
        return daily_pnl_df
    
    def getPortfolioDailyValues( self, status_dict: dict ) -> pd.Series:
        """ return the portfolio value of the backtest as a pandas Series """
        PortfolioValue = { timeStamp: status.cur_portfolioMtMValue for timeStamp, status in status_dict.items() }
        DailyPortfolioValueSeries = pd.Series( PortfolioValue ).resample( 'D' ).last()
        DailyPortfolioValueSeries.name = 'Portfolio Daily Value'
        return  DailyPortfolioValueSeries
    
    def getTotalMarginSeries( self, status_dict: dict ) -> pd.Series:
        """ return the total margin of the backtest as a pandas Series """
        total_margin_dict = { timeStamp: status.cur_total_margin for timeStamp, status in status_dict.items() }
        total_margin_series = pd.Series( total_margin_dict, name = 'Total Margin' )
        return total_margin_series
    
    def getTotalNotionalSeries( self, status_dict: dict ) -> pd.Series:
        """ return the total notional value of the backtest as a pandas Series """
        total_notional_dict = { timeStamp: status.cur_total_notional for timeStamp, status in status_dict.items() }
        total_notional_series = pd.Series( total_notional_dict, name = 'Total Notional' )
        return total_notional_series
    
    def getTotalLeverageSeries( self, status_dict: dict ) -> pd.Series:
        """ return the total leverage of the backtest as a pandas Series """
        total_leverage_dict = { timeStamp: status.cur_total_leverage for timeStamp, status in status_dict.items() }
        total_leverage_series = pd.Series( total_leverage_dict, name = 'Total Leverage' ) 
        return total_leverage_series
    
    def getPortfolioDailyRelativeReturns( self, status_dict: dict ) -> pd.Series:
        """ return the relative returns of the backtest as a pandas DataFrame """
        daily_MtM_series  = self.getPortfolioDailyValues( status_dict )
        daily_MtM_returns = daily_MtM_series.pct_change()
        daily_MtM_returns.name = 'Daily Relative Returns' 
        return daily_MtM_returns
    
    def computeInstrumentTotalReturn( self, status_dict: dict ) -> pd.Series:
        """ return the total PnL of the backtest """
        pnl_df = self.getPnL( status_dict )
        total_pnl = pnl_df.sum()
        return total_pnl
    
    def computeTotalReturn( self, status_dict: dict ) -> float:
        """ return the total return of the backtest """
        pnl_df = self.getPnL( status_dict )
        total_return = pnl_df.sum().sum()
        return total_return
        
    def computeSharpeRatio( self, status_dict: dict ) -> float:
        """ return the Sharpe Ratio of the backtest """
        daily_returns = self.getPortfolioDailyRelativeReturns( status_dict )
        sharpe_ratio = daily_returns.mean() / daily_returns.std()

        return sharpe_ratio * np.sqrt( 365 )  # annualize the Sharpe Ratio; crypto market is 24/7
    
    def computeMaxDrawdown( self, status_dict: dict ) -> float:
        """ return the maximum drawdown of the backtest """
        PortfolioValueSeries = self.getPortfolioDailyValues( status_dict )
        cum_max = PortfolioValueSeries.cummax()
        drawdown = ( PortfolioValueSeries - cum_max ) / cum_max
        max_drawdown = drawdown.min()

        return max_drawdown


    def computeCVaR( self, status_dict: dict, level: float ) -> float:
        """ return the CVaR of the backtest """
        portfolio_returns = self.getPortfolioDailyRelativeReturns( status_dict )
        n = portfolio_returns.shape[ 0 ]
        level_ordinal = int( n * level )
        res_ = portfolio_returns.sort_values()
        return res_.iloc[ : level_ordinal ].mean()

    def plotEquityCurve(self, status_dict: dict) -> None:
        """ Plot the equity curve of the backtest with dollar formatting on the y-axis and trade markers """
        # Get the portfolio value time series
        PortfolioValueSeries = self.getPortfolioDailyValues(status_dict)

        # Create a plotly figure
        fig = go.Figure()

        # Add the equity curve to the figure
        fig.add_trace(go.Scatter(
            x=PortfolioValueSeries.index,
            y=PortfolioValueSeries.values,
            mode='lines',
            name='Equity Curve',
            hoverinfo='x+y',
            line=dict(color='blue')
        ))


        status = next( reversed( status_dict.values() ) )
        # Plot trades: Retrieve trades from the position managers and plot them on the equity curve
        for imnt, pos_manager in status.cur_positions.items():
            for trade in pos_manager.positions:
                # Plot the entry (open) of the trade
                open_time = pd.Timestamp(trade.open_time.date())
                fig.add_trace(go.Scatter(
                    x=[open_time],
                    y=[PortfolioValueSeries[open_time]],
                    mode='markers',
                    marker=dict(color='green', symbol='circle', size = 5),
                    #name=f'Open {imnt}',
                    hoverinfo='text',
                    text=f'Open {imnt} <br>Open Time: {open_time}<br>Entry Price: {trade.entry_price:.2f}'
                ))
                logging.info(f"Plotted open trade at {trade.open_time} for {imnt}.")

                # Plot the exit (close) of the trade
                if trade.close_time:
                    close_time = pd.Timestamp(trade.close_time.date())
                    fig.add_trace(go.Scatter(
                        x=[close_time],
                        y=[PortfolioValueSeries[close_time]],
                        mode='markers',
                        marker=dict(color='red', symbol = 'x', size = 5),
                        #name=f'Close {imnt}',
                        hoverinfo='text',
                        text=f'Close {imnt} <br>Close Time: {close_time} <br>Closed PnL: {trade.closed_pnl:.2f}'
                    ))
                    logging.info(f"Plotted close trade at {trade.close_time} for {imnt}.")

        # Set y-axis label to emphasize that values are in dollars
        fig.update_layout(
            yaxis=dict(
                title='Dollars',
                tickformat='$,.0f'
            ),
            title = 'Equity Curve of the Strategy',
            hovermode = 'x unified',
            showlegend = False  # Disable the legend
        )

        # Show the plot
        fig.show()


    def plotEntryExitTrades(self, status_dict: dict) -> None:
        """ Plot the entry and exit trades on the price graph of each instrument using Plotly """
        
        # Get the last status to retrieve the positions
        last_status = next(reversed(status_dict.values()))
        TradesHistory = self.getTradesHistoryDf(status_dict)

        startTime = next(iter(status_dict.keys()))
        endTime = next(reversed(status_dict.keys()))

        for imnt in self._tradables:
            # Get the price data for the instrument
            price_data = self._dataDict[imnt]['Close'].loc[ startTime: endTime ]

            # Create a plotly figure
            fig = go.Figure()

            # Add the price data to the figure
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data.values,
                mode='lines',
                name='Price',
                hoverinfo='x+y',
                line=dict(color='blue')
            ))

            # Plot trades: Retrieve trades from the position managers and plot them on the price graph
            for _, trade in TradesHistory[imnt].iterrows():
                # Plot the entry (open) of the trade
                open_time = trade['Open Time']
                fig.add_trace(go.Scatter(
                    x=[open_time],
                    y=[price_data.loc[open_time]],
                    mode='markers',
                    marker=dict(color='green', symbol='circle', size=5),
                    name='Open',
                    hoverinfo='text',
                    text=f'Open {imnt} <br>Open Time: {open_time}<br>Entry Price: {trade["Entry Price"]:.2f}'
                ))

                # Plot the exit (close) of the trade
                close_time = trade['Close Time']
                if pd.notna(close_time):
                    fig.add_trace(go.Scatter(
                        x=[close_time],
                        y=[price_data.loc[close_time]],
                        mode='markers',
                        marker=dict(color='red', symbol='x', size=5),
                        name='Close',
                        hoverinfo='text',
                        text=f'Close {imnt} <br>Close Time: {close_time} <br>Closed PnL: {trade["Close PnL"]:.2f}'
                    ))

            # Set y-axis label to emphasize that values are in dollars
            fig.update_layout(
                yaxis=dict(
                    title='Price',
                    tickformat='$,.2f'
                ),
                title=f'Price and Trades for {imnt}',
                hovermode='x unified',
                showlegend=True
            )

            # Show the plot
            fig.show()

    def plotPnL( self, status_dict: dict ) -> None:
        """ plot the PnL of the backtest """
        pnl_df = pd.DataFrame( { timeStamp: status.cur_PnLVector for timeStamp, status in status_dict.items() } ).T
        pnl_df.plot( title = "PnL of the Backtest" )
        plt.show()

    def plotDailyPnL( self, status_dict: dict ) -> None:
        """ Plot each instrument's daily PnL as a stacked bar chart with different colors """

        # Get the daily PnL DataFrame where columns represent tradables (instruments)
        daily_pnl_df = self.getDailyPnL( status_dict )

        # Set up the figure and axes
        fig, ax = plt.subplots()

        # Initialize the bottom for stacking bars
        bottom = None
        
        # Define colors for each instrument (optional, let matplotlib handle it otherwise)
        colors = plt.cm.get_cmap('tab10', len(daily_pnl_df.columns))  # Color map

        # Iterate over each instrument (column) and plot its PnL as a bar with a different color
        for i, instrument in enumerate(daily_pnl_df.columns):
            pnl_values = daily_pnl_df[instrument]
            
            # If it's the first instrument, no need for 'bottom' (base), otherwise stack
            if bottom is None:
                bottom = [0] * len(pnl_values)  # Initialize the bottom at zero for the first instrument
                ax.bar(pnl_values.index, pnl_values.values, label=instrument, color=colors(i))
            else:
                ax.bar(pnl_values.index, pnl_values.values, bottom=bottom, label=instrument, color=colors(i))
            
            # Update the bottom to stack the next instrument's PnL on top of the previous ones
            bottom += pnl_values

        # Add title and labels
        plt.title("Daily PnL by Instrument (Stacked Bar Chart)")
        plt.xlabel("Date")
        plt.ylabel("PnL (in $)")  # Emphasize that the y-axis is in dollars

        # Rotate x-axis labels for better readability if needed
        plt.xticks( rotation = 45 )

        # Add a legend to show which color corresponds to which instrument
        plt.legend(title="Instruments", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Format the y-axis ticks to show dollar sign
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))

        # Show the plot with tight layout
        plt.tight_layout()
        plt.show()
