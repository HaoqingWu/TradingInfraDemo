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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure

from typing import Optional, Union
from enum import Enum
from dataclasses import dataclass, field
import copy

# Configure logging
logging.basicConfig( level = logging.ERROR, 
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
    
    def _orderFilledToTrade( self, open_time: pd.Timestamp ) -> 'Trade':
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
            logging.info( f"{order.cur_time} Added order: {order.info}" )

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
            logging.info(f"{order.cur_time}: Order removed for {imnt}.")
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
                        logging.info(f"{imnt} order: at Price {order.price} has expired and will be removed.")
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
                partial_pnl = ( close_price - self.entry_price ) * close_size
            else:
                partial_pnl = ( self.entry_price - close_price ) * close_size
            self.closed_pnl += partial_pnl
            self.size       -= close_size

    def _update_unrealized_pnl( self, cur_price: float ) -> None:
        """ update the current MtM pnl of the trade """
        if self.direction == Direction.LONG:
            self.open_pnl = ( cur_price - self.entry_price ) * self.size  
        else:
            self.open_pnl = ( self.entry_price - cur_price ) * self.size
        logging.info( f"Updated trade: {self.direction.value} {self.imnt}. Open PnL: {self.open_pnl}")


class PositionManager:
    """
    A class to manage multiple trades for ONE instrument.
    It ensures that at most one LONG and one SHORT trade is open per instrument.
    Attributes:
        imnt (str): The symbol of the instrument being traded (e.g., 'BTCUSDT').
        positions (list[Trade]): A list of Trade objects representing the current open trades.
        cumClosedPnL (float): The cumulative closed PnL for all trades.
    """
    def __init__( self, imnt: str, positions: list[ Trade ] = [] ):
        self.imnt: str = imnt
        self.positions: list[ Trade ] = positions if positions else []
        self.cumClosedPnL: float = 0.0

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
                # Store the initial closed_pnl before closing
                initial_closed_pnl = cur_position.closed_pnl
                cur_position.close_position( new_trade.entry_price, new_trade.size, new_trade.open_time )
                
                # Add only the new closed PnL (not double counting)
                self.cumClosedPnL += ( cur_position.closed_pnl - initial_closed_pnl )
                
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
        cur_OpenPnLVector (dict[str, float]): A dictionary that stores the current Open PnL for each instrument.
        cur_CumulativePnL (dict[str, float]): A dictionary that stores the cumulative PnL for each instrument.
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
    cur_OpenPnLVector: dict[str, float] = field(init=False)
    cur_CumulativePnL: dict[str, float] = field(init=False)
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
        self.cur_OpenPnLVector = {imnt: 0.0 for imnt in self.tradables}
        self.cur_CumulativePnL = {imnt: 0.0 for imnt in self.tradables}
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
               f"Current Open PnL: {self.cur_OpenPnLVector}, "
               f"Current Cumulative PnL: {self.cur_CumulativePnL}, "
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
        new_copy.cur_OpenPnLVector = self.cur_OpenPnLVector.copy()
        new_copy.cur_CumulativePnL = self.cur_CumulativePnL.copy()
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
        self._performance_cache = None
        
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
                        trade = order._orderFilledToTrade( curTime )
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
                            trade = order._orderFilledToTrade( curTime )
                            initial_trade_margin = last_trade.margin
                            positions_t[ imnt ].process_new_trade( trade )

                            # need to be careful handling the cash here
                            new_margin = positions_t[ imnt ].positions[ -1 ].margin
                            if last_trade.direction == Direction.LONG:
                                closed_pnl = ( order.price - last_trade.entry_price ) * trade.size
                            else: # SHORT trade pnl
                                closed_pnl = ( last_trade.entry_price - order.price ) * trade.size
                            cash_t += ( ( initial_trade_margin - new_margin ) - ( trade.notional * TRANSACTION_FEE ) + closed_pnl )
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

            cur_cash = self._status.cur_cash
            if cur_pos.direction == Direction.LONG:
                if fundingRate > 0:
                    if self._status.cur_cash > abs( notional * fundingRate ):
                        self._status.cur_cash -= notional * fundingRate
                    else:
                        # not enough cash to pay the funding cost, substract the margin which is an estimate
                        cur_pos.size -= notional * fundingRate / cur_price 
                else:
                    self._status.cur_cash -= notional * fundingRate            
            else: # SHORT trade
                if fundingRate < 0:
                    if self._status.cur_cash > abs( notional * fundingRate ):
                        self._status.cur_cash += notional * fundingRate
                    else:
                        cur_pos.size += notional * fundingRate / cur_price
                else:
                    self._status.cur_cash += notional * fundingRate

            self._status.cur_fundingCost[ imnt ] += ( cur_cash - self._status.cur_cash )

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
            # prev_cumPnL = self._status.cur_OpenPnLVector[ imnt ]
            if positions_t[ imnt ].positions:
                # Update the unrealized/open PnL of the open trades
                positions_t[ imnt ]._update_position_unrealized_pnl( self._status.cur_priceVector[ imnt ] )
                
                # Update pnl vector
                # TODO: Currently, this part doesn't support longing and shorting on the same instrument

                last_position = positions_t[ imnt ].positions[ -1 ]
                self._status.cur_OpenPnLVector[ imnt ] = last_position.open_pnl
                
                # Update the cumulative PnL
                self._status.cur_CumulativePnL[ imnt ] = self._status.cur_OpenPnLVector[ imnt ] \
                                                        + positions_t[ imnt ].cumClosedPnL 

        if self._status.cur_cash < 0:
            # raise ValueError( "Cash cannot be negative!" )        
            logging.warning( f"Cash is negative: {self._status.cur_cash}. This may lead to liquidation." )

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


    ###############################
    ### Performance Summary API ###
    ###############################

    def _getPerformanceMetrics( self, status_dict: dict ) -> dict:
        """
        Calculate all performance metrics at once to avoid redundant computations.
        This private method serves as a cache for all performance calculations.
        
        Args:
            status_dict (dict): Dictionary of backtest status snapshots
            
        Returns:
            dict: Dictionary containing all performance metrics
        """

        if not hasattr( self, '_performance_cache' ) or self._performance_cache is None:
            # PnL data - represents the profit/loss for each instrument at each timestamp
            cum_pnl_dict = { timeStamp: status.cur_CumulativePnL for timeStamp, status in status_dict.items() }
            cum_pnl_df = pd.DataFrame.from_dict( cum_pnl_dict, orient = 'index', columns = self._tradables )

            # Daily PnL data - represents the daily profit/loss for each instrument
            pnl_df = cum_pnl_df.diff().fillna( 0 )
            
            # Portfolio value data - represents total portfolio value (including cash) at each timestamp
            portfolio_value_dict = {timeStamp: status.cur_portfolioMtMValue for timeStamp, status in status_dict.items()}
            portfolio_value_series = pd.Series(portfolio_value_dict, name='Portfolio Value')
            
            # Daily aggregations
            daily_pnl_df = pnl_df.resample('D').sum()
            daily_portfolio_value = portfolio_value_series.resample('D').last()
            
            # Returns calculation
            daily_returns = daily_portfolio_value.pct_change()
            daily_returns.iloc[0] = daily_portfolio_value.iloc[0] / self._initialCash - 1
            
            total_return = portfolio_value_series.iloc[-1] / self._initialCash - 1

            # Store all results in cache
            self._performance_cache = {
                'cum_pnl_df': cum_pnl_df,
                'pnl_df': pnl_df,
                'portfolio_value_series': portfolio_value_series,
                'daily_pnl_df': daily_pnl_df,
                'daily_portfolio_value': daily_portfolio_value,
                'daily_returns': daily_returns,
                'total_pnl_by_instrument': pnl_df.sum(),
                'total_pnl': pnl_df.sum().sum(),
                'total_return': total_return,
                "total_funding_cost": pd.DataFrame({t: s.cur_fundingCost for t, s in status_dict.items()} ).iloc[ :, -1],
                'margin_series': pd.Series({t: s.cur_total_margin for t, s in status_dict.items()}, name='Total Margin'),
                'notional_series': pd.Series({t: s.cur_total_notional for t, s in status_dict.items()}, name='Total Notional'),
                'leverage_series': pd.Series({t: s.cur_total_leverage for t, s in status_dict.items()}, name='Total Leverage'),
            }
            
        return self._performance_cache

    def getPnL(self, status_dict: dict) -> pd.DataFrame:
        """
        Return the PnL of each time bar from the backtest.
        This represents the profit/loss for each instrument at each timestamp.
        
        Args:
            status_dict (dict): Dictionary of backtest status snapshots
            
        Returns:
            pd.DataFrame: DataFrame with timestamps as index and instruments as columns
        """
        return self._getPerformanceMetrics(status_dict)['pnl_df']

    def getPortfolioDailyValues(self, status_dict: dict) -> pd.Series:
        """
        Return the daily portfolio value (MtM) of the backtest.
        This represents the total portfolio value including cash, margin, and unrealized P&L.
        
        Args:
            status_dict (dict): Dictionary of backtest status snapshots
            
        Returns:
            pd.Series: Series with dates as index and portfolio values as values
        """
        return self._getPerformanceMetrics(status_dict)['daily_portfolio_value']

    def getDailyPnL(self, status_dict: dict) -> pd.DataFrame:
        """
        Return the daily PnL of the backtest aggregated by day.
        This represents the daily profit/loss for each instrument.
        
        Args:
            status_dict (dict): Dictionary of backtest status snapshots
            
        Returns:
            pd.DataFrame: DataFrame with dates as index and instruments as columns
        """
        return self._getPerformanceMetrics(status_dict)['daily_pnl_df']

    def getTotalMarginSeries(self, status_dict: dict) -> pd.Series:
        """
        Return the total margin used in the backtest at each timestamp.
        
        Args:
            status_dict (dict): Dictionary of backtest status snapshots
            
        Returns:
            pd.Series: Series with timestamps as index and total margin as values
        """
        return self._getPerformanceMetrics(status_dict)['margin_series']

    def getTotalNotionalSeries(self, status_dict: dict) -> pd.Series:
        """
        Return the total notional value of positions in the backtest at each timestamp.
        
        Args:
            status_dict (dict): Dictionary of backtest status snapshots
            
        Returns:
            pd.Series: Series with timestamps as index and total notional as values
        """
        return self._getPerformanceMetrics(status_dict)['notional_series']

    def getTotalLeverageSeries(self, status_dict: dict) -> pd.Series:
        """
        Return the total leverage of the portfolio in the backtest at each timestamp.
        
        Args:
            status_dict (dict): Dictionary of backtest status snapshots
            
        Returns:
            pd.Series: Series with timestamps as index and total leverage as values
        """
        return self._getPerformanceMetrics(status_dict)['leverage_series']

    def getPortfolioDailyRelativeReturns(self, status_dict: dict) -> pd.Series:
        """
        Return the daily percentage changes in portfolio value.
        
        Args:
            status_dict (dict): Dictionary of backtest status snapshots
            
        Returns:
            pd.Series: Series with dates as index and daily returns as values
        """
        return self._getPerformanceMetrics(status_dict)['daily_returns']

    def computeInstrumentTotalReturn(self, status_dict: dict) -> pd.Series:
        """
        Return the total PnL for each instrument over the entire backtest period.
        
        Args:
            status_dict (dict): Dictionary of backtest status snapshots
            
        Returns:
            pd.Series: Series with instruments as index and total PnL as values
        """
        return self._getPerformanceMetrics(status_dict)['total_pnl_by_instrument']

    def computeTotalReturn(self, status_dict: dict) -> float:
        """
        Return the total return of the backtest as a single number.
        
        Args:
            status_dict (dict): Dictionary of backtest status snapshots
            
        Returns:
            float: Total return across all instruments
        """
        return self._getPerformanceMetrics(status_dict)['total_return']

    def computeSharpeRatio(self, status_dict: dict) -> float:
        """
        Return the annualized Sharpe Ratio of the backtest.
        
        Args:
            status_dict (dict): Dictionary of backtest status snapshots
            
        Returns:
            float: Annualized Sharpe Ratio
        """
        daily_returns = self._getPerformanceMetrics(status_dict)['daily_returns']
        
        # Avoid division by zero
        if daily_returns.std() == 0:
            return 0
        
        sharpe_ratio = daily_returns.mean() / daily_returns.std()
        return sharpe_ratio * np.sqrt( 365 )  # Annualize for crypto's 24/7 market
    
    def computeSortinoRatio( self, status_dict: dict ) -> float:
        """
        Return the annualized Sortino Ratio of the backtest.
        
        Args:
            status_dict (dict): Dictionary of backtest status snapshots
            
        Returns:
            float: Annualized Sortino Ratio
        """

        daily_returns = self._getPerformanceMetrics(status_dict)['daily_returns']
        
        # Calculate downside deviation
        downside_returns = daily_returns[ daily_returns < 0 ]
        if downside_returns.std() == 0:
            return 0
        
        sortino_ratio = daily_returns.mean() / downside_returns.std()
        
        return sortino_ratio * np.sqrt( 365 )
    
    def computeCalmarRatio( self, status_dict: dict ) -> float:
        """
        Return the Calmar Ratio of the backtest.
        
        Args:
            status_dict (dict): Dictionary of backtest status snapshots
            
        Returns:
            float: Calmar Ratio
        """
        total_return = self.computeTotalReturn(status_dict)
        max_drawdown = self.computeMaxDrawdown(status_dict)
        
        # Avoid division by zero
        if max_drawdown == 0:
            return np.inf
        
        return total_return / abs( max_drawdown )

    def computeMaxDrawdown(self, status_dict: dict) -> float:
        """
        Return the maximum drawdown of the backtest.
        
        Args:
            status_dict (dict): Dictionary of backtest status snapshots
            
        Returns:
            float: Maximum drawdown as a negative percentage
        """
        portfolio_values = self._getPerformanceMetrics(status_dict)['daily_portfolio_value']
        cum_max = portfolio_values.cummax()
        drawdown = (portfolio_values - cum_max) / cum_max
        return drawdown.min()

    def computeCVaR(self, status_dict: dict, level: float) -> float:
        """
        Return the Conditional Value at Risk (CVaR) of the backtest.
        
        Args:
            status_dict (dict): Dictionary of backtest status snapshots
            level (float): Confidence level (e.g., 0.05 for 5%)
            
        Returns:
            float: CVaR value
        """
        portfolio_returns = self._getPerformanceMetrics(status_dict)['daily_returns']
        n = len(portfolio_returns)
        if n == 0:
            return 0
            
        level_ordinal = int(n * level)
        if level_ordinal == 0:  # Avoid empty slice
            level_ordinal = 1
            
        res_ = portfolio_returns.sort_values()
        return res_.iloc[:level_ordinal].mean()

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
                'Closed PnL': [],
                'Open PnL': []
            }
            for trade in status_dict[ lastTimeStamp ].cur_positions[ imnt ].positions: # loop trades
                res[ imnt ][ 'Open Time' ].append( trade.open_time )
                res[ imnt ][ 'Entry Price' ].append( trade.entry_price )
                res[ imnt ][ 'Direction' ].append( trade.direction )
                res[ imnt ][ 'Close Time' ].append( trade.close_time )
                res[ imnt ][ 'Closed PnL' ].append( trade.closed_pnl )
                res[ imnt ][ 'Open PnL' ].append( trade.open_pnl )
                
            res[ imnt ] = pd.DataFrame( res[ imnt ] )

        return res


    ##########################
    ### Performance Graphs ###
    ##########################


    def plotEquityCurve(self, 
                   status_dict: dict, 
                   plot_trades: bool = False,
                   plot_btc_benchmark: bool = True) -> Figure:
        """ 
        Plot the equity curve of the backtest with dollar formatting on the y-axis, 
        trade markers, drawdown subplot, and optional BTC buy-and-hold comparison
        """
        
        # Get the portfolio value time series
        portfolio_value_series = self.getPortfolioDailyValues(status_dict)
        
        # Calculate drawdown series
        cum_max = portfolio_value_series.cummax()
        drawdown_series = (portfolio_value_series - cum_max) / cum_max * 100
        
        # Create BTC buy-and-hold benchmark if requested
        btc_equity_series = None
        btc_drawdown_series = None
        
        if plot_btc_benchmark and 'BTCUSDT' in self._tradables:
            # Get BTC price data for the same period
            start_time = portfolio_value_series.index[0]
            end_time = portfolio_value_series.index[-1]
            
            # Get BTC prices (daily close prices)
            btc_prices = self._dataDict['BTCUSDT']['Close'].resample('D').last()
            btc_prices = btc_prices.loc[start_time:end_time]
            
            # Calculate BTC buy-and-hold equity curve
            initial_btc_price = btc_prices.iloc[0]
            btc_shares = self._initialCash / initial_btc_price  # Number of BTC shares bought
            btc_equity_series = btc_prices * btc_shares
            btc_equity_series.name = 'BTC Buy & Hold'
            
            # Calculate BTC drawdown
            btc_cum_max = btc_equity_series.cummax()
            btc_drawdown_series = (btc_equity_series - btc_cum_max) / btc_cum_max * 100
        
        # Create subplots: 2 rows, 1 column with shared x-axis
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=("Portfolio Equity vs BTC Buy & Hold", "Drawdown Comparison (%)")
        )
        
        # Add the strategy equity curve to the top subplot
        fig.add_trace(
            go.Scatter(
                x=portfolio_value_series.index,
                y=portfolio_value_series.values,
                mode='lines',
                name='Strategy',
                hoverinfo='x+y',
                line=dict(color='royalblue', width=2)
            ),
            row=1, col=1
        )
        
        # Add BTC buy-and-hold equity curve if available
        if btc_equity_series is not None:
            fig.add_trace(
                go.Scatter(
                    x = btc_equity_series.index,
                    y = btc_equity_series.values,
                    mode = 'lines',
                    name = 'BTC Buy & Hold',
                    hoverinfo = 'x+y',
                    line = dict(color='orange', width=2, dash='dash')
                ),
                row = 1, col = 1
            )
        
        # Add strategy drawdown to the bottom subplot
        fig.add_trace(
            go.Scatter(
                x=drawdown_series.index,
                y=drawdown_series.values,
                mode='lines',
                name='Strategy Drawdown',
                fill='tozeroy',
                fillcolor='rgba(65,105,225,0.2)',
                line=dict(color='royalblue'),
                hoverinfo='x+y',
                hovertemplate='%{x}<br>Strategy DD: %{y:.2f}%'
            ),
            row=2, col=1
        )
        
        # Add BTC drawdown to the bottom subplot
        if btc_drawdown_series is not None:
            fig.add_trace(
                go.Scatter(
                    x=btc_drawdown_series.index,
                    y=btc_drawdown_series.values,
                    mode='lines',
                    name='BTC Drawdown',
                    line=dict(color = 'orange', dash='dash'),
                    hoverinfo='x+y',
                    hovertemplate='%{x}<br>BTC DD: %{y:.2f}%'
                ),
                row=2, col=1
            )
        
        # Calculate and display performance comparison
        strategy_total_return = (portfolio_value_series.iloc[-1] / self._initialCash - 1) * 100
        strategy_max_dd = drawdown_series.min()
        
        if btc_equity_series is not None:
            btc_total_return = (btc_equity_series.iloc[-1] / self._initialCash - 1) * 100
            btc_max_dd = btc_drawdown_series.min()
            
            # Calculate daily returns for both
            strategy_returns = portfolio_value_series.pct_change().dropna()
            btc_returns = btc_equity_series.pct_change().dropna()
            corr = strategy_returns.corr( btc_returns )

            # Add performance comparison annotation
            comparison_text = (
                f"Strategy: {strategy_total_return:.1f}% return, {strategy_max_dd:.1f}% MDD <br>"
                f"BTC B&H: {btc_total_return:.1f}% return, {btc_max_dd:.1f}% MDD <br>"
                f"Strategy BTC Correlation: {corr:.2f}"
            )
            
            fig.add_annotation(
                text=comparison_text,
                xref="paper", yref="paper",
                x=0.95, y=0.98,  # Adjust these values to change position:
                                 # x: 0 (left) to 1 (right)
                                 # y: 0 (bottom) to 1 (top)
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
            )
        
        # Format y-axis on equity curve subplot
        fig.update_yaxes(
            title_text='Portfolio Value ($)',
            tickformat='$,.0f',
            row=1, col=1
        )
        
        # Format y-axis on drawdown subplot
        fig.update_yaxes(
            title_text='Drawdown (%)',
            tickformat='.1f',
            row=2, col=1
        )
        
        # Add horizontal line at y=0 on drawdown subplot
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Display the maximum drawdown value as a horizontal line
        max_dd = drawdown_series.min()
        fig.add_shape(
            type="line",
            x0=drawdown_series.index[0],
            y0=max_dd,
            x1=drawdown_series.index[-1],
            y1=max_dd,
            line=dict(color="red", width=1, dash="dash"),
            row=2, col=1
        )
        
        # Add annotation for maximum drawdown
        max_dd_date = drawdown_series.idxmin()
        fig.add_annotation(
            x=max_dd_date,
            y=max_dd,
            text=f"Max Drawdown: {max_dd:.2f}%",
            showarrow=True,
            arrowhead=1,
            row=2, col=1,
            arrowcolor="red",
            font=dict(color="red"),
            bgcolor="white",
            bordercolor="red",
            borderwidth=1
        )
        

        # Update layout
        fig.update_layout(
            title='Equity Curve and Drawdown',
            hovermode='x unified',
            height=800,
            margin=dict(t=80, b=50, l=50, r=50),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Show the plot
        fig.show()
        
        return fig

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
                        text=f'Close {imnt} <br>Close Time: {close_time} <br>Closed PnL: {trade["Closed PnL"]:.2f}'
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
        pnl_df = pd.DataFrame( { timeStamp: status.cur_CumulativePnL for timeStamp, status in status_dict.items() } ).T
        pnl_df.plot( title = "PnL of the Backtest" )
        plt.show()

    def plotDailyPnL(self, status_dict: dict) -> None:
        """ Plot each instrument's daily PnL as a stacked bar chart with different colors using Plotly """

        # Get the daily PnL DataFrame where columns represent tradables (instruments)
        daily_pnl_df = self.getDailyPnL(status_dict)

        # Create a plotly figure
        fig = go.Figure()

        # Define colors for each instrument using plotly's built-in colorscales
        colorscale = px.colors.qualitative.Plotly[:len(daily_pnl_df.columns)]

        # Iterate over each instrument (column) and add a bar trace
        for i, instrument in enumerate(daily_pnl_df.columns):
            fig.add_trace(go.Bar(
                x=daily_pnl_df.index,
                y=daily_pnl_df[instrument],
                name=instrument,
                marker_color=colorscale[i % len(colorscale)]
            ))

        # Configure the layout for a stacked bar chart
        fig.update_layout(
            title="Daily PnL by Instrument (Stacked Bar Chart)",
            xaxis_title="Date",
            yaxis_title="PnL (in $)",
            barmode='stack',
            yaxis=dict(
                tickformat="$,.0f",
            ),
            legend=dict(
                title="Instruments",
                x=1.05,
                y=1,
                xanchor='left'
            ),
        )

        # Show the plot
        fig.show()
