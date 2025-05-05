import sys
sys.path.append("../")  # Adjust the path to the root directory of your project
import pandas as pd
from Backtesting.BacktestingEngine import *

class MomentumLongShortStrategy( Backtest ):
    """
    A long-short momentum strategy that goes long on the top N coins
    and short on the bottom N coins based on momentum scores.
    """
    
    def __init__( self, dataDict, initial_cash = INITIAL_CASH, n_top = 3, n_bottom = 3, leverage = 5 ):
        """
        Initialize the momentum long-short strategy.
        
        Args:
            dataDict: Dictionary of market data for each tradable
            initial_cash: Initial cash for the backtest
            n_top: Number of top coins to long
            n_bottom: Number of bottom coins to short
            leverage: Leverage for the strategy
        """
        super().__init__( dataDict, initial_cash )
        self.n_top    = n_top
        self.n_bottom = n_bottom
        self.leverage = leverage
    
    def myStrategy(self) -> OrderTicket:
        """
        Implements the momentum long-short strategy.
        
        Returns:
            OrderTicket: Order ticket with long and short positions
        """
        return self.momLS( self.n_top, self.n_bottom )
    
    def momLS( self, n_top: int = 3, n_bottom: int = 3 ) -> OrderTicket:
        """
        Long-Short momentum strategy that longs the top n_top coins and
        shorts the bottom n_bottom coins based on momentum scores.
        
        Args:
            n_top: Number of top momentum coins to long
            n_bottom: Number of bottom momentum coins to short
            
        Returns:
            OrderTicket: Order ticket with long and short positions
        """
        # Initialize the order ticket
        order_ticket = OrderTicket()
        
        # Get current timestamp
        cur_time = self.getCurTime()
        
        # Get current available data including status
        available_data = self.getCurrentAvailableData(lookBack=1)
        status = available_data['status']
        market_data = available_data['marketData']

        # Skip if we don't have enough data for momentum calculation 
        
        # Collect momentum scores for all coins
        mom_scores = {}
        for coin in self._tradables:
            if "mom_score" not in market_data[ coin ].columns:  # Using lookback parameter from getMomScore
                raise AttributeError("Momentum score not found in the DataFrame.")

            cur_score = market_data[coin]["mom_score"][ -1 ] # Get the current momentum score

            if not pd.isna(cur_score):
                mom_scores[coin] = cur_score
        
        # Skip if not enough coins have valid momentum scores
        if len(mom_scores) < n_top + n_bottom:
            return order_ticket
        
        # Sort coins by momentum score (descending)
        sorted_coins = sorted( mom_scores.items(), key = lambda x: x[ 1 ], reverse = True )
        
        # Get top and bottom coins
        top_coins    = sorted_coins[ : n_top]
        bottom_coins = sorted_coins[ -n_bottom : ]
        
        # CLOSE existing positions that are no longer in our target allocation
        for coin in self._tradables:
            if coin in status.cur_positions \
                        and status.cur_positions[coin].positions \
                        and status.cur_positions[coin].positions[-1].trade_status == TradeStatus.OPEN:
                
                # Get the last trade for this coin
                last_trade = status.cur_positions[coin].positions[-1]
                
                # If current long position is not in top_coins OR current short position is not in bottom_coins
                should_close = (last_trade.direction == Direction.LONG and coin not in [c for c, _ in top_coins]) or \
                            (last_trade.direction == Direction.SHORT and coin not in [c for c, _ in bottom_coins])
                            
                if should_close:
                    # Create a CLOSE order with opposite direction
                    close_order = Order(
                        imnt = coin,
                        price = self._getNextOpenPrice( coin ),
                        direction = Direction.SHORT if last_trade.direction == Direction.LONG else Direction.LONG,
                        type = OrderType.CLOSE,
                        leverage = last_trade.leverage,
                        size = last_trade.size,
                        duration = 1,
                        cur_time = cur_time
                    )
                    order_ticket.add_order(close_order)
        
        # LONG the top coins
        for coin, _ in top_coins:
            # Skip if already long this coin
            if coin in status.cur_positions and status.cur_positions[coin].positions and \
            status.cur_positions[coin].positions[-1].trade_status == TradeStatus.OPEN and \
            status.cur_positions[coin].positions[-1].direction == Direction.LONG:
                continue
            
            # Get current price
            cur_price = market_data[ coin ].iloc[ -1 ][ "Close" ]
            
            # Calculate position size using equal weight allocation (1/n_top of portfolio)
            
            # With 5x leverage
            target_allocation_size = status.cur_portfolioMtMValue / ( ( n_top + n_bottom) * cur_price ) * self.leverage
            
            # Create LONG order
            long_order = Order(
                imnt = coin,
                price = cur_price,
                direction = Direction.LONG,
                type = OrderType.OPEN,
                leverage = self.leverage,
                size = target_allocation_size,
                duration = 1,  # Execute on next bar only
                cur_time = cur_time
            )
            order_ticket.add_order(long_order)
        
        # SHORT the bottom coins
        for coin, _ in bottom_coins:
            # Skip if already short this coin
            if coin in status.cur_positions and status.cur_positions[coin].positions and \
            status.cur_positions[coin].positions[-1].trade_status == TradeStatus.OPEN and \
            status.cur_positions[coin].positions[-1].direction == Direction.SHORT:
                continue
            
            # Get current price
            cur_price = self._getNextOpenPrice(coin)
            
            # Calculate position size using equal weight allocation (1/n_bottom of portfolio)
            # With 5x leverage
            target_allocation_size = status.cur_portfolioMtMValue / ( ( n_top + n_bottom) * cur_price ) * self.leverage
            
            # Create SHORT order
            short_order = Order(
                imnt = coin,
                price = cur_price,
                direction = Direction.SHORT,
                type = OrderType.OPEN,
                leverage = self.leverage,
                size = target_allocation_size,
                duration = 1,  # Execute on next bar only
                cur_time = cur_time
            )
            order_ticket.add_order(short_order)
        
        return order_ticket