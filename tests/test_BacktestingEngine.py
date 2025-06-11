import unittest
from Backtesting.BacktestingEngine import ( OrderTicket, 
                                            Order, 
                                            Trade, 
                                            Direction, 
                                            TradeStatus, 
                                            OrderType, 
                                            PositionManager, 
                                            BacktestStatus, 
                                            Backtest
            )
import datetime as dt
import pandas as pd
import numpy as np
from unittest.mock import patch



class TradeTest(unittest.TestCase):

    def setUp(self):
        """ Set up initial trade for testing. """
        self.trade      = Trade(
            imnt        = "BTCUSDT",
            open_time   = dt.datetime(2023, 9, 5, 14, 0),
            entry_price = 5000.0,
            direction   = Direction.LONG,
            size        = 2.0,  # Long position with size 2
            leverage    = 5
        )

        self.short_trade = Trade(
            imnt        = "BTCUSDT",
            open_time   = dt.datetime(2023, 9, 5, 14, 0),
            entry_price = 5500.0,
            direction   = Direction.SHORT,
            size        = 3.0,  # Short position with size 2,
            leverage    = 5
        )

    def test_initial_margin(self):
        """ Test margin calculation for the initial trade. """
        expected_margin = 5000.0 * 2.0 / 5  # price * size / leverage
        self.assertEqual(self.trade.margin, expected_margin)

    def test_add_position(self):
        """ Test adding a position to the trade and recalculating entry price. """
        self.trade.add_position(open_price=5100.0, open_size=1.0)
        
        # New entry price calculation
        expected_entry_price = (5000.0 * 2.0 + 5100.0 * 1.0) / (2.0 + 1.0)
        self.assertEqual(self.trade.entry_price, expected_entry_price)
        self.assertEqual(self.trade.size, 3.0)  # size increases to 3

    def test_close_whole_position( self ):
        """ Test closing the whole position. """
        close_price = 5200.0

        # Close the whole position
        self.trade.close_position( close_price = close_price, close_size = 3 , close_time = dt.datetime.now() )

        # Check if the position is closed correctly
        self.assertEqual( self.trade.trade_status, TradeStatus.CLOSED )
        self.assertEqual( self.trade.size, 0.0 )
        self.assertEqual( self.trade.open_pnl, 0.0 )
        self.assertEqual( self.trade.closed_pnl, ( close_price - 5000.0 ) * 2.0 )
        
    def test_close_partial_position(self):
        """ Test partially closing the trade. """
        close_price = 5200.0
        close_size = 1.0

        # Open PnL should be updated
        self.trade._update_unrealized_pnl( cur_price = close_price )

        # Close a partial position
        self.trade.close_position(close_price=close_price, close_size = close_size, close_time=dt.datetime.now())
        
        # Check remaining size and closed PnL
        self.assertEqual(self.trade.size, 1.0)  # size should reduce to 1.0
        expected_closed_pnl = (close_price - 5000.0) * close_size
        self.assertEqual(self.trade.closed_pnl, expected_closed_pnl)

    def test_close_partial_position_short( self ):
        """ Test partially closing a short position. """
        close_price = 5100.0
        close_size = 1.0

        # Close a partial position
        self.short_trade.close_position( close_price = close_price, close_size = close_size, close_time = dt.datetime.now() )
        self.short_trade._update_unrealized_pnl( cur_price = close_price )

        # Check remaining size and closed PnL
        self.assertEqual( self.short_trade.size, 2.0 )
        self.assertEqual( self.short_trade.trade_status, TradeStatus.OPEN )

        expected_closed_pnl = ( 5500.0 - close_price ) * close_size
        self.assertEqual( self.short_trade.closed_pnl, expected_closed_pnl )
        expected_open_pnl   = ( 5500.0 - close_price ) * self.short_trade.size
        self.assertEqual( self.short_trade.open_pnl, expected_open_pnl )


    def test_update_unrealized_pnl(self):
        """ Test updating the open PnL for the trade based on current market price. """
        current_price = 5200.0
        self.trade._update_unrealized_pnl(cur_price=current_price)

        # Open PnL should be calculated
        expected_open_pnl = (current_price - 5000.0) * 2.0  # (current price - entry price) * size
        self.assertEqual(self.trade.open_pnl, expected_open_pnl)

        cur_price = 5500
        expected_open_pnl = ( cur_price - 5500 ) * self.short_trade.size
        self.assertEqual( self.short_trade.open_pnl, expected_open_pnl )

    def test_orderFilledToTrade( self ):
        """ Test converting an order to a trade. """
        order = Order(
            imnt = 'BTCUSDT',
            price=5100.0,
            direction=Direction.LONG,
            type=OrderType.OPEN,
            leverage=5,
            size=1.0,
            duration=2,
            cur_time=dt.datetime(2024, 9, 9, 22, 41)
        )

        trade = order._orderFilledToTrade( dt.datetime( 2024, 9, 9, 22, 41 ) )

        # Check if the trade is created correctly
        self.assertEqual( trade.imnt, order.imnt )
        self.assertEqual( trade.open_time, order.cur_time )
        self.assertEqual( trade.entry_price, order.price )
        self.assertEqual( trade.direction, order.direction )
        self.assertEqual( trade.size, order.size )
        self.assertEqual( trade.leverage, order.leverage )

class OrderTest( unittest.TestCase ):
    def test_order_creation(self):
        """ Test creating an order. """
        order = Order(
            imnt = 'BTCUSDT',
            price=5100.0,
            direction=Direction.LONG,
            type=OrderType.OPEN,
            leverage=5,
            size=1.0,
            duration=2,
            cur_time=dt.datetime(2024, 9, 9, 22, 41)
        )

        # Check order attributes
        self.assertEqual(order.price, 5100.0)
        self.assertEqual(order.direction, Direction.LONG)
        self.assertEqual(order.type, OrderType.OPEN)
        self.assertEqual(order.leverage, 5)
        self.assertEqual(order.size, 1.0)
        self.assertEqual(order.duration, 2)
        self.assertEqual(order.cur_time, dt.datetime(2024, 9, 9, 22, 41))

    def test_order_info(self):
        """ Test the info property of the Order class. """
        order = Order(
            imnt = 'BTCUSDT',
            price=5100.0,
            direction=Direction.LONG,
            type=OrderType.OPEN,
            leverage=5,
            size=1.0,
            duration=2,
            cur_time=dt.datetime(2024, 9, 9, 22, 41)
        )

        expected_info = 'imnt: BTCUSDT | price: 5100.0 | direction: Direction.LONG | type: OrderType.OPEN | leverage: 5 | size: 1.0 | cur_time: 2024-09-09 22:41:00 | duration: 2'
        self.assertEqual(order.info, expected_info)

class OrderTicketTest(unittest.TestCase):

    def setUp(self):
        """ Set up initial order ticket for testing. """
        self.order_ticket = OrderTicket()

    def test_add_order(self):
        """ Test adding an order to the order ticket. """
        order = Order(
            imnt = 'BTCUSDT',
            price = 5100.0,
            direction = Direction.LONG,
            type = OrderType.OPEN,
            leverage = 5,
            size = 1.0,
            duration = 2,
            cur_time = dt.datetime(2024, 9, 9, 22, 41)
        )

        self.order_ticket.add_order( order )

        # Check if the order is added correctly
        self.assertEqual( len( self.order_ticket ), 1 )
        self.assertEqual( len( self.order_ticket["BTCUSDT"] ), 1 )
        self.assertEqual( self.order_ticket["BTCUSDT"][ 0 ], order )

    def test_remove_order(self):
        """ Test removing an order from the order ticket. """
        order = Order(
            imnt = 'BTCUSDT',
            price = 5100.0,
            direction = Direction.LONG,
            type = OrderType.OPEN,
            leverage = 5,
            size = 1.0,
            duration = 2,
            cur_time = dt.datetime(2024, 9, 9, 22, 41)
        )

        self.order_ticket.add_order( order)
        self.order_ticket.remove_order( order )

        # Check if the order is removed correctly
        self.assertEqual(len(self.order_ticket), 0)
        self.assertNotIn("BTCUSDT", self.order_ticket)

    def test_update_orders(self):
        """ Test updating the orders in the order ticket. """
        order1 = Order(
            imnt = 'BTCUSDT',
            price = 5100.0,
            direction = Direction.LONG,
            type = OrderType.OPEN,
            leverage = 5,
            size = 1.0,
            duration = 2,
            cur_time = dt.datetime(2024, 9, 9, 22, 41)
        )

        order2 = Order(
            imnt = 'BTCUSDT',
            price = 5200.0,
            direction = Direction.SHORT,
            type = OrderType.CLOSE,
            leverage = 5,
            size = 0.5,
            duration = 1,
            cur_time = dt.datetime(2024, 9, 9, 22, 42)
        )

        self.order_ticket.add_order( order1 )
        self.order_ticket.add_order( order2 )

        # Update the orders in the order ticket
        self.order_ticket.update_orders()

        order1_ = Order(
            imnt = 'BTCUSDT',
            price = 5100.0,
            direction = Direction.LONG,
            type = OrderType.OPEN,
            leverage = 5,
            size = 1.0,
            duration = 1,
            cur_time = dt.datetime(2024, 9, 9, 22, 41)
        )

        order2_ = Order(
            imnt = 'BTCUSDT',
            price = 5200.0,
            direction = Direction.SHORT,
            type = OrderType.CLOSE,
            leverage = 5,
            size = 0.5,
            duration = 0,
            cur_time = dt.datetime(2024, 9, 9, 22, 42)
        )

        # Check if the orders are updated correctly
        self.assertEqual( self.order_ticket["BTCUSDT"][-1], order1_ )
        self.assertNotIn( "ETHUSDT", self.order_ticket.keys() )


    def test_aggregate_orders(self):
        """ Test aggregating orders from another order ticket. """
        order0 = Order( 
                    imnt = 'ETHUSDT',
                    price = 5300.0,
                    direction = Direction.LONG,
                    type = OrderType.OPEN,
                    leverage = 5,
                    size = 0.5,
                    duration = 1,
                    cur_time = dt.datetime( 2024, 9, 9, 22, 00 ) )
        
        self.order_ticket = OrderTicket(
            { "ETHUSDT": [ order0 ]
            } 
        ) 
        order1 = Order(
            imnt = 'BTCUSDT',
            price = 5100.0,
            direction = Direction.LONG,
            type = OrderType.OPEN,
            leverage = 5,
            size = 1.0,
            duration = 2,
            cur_time = dt.datetime(2024, 9, 9, 22, 41)
        )

        order2 = Order(
            imnt = 'ETHUSDT',
            price = 5200.0,
            direction = Direction.SHORT,
            type = OrderType.CLOSE,
            leverage = 5,
            size = 0.5,
            duration = 1,
            cur_time = dt.datetime(2024, 9, 9, 22, 42)
        )

        new_order_ticket = OrderTicket()
        new_order_ticket.add_order( order1 )
        new_order_ticket.add_order( order2 )

        self.order_ticket.aggregate_orders(new_order_ticket)

        # Check if the orders are aggregated correctly
        self.assertEqual( len(self.order_ticket[ 'ETHUSDT' ]), 2)
        self.assertEqual( len( self.order_ticket[ 'BTCUSDT' ] ), 1 )
        self.assertEqual( self.order_ticket["BTCUSDT"], [ order1 ] )
        self.assertEqual( self.order_ticket["ETHUSDT"], [ order0, order2 ] ) 

class PositionManagerTest(unittest.TestCase):

    def setUp(self):
        """ Set up initial position manager for testing. """
        self.position_manager = PositionManager("BTCUSDT")

    def test_process_new_trade_open(self):
        """ Test processing a new trade to open a position. """
        new_trade = Trade(
            imnt="BTCUSDT",
            open_time=dt.datetime(2023, 9, 5, 14, 0),
            entry_price=5000.0,
            direction=Direction.LONG,
            size=2.0,  # Long position with size 2
            leverage=5
        )

        self.position_manager.process_new_trade(new_trade)

        # Check if the position is opened correctly
        self.assertEqual( len(self.position_manager.positions ), 1)
        self.assertEqual(self.position_manager.positions[-1], new_trade)

    def test_process_new_trade_update(self):
        """ Test processing a new trade to update an existing position. """
        existing_trade = Trade(
            imnt="BTCUSDT",
            open_time=dt.datetime(2023, 9, 5, 14, 0),
            entry_price=5000.0,
            direction=Direction.LONG,
            size=2.0,  # Long position with size 2
            leverage=5
        )

        self.position_manager.process_new_trade(existing_trade)

        new_trade = Trade(
            imnt="BTCUSDT",
            open_time=dt.datetime(2023, 9, 5, 15, 0),
            entry_price=5100.0,
            direction=Direction.LONG,
            size=1.0,  # Additional size to the existing position
            leverage=5
        )

        self.position_manager.process_new_trade(new_trade)

        # Check if the position is updated correctly
        self.assertEqual(len(self.position_manager.positions ), 1)
        self.assertEqual(self.position_manager.positions[-1].size, 3.0)

    def test_process_new_trade_close(self):
        """ Test processing a new trade to close a position. """
        existing_trade = Trade(
            imnt="BTCUSDT",
            open_time=dt.datetime(2023, 9, 5, 14, 0),
            entry_price=5000.0,
            direction=Direction.LONG,
            size=2.0,  # Long position with size 2
            leverage=5
        )

        self.position_manager.process_new_trade(existing_trade)

        close_trade = Trade(
            imnt="BTCUSDT",
            open_time=dt.datetime(2023, 9, 5, 15, 0),
            entry_price=5100.0,
            direction=Direction.SHORT,
            size=2.0,  # Size to close the position
            leverage=5
        )

        self.position_manager.process_new_trade(close_trade)

        # Check if the position is closed correctly
        self.assertEqual(len(self.position_manager.positions), 1)
        self.assertEqual(self.position_manager.positions[-1].trade_status, TradeStatus.CLOSED)
        self.assertEqual(self.position_manager.positions[-1].closed_pnl, 200.0 )

    def test_update_position_unrealized_pnl( self ):
        """ Test updating the open PnL for the positions. """
        existing_trade = Trade(
            imnt="BTCUSDT",
            open_time=dt.datetime(2023, 9, 5, 14, 0),
            entry_price=5000.0,
            direction=Direction.LONG,
            size=2.0,  # Long position with size 2
            leverage=5
        )

        self.position_manager.process_new_trade(existing_trade)

        current_price = 5200.0
        self.position_manager._update_position_unrealized_pnl(current_price)

        # Check if the open PnL is updated correctly
        self.assertEqual(self.position_manager.positions[-1].open_pnl, (current_price - existing_trade.entry_price) * existing_trade.size)

class BacktestStatusTest(unittest.TestCase):
    def setUp(self):
        """ Set up initial backtest status for testing. """
        self.tradables = ["BTCUSDT", "ETHUSDT"]
        self.initial_cash = 10000.0
        self.backtest_status = BacktestStatus(tradables=self.tradables, initial_cash=self.initial_cash)

    def test_initial_state(self):
        """ Test the initial state of the backtest status. """
        self.assertEqual(len(self.backtest_status.cur_positions), len(self.tradables))
        self.assertEqual(len(self.backtest_status.cur_priceVector), len(self.tradables))
        self.assertEqual(len(self.backtest_status.cur_CumulativePnL), len(self.tradables))
        self.assertEqual(self.backtest_status.cur_cash, self.initial_cash)
        self.assertEqual(self.backtest_status.cur_portfolioMtMValue, self.initial_cash)
        self.assertEqual(len(self.backtest_status.cur_orderTicket), 0)

    def test_update_positions(self):
        """ Test updating the positions in the backtest status. """
        position_manager = PositionManager("BTCUSDT")
        self.backtest_status.cur_positions["BTCUSDT"] = position_manager

        # Check if the positions are updated correctly
        self.assertEqual(self.backtest_status.cur_positions["BTCUSDT"], position_manager)

    def test_update_price_vector(self):
        """ Test updating the price vector in the backtest status. """
        price = 5000.0
        self.backtest_status.cur_priceVector["BTCUSDT"] = price

        # Check if the price vector is updated correctly
        self.assertEqual(self.backtest_status.cur_priceVector["BTCUSDT"], price)

    def test_update_PnL_vector(self):
        """ Test updating the PnL vector in the backtest status. """
        pnl = 100.0
        self.backtest_status.cur_CumulativePnL["BTCUSDT"] = pnl

        # Check if the PnL vector is updated correctly
        self.assertEqual(self.backtest_status.cur_CumulativePnL["BTCUSDT"], pnl)

    def test_update_cash(self):
        """ Test updating the cash in the backtest status. """
        cash = 5000.0
        self.backtest_status.cur_cash = cash

        # Check if the cash is updated correctly
        self.assertEqual(self.backtest_status.cur_cash, cash)

    def test_update_portfolioPnL(self):
        """ Test updating the portfolio PnL in the backtest status. """
        portfolio_pnl = 1000.0
        self.backtest_status.cur_portfolioMtMValue = portfolio_pnl

        # Check if the portfolio PnL is updated correctly
        self.assertEqual(self.backtest_status.cur_portfolioMtMValue, portfolio_pnl)

    def test_update_order_ticket(self):
        """ Test updating the order ticket in the backtest status. """
        order_ticket = OrderTicket()
        self.backtest_status.cur_orderTicket = order_ticket

        # Check if the order ticket is updated correctly
        self.assertEqual(self.backtest_status.cur_orderTicket, order_ticket)


class BacktestTest( unittest.TestCase ):

    def setUp(self):
        """ Set up initial backtest for testing. """
        self.tradables = ["BTCUSDT", "ETHUSDT"]


        #  Load offline data for testing

        self.dataDict = {}
        path = r"/Users/HaoqingWu/Documents/Trading/TradingBinance/MarketDataLoader/OfflineData/"
        for imnt in self.tradables:
            self.dataDict[ imnt ] = pd.read_csv( path + f'{imnt}_4h_Main.csv' )
            self.dataDict[ imnt ][ 'Open Time' ] = pd.to_datetime( self.dataDict[ imnt ][ 'Open Time' ] )
            self.dataDict[ imnt ].set_index( 'Open Time', inplace = True )
        startTime = self.dataDict[ "ETHUSDT" ].index[ 0 ]
        self.dataDict[ "BTCUSDT" ] = self.dataDict[ "BTCUSDT" ].loc[ startTime: ]
        self.initial_cash = 1000000.0
        self.backtest = Backtest( self.dataDict, self.initial_cash )
        self.backtest._status  = BacktestStatus( self.tradables, self.initial_cash ) 

    def test_check_shape(self):
        """ Test the shape check function of the Backtest class. """
        self.assertTrue( self.backtest._Backtest__checkShape() )

    def test_get_cur_time(self):
        """ Test getting the current time in the Backtest class. """
        cur_time = self.backtest.getCurTime()
        self.assertIsInstance(cur_time, pd.Timestamp)

    def test_get_next_time(self):
        """ Test getting the next time in the Backtest class. """
        next_time = self.backtest.getNextTime()
        self.assertIsInstance(next_time, pd.Timestamp)

    def test_increment_time(self):
        """ Test incrementing the time in the Backtest class. """
        prev_time = self.backtest.getCurTime()
        self.backtest._Backtest__incrementTime()
        cur_time = self.backtest.getCurTime()
        self.assertGreater(cur_time, prev_time)

    def test_get_next_open_price(self):
        """ Test getting the next open price in the Backtest class. """
        next_open_price = self.backtest._getNextOpenPrice("BTCUSDT")
        self.assertIsInstance(next_open_price, float)

    def test_get_next_high_price(self):
        """ Test getting the next high price in the Backtest class. """
        next_open_price = self.backtest._getNextOpenPrice("BTCUSDT")
        next_high_price = self.backtest._getNextHighPrice("BTCUSDT")
        self.assertIsInstance(next_high_price, float)
        self.assertGreaterEqual(next_high_price, next_open_price)

    def test_get_next_low_price(self):
        """ Test getting the next low price in the Backtest class. """
        next_low_price = self.backtest._getNextLowPrice("BTCUSDT")
        next_open_price = self.backtest._getNextOpenPrice("BTCUSDT")
        self.assertIsInstance(next_low_price, float)
        self.assertLessEqual( next_low_price, next_open_price )

    def test_get_next_close_price(self):
        """ Test getting the next close price in the Backtest class. """
        next_close_price = self.backtest._getNextClosePrice("BTCUSDT")
        self.assertIsInstance(next_close_price, float)

    def test_get_next_volume(self):
        """ Test getting the next volume in the Backtest class. """
        next_volume = self.backtest._getNextVolume("BTCUSDT")
        self.assertIsInstance(next_volume, float )

    def test_get_current_available_data(self):
        """ Test getting the current available data in the Backtest class. """

        current_available_data = self.backtest.getCurrentAvailableData()
        self.assertIsInstance(current_available_data, dict)

    def test_execute_order(self):
        """Test executing an order in the Backtest class."""

        ### Case 2: Order not filled ###
        # Prepare a ticket with one OPEN order
        order_ticket = OrderTicket()
        test_time = pd.Timestamp( "2021-01-01 00:00:00" )
        order1   = Order(
                            imnt = "BTCUSDT",
                            price = 30000.0,
                            direction = Direction.LONG,
                            type = OrderType.OPEN,
                            leverage = 1,
                            size = 1.0,
                            duration = 1 ,
                            cur_time = test_time
                        )               
        order_ticket.add_order( order1 )

        # Execute orders
        prev_cash = self.backtest._status.cur_cash
        self.backtest._executeOrders( order_ticket, 
                                      self.backtest._status.cur_positions, 
                                      prev_cash, 
                                      test_time
                                    )

        # Check if position is created and cash reduced
        self.assertEqual( [], self.backtest._status.cur_positions["BTCUSDT"].positions )
        self.assertEqual( self.backtest._status.cur_cash, self.initial_cash )

        ### Case 2: Long order filled ###
        order2   = Order(
                    imnt = "BTCUSDT",
                    price = 13500.0,
                    direction = Direction.LONG,
                    type = OrderType.OPEN,
                    leverage = 5,
                    size = 5.0,
                    duration = 1 ,
                    cur_time = test_time
                )               
        order_ticket.add_order( order2 )
        
        # Execute orders
        prev_cash = self.backtest._status.cur_cash
        self.backtest._executeOrders( order_ticket, 
                                      self.backtest._status.cur_positions, 
                                      prev_cash, 
                                      test_time
                                    )

        # Check if position is created and cash reduced
        self.assertIsNotNone( self.backtest._status.cur_positions["BTCUSDT"].positions[ 0 ] )
        expected_value = self.initial_cash - 13500
        tolerance = abs(expected_value) * 0.01  # 1% of expected_value
        self.assertAlmostEqual( self.backtest._status.cur_cash, expected_value, delta = tolerance )

        # check margin 
        self.assertEqual( self.backtest._status.cur_positions["BTCUSDT"].positions[ 0 ].margin, 13500.0 )


    def test_execute_order_insufficient_cash(self):
        """Test executing an order when cash might be insufficient."""
        # Temporarily reduce available cash
        self.backtest._status.cur_cash = 10.0
        order_ticket = OrderTicket()
        expensive_order = Order(
            imnt="BTCUSDT",
            price=50000.0,
            direction=Direction.LONG,
            type=OrderType.OPEN,
            leverage=1,
            size=1.0,
            duration=1,
            cur_time=self.backtest.getCurTime()
        )
        order_ticket.add_order(expensive_order)

        prev_cash = self.backtest._status.cur_cash
        self.backtest._executeOrders(order_ticket, self.backtest._status.cur_positions, prev_cash, self.backtest.getCurTime())

        # The order should fail and no position created
        self.assertFalse(self.backtest._status.cur_positions["BTCUSDT"].positions)
        self.assertEqual(self.backtest._status.cur_cash, 10.0)

    def test_update_funding_cost(self):
        """ Test updating the funding cost in the Backtest class """

        mock_trade1 = Trade(
            imnt="BTCUSDT",
            open_time=dt.datetime(2021, 1, 1, 0, 0),
            entry_price=12500.0,
            direction=Direction.LONG,
            size=1.0,
            leverage=5
        )

        mock_trade2 = Trade(
            imnt = "ETHUSDT",
            open_time = dt.datetime(2021, 1, 1, 0, 0),
            entry_price = 500.0,
            direction = Direction.SHORT,
            size = 10.0,
            leverage = 5
        )
        self.backtest._status.cur_positions["BTCUSDT"] = PositionManager( "BTCUSDT", [mock_trade1])
        self.backtest._status.cur_positions["ETHUSDT"] = PositionManager( "ETHUSDT", [mock_trade2])
        # Test positive funding rate
        with patch.object(self.backtest, "_getNextFundingRate", return_value = 0.0001 ):
            self.backtest._updateFundingCost()
            self.assertEqual( self.backtest._status.cur_cash, 
                              1000000.0 - 12500.0 * 1.0 * 0.0001 + 500 * 10 * 0.0001 )



    def myStrategy( self ):
        """ Test a buy-and-hold strategy for the Backtest class. """
        # Initialize the order ticket
        status = self.backtest.getCurrentAvailableData( lookBack = 3 )[ 'status' ]
        order_ticket = OrderTicket()
        if not status.cur_positions[ 'BTCUSDT' ].positions and not status.cur_positions['ETHUSDT'].positions:

            # Buy 1 BTC
            btc_order = Order(
                imnt = "BTCUSDT",
                price = self.backtest._getNextOpenPrice("BTCUSDT"),
                direction = Direction.LONG,
                type      = OrderType.OPEN,
                leverage  = 1,
                size      = 1.0,
                duration  = 1,
                cur_time  = self.backtest.getCurTime()
            )
            order_ticket.add_order( btc_order )

            # Buy 1 ETH
            eth_order = Order(
                imnt = "ETHUSDT",
                price = self.backtest._getNextOpenPrice("ETHUSDT"),
                direction = Direction.LONG,
                type = OrderType.OPEN,
                leverage = 1,
                size = 1.0,
                duration = 1,
                cur_time = self.backtest.getCurTime()
            )
            order_ticket.add_order( eth_order )

        return order_ticket

    def test_runStrategy(self):
        """ Test running the strategy in the Backtest class. """
        self.backtest.myStrategy = self.myStrategy
        startTime = dt.datetime(2021,1,1,0,0)
        endTime   = dt.datetime(2022,1,1,0,0)
        strategy_result = self.backtest.runStrategy( startTime, endTime )
        self.assertIsInstance(strategy_result, dict)
        
        ### Testing Postprocessing API ###

        # Get the PnL dataframe
        pnl_df = self.backtest.getDailyPnL( strategy_result )

        expected_columns = [ 'BTCUSDT', 'ETHUSDT']
        self.assertListEqual( list(pnl_df.columns), expected_columns )

        # Check if the PnL dataframe has the correct number of rows
        num_days = ( endTime - startTime ).days + 1
        self.assertEqual(num_days, pnl_df.shape[0])

        imnt_total_return = self.backtest.computeInstrumentTotalReturn( strategy_result )
        buy_and_hold_return = self.dataDict[ "BTCUSDT" ].loc[ endTime ][ "Close" ] - self.dataDict[ "BTCUSDT" ].loc[ startTime ][ "Close" ]
        
        self.assertAlmostEqual( imnt_total_return[ 'BTCUSDT' ], buy_and_hold_return, delta = buy_and_hold_return * 0.005 )

        # plot the equity curve
        self.backtest.plotEquityCurve( strategy_result )

        # plot the daily PnL    
        self.backtest.plotDailyPnL( strategy_result )

        # Compute the Sharpe ratio
        expected_sharpe_ratio = 0.29466
        actual_sharpe_ratio = self.backtest.computeSharpeRatio(strategy_result)
        self.assertAlmostEqual(actual_sharpe_ratio, expected_sharpe_ratio, delta=1e-3)

        MDD = self.backtest.computeMaxDrawdown( strategy_result )
        expected_MDD = -0.0345
        self.assertAlmostEqual( MDD, expected_MDD, delta = 1e-2 )

        CVaR = self.backtest.computeCVaR( strategy_result, level = 0.05 )
        expected_CVaR = -0.0048
        self.assertAlmostEqual( CVaR, expected_CVaR, delta = 1e-3 )




if __name__ == "__main__":
    unittest.main()
