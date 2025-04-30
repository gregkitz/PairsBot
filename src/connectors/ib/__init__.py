"""
Interactive Brokers (IB) Connector Module for the Intraday Statistical Arbitrage System.

This module provides connectivity to Interactive Brokers for market data retrieval,
order execution, and account management using the ib_insync library.
"""

from .ib_connector import IBConnector
from .ib_utils import contract_to_symbol, symbol_to_contract

__all__ = ['IBConnector', 'contract_to_symbol', 'symbol_to_contract'] 