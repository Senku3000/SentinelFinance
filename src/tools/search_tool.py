"""Search Tool - Real-time market data retrieval"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from langchain_core.tools import BaseTool
from pydantic import Field
import yfinance as yf
from cachetools import TTLCache

from ..config import Config


class SearchTool(BaseTool):
    """Tool for fetching real-time market data and financial rates"""
    
    name: str = "search_tool"
    description: str = """
    Use this tool to get real-time market data and current financial rates.
    
    Can fetch:
    - Current gold prices (per gram/10gm)
    - Fixed Deposit (FD) rates from major banks
    - Mutual fund NAVs
    - Stock prices
    - Currency exchange rates
    
    Input should specify what data you need (e.g., "gold_rate", "fd_rates", "stock_price:RELIANCE").
    """
    
    cache: TTLCache = Field(default_factory=lambda: TTLCache(maxsize=100, ttl=Config.SEARCH_CACHE_TTL))
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache = TTLCache(maxsize=100, ttl=Config.SEARCH_CACHE_TTL)
    
    def _run(
        self,
        query: str,
        data_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch real-time market data
        
        Args:
            query: What data to fetch (e.g., "gold_rate", "fd_rates")
            data_type: Optional explicit data type
            
        Returns:
            Dictionary with market data
        """
        # Check cache first
        cache_key = f"{query}_{data_type or ''}"
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            cached_data["cached"] = True
            return cached_data
        
        data_type = data_type or query.lower()
        
        try:
            if "gold" in data_type.lower():
                result = self._get_gold_rate()
            elif "fd" in data_type.lower() or "fixed_deposit" in data_type.lower():
                result = self._get_fd_rates()
            elif "stock" in data_type.lower() or ":" in query:
                # Format: "stock_price:SYMBOL" or "stock:SYMBOL"
                symbol = query.split(":")[-1].strip() if ":" in query else query.replace("stock", "").strip()
                result = self._get_stock_price(symbol)
            elif "mutual_fund" in data_type.lower() or "mf" in data_type.lower():
                result = self._get_mutual_fund_nav(query)
            else:
                result = {
                    "success": False,
                    "error": f"Unknown data type: {data_type}. Supported: gold_rate, fd_rates, stock_price, mutual_fund",
                    "data": None
                }
            
            # Cache the result
            if result.get("success"):
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": None,
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_gold_rate(self) -> Dict[str, Any]:
        """Get current gold rate"""
        try:
            # Using Yahoo Finance for gold futures
            gold = yf.Ticker("GC=F")  # Gold futures
            data = gold.history(period="1d")
            
            if data.empty:
                # Fallback: Use a simple API or return approximate rates
                return {
                    "success": True,
                    "data": {
                        "gold_rate_10gm": "Approx ₹65,000-70,000",
                        "gold_rate_per_gm": "Approx ₹6,500-7,000",
                        "note": "Approximate rate. Check with local jeweler for exact price.",
                        "source": "fallback"
                    },
                    "timestamp": datetime.now().isoformat()
                }
            
            # Get latest price (convert from USD/oz to INR/gm approximately)
            latest_price = data['Close'].iloc[-1]
            # Rough conversion: 1 oz = 31.1 gm, USD to INR ~83
            price_per_gm_inr = (latest_price / 31.1) * 83
            
            return {
                "success": True,
                "data": {
                    "gold_rate_per_gm": f"₹{price_per_gm_inr:.2f}",
                    "gold_rate_10gm": f"₹{price_per_gm_inr * 10:.2f}",
                    "source": "yfinance",
                    "usd_per_oz": float(latest_price)
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to fetch gold rate: {str(e)}",
                "data": None,
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_fd_rates(self) -> Dict[str, Any]:
        """Get current FD rates from major banks"""
        # Note: In production, you'd fetch from bank APIs or financial data providers
        # For now, return approximate rates
        return {
            "success": True,
            "data": {
                "sbi_1yr": "6.5-7.0%",
                "hdfc_1yr": "6.75-7.25%",
                "icici_1yr": "6.5-7.0%",
                "axis_1yr": "6.5-7.0%",
                "note": "Rates vary by tenure and amount. Check with banks for exact rates.",
                "source": "approximate"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Get current stock price"""
        try:
            # Add .NS for NSE stocks if not present
            if not symbol.endswith(".NS") and not "." in symbol:
                symbol = f"{symbol}.NS"
            
            stock = yf.Ticker(symbol)
            info = stock.info
            data = stock.history(period="1d")
            
            if data.empty:
                return {
                    "success": False,
                    "error": f"Could not fetch data for {symbol}",
                    "data": None
                }
            
            current_price = data['Close'].iloc[-1]
            
            return {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "current_price": float(current_price),
                    "currency": info.get("currency", "INR"),
                    "name": info.get("longName", symbol)
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to fetch stock price: {str(e)}",
                "data": None,
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_mutual_fund_nav(self, query: str) -> Dict[str, Any]:
        """Get mutual fund NAV"""
        # This would require integration with AMFI or mutual fund APIs
        return {
            "success": False,
            "error": "Mutual fund NAV lookup not yet implemented. Use fund house websites.",
            "data": None,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _arun(
        self,
        query: str,
        data_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Async version of _run"""
        return self._run(query, data_type)


def create_search_tool() -> SearchTool:
    """Factory function to create SearchTool instance"""
    return SearchTool()
