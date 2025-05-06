import requests
import pandas as pd
import streamlit as st

class FinancialAPIClient:
    """
    Client for handling all communication with Financial Modeling Prep API.
    This centralizes all API calls and standardizes error handling.
    """
    
    def __init__(self, api_key):
        """Initialize the API client with an API key."""
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
    
    def _handle_response(self, response, default_return=None):
        """
        Handle API response with standardized error handling.
        Returns the first item in the response list if it's a list, or the whole response if it's not.
        Returns default_return if the response is empty or contains an error message.
        """
        try:
            if response.status_code != 200:
                st.error(f"API Error: Status code {response.status_code}")
                return default_return
                
            data = response.json()
            
            if not data:
                return default_return
                
            if isinstance(data, list) and data:
                return data[0]
            elif isinstance(data, dict) and "Error Message" in data:
                st.error(f"API Error: {data['Error Message']}")
                return default_return
                
            return data
            
        except Exception as e:
            st.error(f"Error processing API response: {e}")
            return default_return

    def get_financial_ratios(self, ticker):
        """Fetch financial ratios from FMP API."""
        endpoint = f"{self.base_url}/ratios-ttm/{ticker}?apikey={self.api_key}"
        response = requests.get(endpoint)
        return self._handle_response(response)
    
    def get_company_profile(self, ticker):
        """Fetch company profile from FMP API."""
        endpoint = f"{self.base_url}/profile/{ticker}?apikey={self.api_key}"
        response = requests.get(endpoint)
        return self._handle_response(response)
    
    def get_dcf_value(self, ticker):
        """Fetch DCF value from FMP API."""
        endpoint = f"{self.base_url}/discounted-cash-flow/{ticker}?apikey={self.api_key}"
        response = requests.get(endpoint)
        return self._handle_response(response)
    
    def get_income_statement(self, ticker, limit=1):
        """Fetch income statement data from FMP API."""
        endpoint = f"{self.base_url}/income-statement/{ticker}?limit={limit}&apikey={self.api_key}"
        response = requests.get(endpoint)
        if limit == 1:
            return self._handle_response(response)
        return response.json() if response.status_code == 200 else []
    
    def get_balance_sheet(self, ticker, limit=1):
        """Fetch balance sheet data from FMP API."""
        endpoint = f"{self.base_url}/balance-sheet-statement/{ticker}?limit={limit}&apikey={self.api_key}"
        response = requests.get(endpoint)
        if limit == 1:
            return self._handle_response(response)
        return response.json() if response.status_code == 200 else []
    
    def get_cash_flow_statement(self, ticker, limit=1):
        """Fetch cash flow statement data from FMP API."""
        endpoint = f"{self.base_url}/cash-flow-statement/{ticker}?limit={limit}&apikey={self.api_key}"
        response = requests.get(endpoint)
        if limit == 1:
            return self._handle_response(response)
        return response.json() if response.status_code == 200 else []
    
    def get_key_metrics(self, ticker):
        """Fetch key metrics from FMP API."""
        endpoint = f"{self.base_url}/key-metrics-ttm/{ticker}?apikey={self.api_key}"
        response = requests.get(endpoint)
        return self._handle_response(response)
    
    def get_historical_prices(self, ticker, period='1y'):
        """Fetch historical prices for beta and volatility calculation"""
        endpoint = f"{self.base_url}/historical-price-full/{ticker}?apikey={self.api_key}"
        response = requests.get(endpoint)
        
        try:
            data = response.json()
            if not data:
                st.error(f"No data received for {ticker}")
                return None
                
            if 'historical' not in data:
                st.error(f"Unexpected API response format for {ticker}. Response: {data}")
                return None
                
            df = pd.DataFrame(data['historical'])
            df['date'] = pd.to_datetime(df['date'])
            return df
            
        except Exception as e:
            st.error(f"Error processing historical data for {ticker}: {str(e)}")
            return None
    
    def get_technical_indicator(self, ticker, indicator_type, period=14):
        """Fetch technical indicators from FMP API."""
        endpoint = f"{self.base_url}/technical_indicator/daily/{ticker}?period={period}&type={indicator_type}&apikey={self.api_key}"
        response = requests.get(endpoint)
        return response.json() if response.status_code == 200 and response.json() else None
    
    def validate_ticker(self, ticker):
        """Verify if ticker exists and return basic info."""
        endpoint = f"{self.base_url}/quote/{ticker}?apikey={self.api_key}"
        response = requests.get(endpoint)
        return bool(response.json() if response.status_code == 200 else False)
    
    def search_tickers(self, partial_query):
        """Get stock suggestions as user types."""
        endpoint = f"{self.base_url}/search?query={partial_query}&apikey={self.api_key}"
        response = requests.get(endpoint)
        if response.status_code != 200 or not response.json():
            return []
        return [item['symbol'] for item in response.json()]
    
    @st.cache_data(ttl=3600)  # Cache results for 1 hour
    def get_sector_performance(self):
        """Fetch sector performance data."""
        endpoint = f"{self.base_url}/sectors-performance?apikey={self.api_key}"
        response = requests.get(endpoint)
        return response.json() if response.status_code == 200 else []
    
    @st.cache_data(ttl=3600)  # Cache results for 1 hour
    def get_market_index_data(self, index='^GSPC'):
        """Fetch market index (S&P 500 by default) data."""
        endpoint = f"{self.base_url}/historical-price-full/{index}?apikey={self.api_key}"
        response = requests.get(endpoint)
        return self.get_historical_prices(index)
    
    def get_company_news(self, ticker, limit=10):
        """Fetch news for a company."""
        endpoint = f"{self.base_url}/stock_news?tickers={ticker}&limit={limit}&apikey={self.api_key}"
        response = requests.get(endpoint)
        return response.json() if response.status_code == 200 else []