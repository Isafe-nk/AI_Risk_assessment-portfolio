import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt 
import os 
from dotenv import load_dotenv
from core.sector_analysis import SectorType, SectorMetrics
from core.utils import get_api_key
from core.portfolio_function import classify_stocks_alpha
from core.api_client import FinancialAPIClient
from core.technical_analysis import TechnicalAnalyzer
from core.visualization import (
    display_summary_dashboard,
    display_technical_analysis,
    display_stock_comparison,
    display_key_insights,
    display_detailed_analysis,
    display_sector_analysis,
    create_sector_visualizations,
    color_recommendation,
    display_stock_info,
    map_sector_to_type
)

# # Must be the first Streamlit command
# st.set_page_config(
#     page_title="Smart Stock Analyzer",
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

class ValuationAnalyzer:
    def __init__(self):
        self.sector_metrics = SectorMetrics()
        
        # Base weights (60% of total score)
        self.base_weights = {
            'P/E Score': 0.15,      
            'P/B Score': 0.10,      
            'PEG Score': 0.15,      
            'DCF Score': 0.20       
        }
        
        # Technical weights (20% of total score)
        self.technical_weights = {
            'RSI_Score': 0.05,     
            'MA_Score': 0.05,      
            'Growth_Score': 0.10    
        }

    def analyze_valuation(self, metrics, sector):
        """
        Analyze stock valuation using multiple metrics and return a comprehensive score
        Score range: 0 (extremely overvalued) to 100 (extremely undervalued)
        """
        try:
            sector_type = SectorType(sector)
        except ValueError:
            sector_type = SectorType.TECHNOLOGY  # Default fallback
            
        scores = {}
        
        # Get industry averages for the sector
        industry_pe = self.sector_metrics.INDUSTRY_PE[sector_type]
        industry_pb = self.sector_metrics.INDUSTRY_PB[sector_type]
        
        # Calculate scores using industry-specific benchmarks
        if metrics['P/E Ratio'] and metrics['P/E Ratio'] > 0:
            scores['P/E Score'] = self._score_pe_ratio(metrics['P/E Ratio'], industry_pe)
            
        if metrics['P/B Ratio'] and metrics['P/B Ratio'] > 0:
            scores['P/B Score'] = self._score_pb_ratio(metrics['P/B Ratio'], industry_pb)
        
        # 3. PEG Ratio Analysis (Weight: 20%)
        if metrics['PEG Ratio'] and metrics['PEG Ratio'] > 0:
            peg_score = self._score_peg_ratio(metrics['PEG Ratio'])
            scores['PEG Score'] = peg_score
        
        # 4. DCF Valuation Analysis (Weight: 30%)
        if metrics['Current Price'] and metrics['DCF Value']:
            dcf_score = self._score_dcf_value(metrics['Current Price'], metrics['DCF Value'])
            scores['DCF Score'] = dcf_score
        
        # 5. Financial Health Score (Weight: 10%)
        health_score = self._score_financial_health(metrics)
        scores['Financial Health Score'] = health_score
        
        # Calculate weighted average score
        weights = {
            'P/E Score': 0.25,
            'P/B Score': 0.15,
            'PEG Score': 0.20,
            'DCF Score': 0.30,
            'Financial Health Score': 0.10
        }
        
        final_score = 0
        valid_weights_sum = 0
        
        for metric, score in scores.items():
            if score is not None:
                final_score += score * weights[metric]
                valid_weights_sum += weights[metric]
        
        if valid_weights_sum > 0:
            final_score = final_score / valid_weights_sum
        
        # Generate recommendation
        recommendation = self.get_recommendation(metrics, final_score)  # Changed from _get_recommendation
            
        return {
            'detailed_scores': scores,
            'final_score': final_score,
            'valuation_status': self._get_valuation_status(final_score),
            'recommendation': recommendation
        }

    def _calculate_technical_score(self, metrics):
        """Calculate technical analysis score"""
        score = 50  # Start at neutral
    
        # RSI Analysis
        rsi = metrics.get('RSI')
        if rsi is not None:
            if rsi < 30:  # Oversold
                score += 20
            elif rsi < 40:
                score += 10
            elif rsi > 70:  # Overbought
                score -= 20
            elif rsi > 60:
                score -= 10
    
        # Moving Average Analysis
        ma50 = metrics.get('MA50')
        ma200 = metrics.get('MA200')
        if ma50 is not None and ma200 is not None:
            if ma50 > ma200:  # Golden Cross
                score += 15
            else:  # Death Cross
                score -= 15
    
        # Volume Analysis
        vol_avg = metrics.get('Volume_Average')
        vol_current = metrics.get('Volume_Current')
        if vol_avg is not None and vol_current is not None:
            if vol_current > vol_avg * 1.5:  # High volume
                score += 10
    
        return max(0, min(100, score))

    def _calculate_sector_score(self, metrics, sector):
        """Calculate sector-specific score"""
        try:
            sector_type = SectorType(sector)
        except ValueError:
            sector_type = SectorType.TECHNOLOGY  # Default fallback
            
        score = 50  # Start at neutral
        sector_metrics = self.sector_metrics.BASE_METRICS.copy()
        
        # Add sector-specific metrics if available
        if sector_type in self.sector_metrics.SECTOR_CONFIGS:
            sector_metrics.update(
                self.sector_metrics.SECTOR_CONFIGS[sector_type]['specific_metrics']
            )
        
        total_weight = 0
        weighted_score = 0
        
        for metric_name, config in sector_metrics.items():
            value = metrics.get(metric_name)
            if value is not None and config['threshold'] is not None:
                # Calculate metric score
                metric_score = self._score_metric(value, config['threshold'])
                weighted_score += metric_score * config['weight']
                total_weight += config['weight']
        
        # Adjust for risk factors
        risk_adjustment = self._calculate_risk_adjustment(metrics, sector_type)
        
        # Calculate final score
        if total_weight > 0:
            final_score = (weighted_score / total_weight) * 0.8 + risk_adjustment * 0.2
            return max(0, min(100, final_score))
        
        return score

    def _score_metric(self, value, threshold):
        """Convert metric value to score between 0 and 100"""
        if threshold == 0:
            return 50
        
        relative_performance = value / threshold
        if relative_performance >= 2:
            return 100
        elif relative_performance >= 1:
            return 75 + (relative_performance - 1) * 25
        else:
            return max(0, relative_performance * 75)

    def _calculate_risk_adjustment(self, metrics, sector_type):
        """Calculate risk adjustment based on sector-specific risk factors"""
        risk_score = 50  # Neutral starting point
        
        # Get sector-specific risk factors
        sector_config = self.sector_metrics.SECTOR_CONFIGS.get(sector_type)
        if not sector_config or 'risk_factors' not in sector_config:
            return risk_score
        
        risk_factors = sector_config['risk_factors']
        
        # Technology sector risk adjustments
        if sector_type == SectorType.TECHNOLOGY:
            # Tech obsolescence risk
            if metrics.get('R&D_Ratio', 0) < 0.05:
                risk_score -= 10
            elif metrics.get('R&D_Ratio', 0) > 0.15:
                risk_score += 10
                
            # Cybersecurity risk
            if metrics.get('Security_Investment_Ratio', 0) > 0.05:
                risk_score += 5
                
            # Competition risk
            if metrics.get('Market_Share', 0) > 0.20:
                risk_score += 10
                
        # Financial sector risk adjustments
        elif sector_type == SectorType.FINANCIAL:
            # Interest rate risk
            if metrics.get('Interest_Rate_Sensitivity', 0) > 0.5:
                risk_score -= 10
                
            # Credit risk
            if metrics.get('NPL_Ratio', 0) < 0.02:
                risk_score += 10
            elif metrics.get('NPL_Ratio', 0) > 0.05:
                risk_score -= 10
                
            # Capital adequacy
            if metrics.get('Capital_Adequacy_Ratio', 0) > 0.12:
                risk_score += 10
                
        # Healthcare sector risk adjustments
        elif sector_type == SectorType.HEALTHCARE:
            # Regulatory risk
            if metrics.get('Regulatory_Compliance_Score', 0) > 0.8:
                risk_score += 10
                
            # Patent expiry risk
            if metrics.get('Patent_Protection_Years', 0) > 5:
                risk_score += 5
                
            # Clinical trial risk
            if metrics.get('Clinical_Trial_Success_Rate', 0) > 0.6:
                risk_score += 10
                
        # Energy sector risk adjustments
        elif sector_type == SectorType.ENERGY:
            # Environmental risk
            if metrics.get('ESG_Score', 0) > 70:
                risk_score += 10
                
            # Resource depletion risk
            if metrics.get('Reserve_Life', 0) > 15:
                risk_score += 10
                
            # Production efficiency
            if metrics.get('Production_Cost', 0) < self.sector_metrics.BASE_METRICS['Operating_Margin']['threshold']:
                risk_score += 5
        
        # Add similar conditions for other sectors...
        
        return max(0, min(100, risk_score))

    def get_sector_metrics_breakdown(self, metrics, sector_type):
        """Get detailed breakdown of sector-specific metrics"""
        breakdown = {
            'base_metrics': {},
            'sector_specific': {},
            'risk_factors': {}
        }
        
        # Add base metrics
        for metric, config in self.sector_metrics.BASE_METRICS.items():
            value = metrics.get(metric)
            if value is not None:
                breakdown['base_metrics'][metric] = {
                    'value': value,
                    'threshold': config['threshold'],
                    'score': self._score_metric(value, config['threshold'])
                }
        
        # Add sector-specific metrics
        sector_config = self.sector_metrics.SECTOR_CONFIGS.get(sector_type)
        if sector_config and 'specific_metrics' in sector_config:
            for metric, config in sector_config['specific_metrics'].items():
                value = metrics.get(metric)
                if value is not None:
                    breakdown['sector_specific'][metric] = {
                        'value': value,
                        'threshold': config['threshold'],
                        'score': self._score_metric(value, config['threshold'])
                    }
        
        # Add risk factors
        if sector_config and 'risk_factors' in sector_config:
            for risk_factor in sector_config['risk_factors']:
                value = metrics.get(f'{risk_factor}_Risk')
                if value is not None:
                    breakdown['risk_factors'][risk_factor] = value
        
        return breakdown

    def _score_pe_ratio(self, pe_ratio, industry_avg):
        """Score P/E ratio relative to industry average"""
        if pe_ratio <= 0:
            return None
            
        if pe_ratio < industry_avg:
            return min(100, (1 - (pe_ratio / industry_avg)) * 100 + 50)
        else:
            return max(0, (2 - (pe_ratio / industry_avg)) * 50)
    
    def _score_pb_ratio(self, pb_ratio, industry_avg):
        """Score P/B ratio relative to industry average"""
        if pb_ratio <= 0:
            return None
            
        if pb_ratio < industry_avg:
            return min(100, (1 - (pb_ratio / industry_avg)) * 100 + 50)
        else:
            return max(0, (2 - (pb_ratio / industry_avg)) * 50)
    
    def _score_peg_ratio(self, peg_ratio):
        """Score PEG ratio (1 is considered fair value)"""
        if peg_ratio <= 0:
            return None
            
        if peg_ratio < 1:
            return min(100, (1 - peg_ratio) * 50 + 50)
        else:
            return max(0, (2 - peg_ratio) * 50)
    
    def _score_dcf_value(self, current_price, dcf_value):
        """Score based on DCF valuation"""
        if current_price <= 0 or dcf_value <= 0:
            return None
            
        ratio = current_price / dcf_value
        if ratio < 1:
            return min(100, (1 - ratio) * 100 + 50)
        else:
            return max(0, (2 - ratio) * 50)
    
    def _score_financial_health(self, metrics):
        """Score overall financial health"""
        score = 50  # Start at neutral
        
        # Check Free Cash Flow
        if metrics.get('FCF'):
            if metrics['FCF'] > 0:
                score += 10
            else:
                score -= 10
        
        # Check Debt/Equity
        if metrics.get('Debt/Equity'):
            if metrics['Debt/Equity'] < 1:
                score += 10
            elif metrics['Debt/Equity'] > 2:
                score -= 10
        
        return max(0, min(100, score))
    
    def _get_valuation_status(self, score):
        """Convert numerical score to valuation status"""
        if score >= 80:
            return "Significantly Undervalued"
        elif score >= 60:
            return "Moderately Undervalued"
        elif score >= 40:
            return "Fairly Valued"
        elif score >= 20:
            return "Moderately Overvalued"
        else:
            return "Significantly Overvalued"
    
    def get_recommendation(self, metrics, score):  # Changed from _get_recommendation to get_recommendation
        """Generate investment recommendation based on multiple factors"""
        # Base recommendation on valuation score
        if score >= 80:
            base_rec = "Strong Buy"
        elif score >= 60:
            base_rec = "Buy"
        elif score >= 40:
            base_rec = "Hold"
        elif score >= 20:
            base_rec = "Sell"
        else:
            base_rec = "Strong Sell"
        
        # Adjust recommendation based on additional factors
        adjustment_points = 0
        
        # Financial Health Adjustments
        fcf = metrics.get('FCF')
        if fcf is not None and fcf > 0:
            adjustment_points += 1

        debt_equity = metrics.get('Debt/Equity')
        if debt_equity is not None and debt_equity < 1:
            adjustment_points += 1
        
        # Risk Adjustments
        beta = metrics.get('Beta')
        if beta is not None:
            if beta < 0.8:  # Low volatility
                adjustment_points += 1
            elif beta > 1.5:  # High volatility
                adjustment_points -= 1
        
        # Dividend Consideration
        div_yield = metrics.get('Dividend Yield')
        if div_yield is not None and div_yield > 0.02:  # 2% yield threshold
            adjustment_points += 1
        
        # DCF Value vs Current Price
        current_price = metrics.get('Current Price')
        dcf_value = metrics.get('DCF Value')
        if current_price is not None and dcf_value is not None and current_price > 0:
            dcf_premium = (dcf_value - current_price) / current_price
            if dcf_premium > 0.3:  # 30% upside
                adjustment_points += 1
            elif dcf_premium < -0.3:  # 30% downside
                adjustment_points -= 1
        
        # Adjust final recommendation based on points
        rec_scale = {
            "Strong Buy": 2,
            "Buy": 1,
            "Hold": 0,
            "Sell": -1,
            "Strong Sell": -2
        }
        
        base_value = rec_scale[base_rec]
        adjusted_value = base_value + (adjustment_points * 0.5)  # Scale adjustment impact
        
        # Convert back to recommendation
        for rec, value in rec_scale.items():
            if adjusted_value >= value - 0.25:
                return rec
        
        return base_rec

class StockScreener:
    def __init__(self, api_key):
        self.api_client = FinancialAPIClient(api_key)
        self.sector_metrics = SectorMetrics()
        self.valuation_analyzer = ValuationAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer(self.api_client)
    
    def analyze_stock(self, ticker, sector):
        """Perform comprehensive stock analysis with enhanced metrics"""
        try:
            # Get base metrics using the API client
            ratios = self.api_client.get_financial_ratios(ticker)
            profile = self.api_client.get_company_profile(ticker)
            dcf = self.api_client.get_dcf_value(ticker)
            
            if not ratios:
                st.warning(f"Financial ratios not found for {ticker}. Skipping analysis.")
            if not profile:
                st.warning(f"Company profile not found for {ticker}. Skipping analysis.")
            if not dcf:
                st.warning(f"DCF value not found for {ticker}. Skipping analysis.")
            
            if not all([ratios, profile, dcf]):
                st.error(f"Missing required base data for {ticker}")
                return None
                
            # Get enhanced metrics
            composite_metrics = self._calculate_composite_metrics(ticker, sector)
            
            # Get technical indicators using the dedicated technical analyzer
            technical_indicators = self.technical_analyzer.get_indicators(ticker)
            technical_score = self.technical_analyzer.calculate_technical_score(technical_indicators) if technical_indicators else 50
            
            # Get sector-specific metrics
            sector_metrics = self.get_sector_metrics(ticker, sector)
            sector_score = self._calculate_sector_score(sector_metrics, sector) if sector_metrics else 50
            
            # Get raw sector from profile for display
            raw_sector = profile.get('sector', 'Unknown')
            
            # Calculate valuation score
            valuation_score = self._calculate_valuation_score(ratios, dcf, sector)
            
            # Calculate additional risk metrics
            volatility = self.technical_analyzer.calculate_volatility(ticker, days=30)
            beta = self.technical_analyzer.calculate_beta(ticker)
            
            # Get chart patterns
            patterns = self.technical_analyzer.detect_patterns(ticker)

            historical_prices = self.api_client.get_historical_prices(ticker, period="5d")  # Get 5 days of data for better accuracy
            if historical_prices is not None and not historical_prices.empty and len(historical_prices) >= 2:
                # Get the most recent close price (first row since data is sorted newest first)
                close_price_value = historical_prices['close'].iloc[0]
                # Get the previous day's close price
                prev_close = historical_prices['close'].iloc[1]
                daily_return_value = (close_price_value - prev_close) / prev_close if prev_close != 0 else 0
            else:
                close_price_value = profile.get('price', 0)
                daily_return_value = 0
            
            # Combine all metrics
            metrics = {
                'Ticker': ticker,
                'Company Name': profile.get('companyName', 'Unknown'),
                'Sector': raw_sector,  # Use raw sector name from API
                'Current Price': profile.get('price', 0),
                'Market Cap': profile.get('mktCap', 0),
                'P/E Ratio': ratios.get('peRatioTTM', 0),
                'P/B Ratio': ratios.get('priceToBookRatioTTM', 0),
                'Valuation Score': valuation_score,
                'Technical Score': technical_score,
                'Sector Score': sector_score,
                'RSI': technical_indicators.get('RSI', 50) if technical_indicators else 50,
                'MA50': technical_indicators.get('MA50', 0) if technical_indicators else 0,
                'MA200': technical_indicators.get('MA200', 0) if technical_indicators else 0,
                'Volume': technical_indicators.get('Volume', 0) if technical_indicators else 0,
                'Daily_Return': daily_return_value,
                'Close': close_price_value,
                '30D Volatility': volatility,
                'Recommendation': self._generate_recommendation(
                    valuation_score=valuation_score,
                    technical_score=technical_score,
                    sector_score=sector_score
                )
            }
            
            # Add pattern detection results if available
            if patterns:
                for pattern_name, detected in patterns.items():
                    metrics[f"Pattern_{pattern_name}"] = detected
                
            # Generate recommendation based on all metrics
            metrics['Recommendation'] = self._generate_recommendation(
                valuation_score=valuation_score,
                technical_score=technical_score,
                sector_score=sector_score
            )
            
            # Add any sector-specific metrics if available
            if sector_metrics:
                metrics.update({
                    'Revenue': sector_metrics.get('Revenue', 0),
                    'Operating Margin': sector_metrics.get('Operating Margin', 0),
                    'Asset Turnover': sector_metrics.get('Asset Turnover', 0),
                    'Debt to Equity': sector_metrics.get('Debt to Equity', 0)
                })
            
            return metrics
            
        except Exception as e:
            st.error(f"Failed to analyze {ticker}: {str(e)}")
            return None

    def get_technical_indicators(self, ticker):
        """Get technical indicators for the stock using the TechnicalAnalyzer"""
        return self.technical_analyzer.get_indicators(ticker)

    def calculate_30d_volatility(self, ticker):
        """Calculate 30-day annualized volatility for a stock using the TechnicalAnalyzer"""
        return self.technical_analyzer.calculate_volatility(ticker, days=30)

    def calculate_beta(self, ticker, market_index='^GSPC'):
        """Calculate beta relative to S&P 500 using the TechnicalAnalyzer"""
        return self.technical_analyzer.calculate_beta(ticker, market_index)

    def validate_ticker(self, ticker):
        """Verify if ticker exists and get basic info"""
        return self.api_client.validate_ticker(ticker)

    def search_suggestions(self, partial_query):
        """Get stock suggestions as user types"""
        return self.api_client.search_tickers(partial_query)
        
    # Keep other methods that don't directly interact with the API

    def _calculate_composite_metrics(self, ticker, sector):
        """Calculate composite metrics from multiple data sources"""
        try:
            metrics = {}
            
            # Get base financial data
            income_stmt = self.api_client.get_income_statement(ticker)
            balance_sheet = self.api_client.get_balance_sheet(ticker)
            cash_flow = self.api_client.get_cash_flow_statement(ticker)
            key_metrics = self.api_client.get_key_metrics(ticker)
            
            if all([income_stmt, balance_sheet, cash_flow, key_metrics]):
                # Financial Health Score
                metrics['Financial_Health'] = self._calculate_financial_health(
                    income_stmt, balance_sheet, cash_flow
                )
                
                # Growth Metrics
                metrics['Revenue_Growth_3Y'] = self._calculate_cagr(ticker, 'revenue', years=3)
                metrics['Earnings_Growth_3Y'] = self._calculate_cagr(ticker, 'netIncome', years=3)
                
                # Quality Metrics
                metrics['Earnings_Quality'] = self._calculate_earnings_quality(
                    income_stmt, cash_flow
                )
                
                # Sector-specific composite metrics
                try:
                    sector_type = SectorType(sector)
                except ValueError:
                    sector_type = SectorType.TECHNOLOGY  # Default fallback
                    
                if sector_type == SectorType.TECHNOLOGY:
                    metrics.update(self._calculate_tech_metrics(ticker, income_stmt, balance_sheet))
                elif sector_type == SectorType.FINANCIAL:
                    metrics.update(self._calculate_financial_metrics(ticker, income_stmt, balance_sheet))
                elif sector_type == SectorType.HEALTHCARE:
                    metrics.update(self._calculate_healthcare_metrics(ticker, income_stmt, balance_sheet))
            
            return metrics
            
        except Exception as e:
            st.warning(f"Error calculating composite metrics: {e}")
            return {}
    
    def _calculate_financial_health(self, income_stmt, balance_sheet, cash_flow):
        """Calculate comprehensive financial health score"""
        try:
            score = 50  # Base score
            
            # Profitability checks
            gross_profit = income_stmt.get('grossProfit', 0)
            revenue = income_stmt.get('revenue', 1)
            gross_margin = gross_profit / revenue if revenue > 0 else 0
            
            if gross_margin > 0.3:
                score += 10
                
            # Liquidity checks
            current_assets = balance_sheet.get('totalCurrentAssets', 0)
            current_liabilities = balance_sheet.get('totalCurrentLiabilities', 1)
            current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
            
            if current_ratio > 1.5:
                score += 10
            elif current_ratio < 1:
                score -= 10
                
            # Cash Flow Quality
            net_income = income_stmt.get('netIncome', 0)
            operating_cash_flow = cash_flow.get('operatingCashFlow', 0)
            
            if net_income > 0 and operating_cash_flow > net_income:
                score += 10
                
            return max(0, min(100, score))  # Ensure score is between 0-100
            
        except Exception as e:
            st.warning(f"Error calculating financial health: {e}")
            return 50  # Return neutral score on error
    
    def _calculate_cagr(self, ticker, metric_name, years=3):
        """Calculate Compound Annual Growth Rate for a given metric"""
        try:
            statements = self.api_client.get_income_statement(ticker, limit=years+1)
            
            if not statements or len(statements) < years+1:
                return 0
                
            start_value = statements[years].get(metric_name, 0)
            end_value = statements[0].get(metric_name, 0)
            
            if start_value <= 0:
                return 0
                
            growth_rate = (end_value / start_value) ** (1/years) - 1
            return growth_rate * 100  # Return as percentage
            
        except Exception as e:
            st.warning(f"Error calculating CAGR: {e}")
            return 0
    
    def _calculate_earnings_quality(self, income_stmt, cash_flow):
        """Assess earnings quality through various metrics"""
        try:
            net_income = income_stmt.get('netIncome', 0)
            operating_cash_flow = cash_flow.get('operatingCashFlow', 0)
            
            if net_income <= 0:
                return 0
                
            # Calculate accruals ratio
            accruals = (operating_cash_flow - net_income) / abs(net_income)
            
            # Score based on accruals ratio (positive is better)
            if accruals > 0:
                return min(100, 50 + accruals * 50)  # Higher score for positive cash flow vs earnings
            else:
                return max(0, 50 + accruals * 50)
                
        except Exception as e:
            st.warning(f"Error calculating earnings quality: {e}")
            return 50  # Return neutral score on error
    
    def _calculate_tech_metrics(self, ticker, income_stmt, balance_sheet):
        """Calculate technology sector-specific metrics"""
        metrics = {}
        
        try:
            revenue = income_stmt.get('revenue', 1)
            rd_expense = income_stmt.get('researchAndDevelopmentExpenses', 0)
            
            if revenue > 0:
                metrics['R&D_Ratio'] = rd_expense / revenue
                
            # Use intangible assets as a proxy for innovation/IP
            intangible_assets = balance_sheet.get('intangibleAssets', 0)
            total_assets = balance_sheet.get('totalAssets', 1)
            
            if total_assets > 0:
                metrics['Intangible_Assets_Ratio'] = intangible_assets / total_assets
                
            return metrics
            
        except Exception as e:
            st.warning(f"Error calculating tech metrics: {e}")
            return {}
    
    def _calculate_financial_metrics(self, ticker, income_stmt, balance_sheet):
        """Calculate financial sector-specific metrics"""
        metrics = {}
        
        try:
            total_assets = balance_sheet.get('totalAssets', 1)
            net_income = income_stmt.get('netIncome', 0)
            
            if total_assets > 0:
                # Return on Assets - key metric for financial companies
                metrics['ROA'] = net_income / total_assets
                
            # Check for financial sector specific items
            if 'netInterestIncome' in income_stmt:
                net_interest_income = income_stmt.get('netInterestIncome', 0)
                total_deposits = balance_sheet.get('deposits', 1)
                
                if total_deposits > 0:
                    metrics['Net_Interest_Margin'] = net_interest_income / total_deposits
                    
            return metrics
            
        except Exception as e:
            st.warning(f"Error calculating financial metrics: {e}")
            return {}
    
    def _calculate_healthcare_metrics(self, ticker, income_stmt, balance_sheet):
        """Calculate healthcare sector-specific metrics"""
        metrics = {}
        
        try:
            revenue = income_stmt.get('revenue', 1)
            rd_expense = income_stmt.get('researchAndDevelopmentExpenses', 0)
            operating_income = income_stmt.get('operatingIncome', 0)
            
            if revenue > 0:
                # R&D is crucial for healthcare companies
                metrics['R&D_Ratio'] = rd_expense / revenue
                metrics['Operating_Margin'] = operating_income / revenue
                
            return metrics
            
        except Exception as e:
            st.warning(f"Error calculating healthcare metrics: {e}")
            return {}
        
    def _calculate_valuation_score(self, ratios, dcf, sector):
        """Calculate valuation score using various metrics"""
        try:
            # Initialize score with neutral value
            score = 50
            
            # 1. P/E Ratio analysis
            pe_ratio = ratios.get('peRatioTTM', 0)
            if pe_ratio > 0:
                # Get sector average P/E
                try:
                    sector_type = SectorType(sector)
                    sector_pe = self.sector_metrics.INDUSTRY_PE.get(sector_type, 20)
                except:
                    sector_pe = 20  # Default if sector not found
                
                # Compare P/E to sector average
                pe_ratio_rel = pe_ratio / sector_pe
                if pe_ratio_rel < 0.7:  # Significantly below average (good)
                    score += 15
                elif pe_ratio_rel < 0.9:  # Somewhat below average (good)
                    score += 10
                elif pe_ratio_rel > 1.5:  # Significantly above average (bad)
                    score -= 15
                elif pe_ratio_rel > 1.2:  # Somewhat above average (bad)
                    score -= 10
            
            # 2. DCF Valuation analysis
            price = dcf.get('price', 0) 
            dcf_value = dcf.get('dcf', 0)
            
            if price > 0 and dcf_value > 0:
                # Calculate upside/downside potential
                potential = (dcf_value - price) / price
                if potential > 0.3:  # >30% upside
                    score += 20
                elif potential > 0.1:  # >10% upside
                    score += 10
                elif potential < -0.2:  # >20% downside
                    score -= 15
                elif potential < -0.1:  # >10% downside
                    score -= 10
            
            # 3. P/B Ratio analysis
            pb_ratio = ratios.get('priceToBookRatioTTM', 0)
            if pb_ratio > 0:
                # Get sector average P/B
                try:
                    sector_type = SectorType(sector)
                    sector_pb = self.sector_metrics.INDUSTRY_PB.get(sector_type, 2)
                except:
                    sector_pb = 2  # Default if sector not found
                
                # Compare P/B to sector average
                pb_ratio_rel = pb_ratio / sector_pb
                if pb_ratio_rel < 0.7:
                    score += 10
                elif pb_ratio_rel > 1.5:
                    score -= 10
            
            # Ensure score is within valid range
            return max(0, min(100, score))
            
        except Exception as e:
            st.warning(f"Error calculating valuation score: {e}")
            return 50  # Return neutral score if error occurs

    def get_sector_metrics(self, ticker, sector):
        """Get sector-specific metrics"""
        try:
            # Get financial statements using API client
            income_stmt = self.api_client.get_income_statement(ticker)
            balance_sheet = self.api_client.get_balance_sheet(ticker)
            
            if not income_stmt or not balance_sheet:
                return None
                
            # Calculate metrics
            revenue = float(income_stmt.get('revenue', 0))
            operating_income = float(income_stmt.get('operatingIncome', 0))
            total_assets = float(balance_sheet.get('totalAssets', 1))
            total_equity = float(balance_sheet.get('totalStockholdersEquity', 1))
            total_debt = float(balance_sheet.get('totalDebt', 0))
            
            # Calculate sector-specific ratios
            metrics = {
                'Revenue': revenue,
                'Operating Margin': operating_income / revenue if revenue > 0 else 0,
                'Asset Turnover': revenue / total_assets if total_assets > 0 else 0,
                'Debt to Equity': total_debt / total_equity if total_equity > 0 else 0
            }
            
            # Add sector-specific additional metrics
            try:
                sector_type = SectorType(sector)
            except ValueError:
                sector_type = SectorType.TECHNOLOGY  # Default fallback
                
            if sector_type == SectorType.TECHNOLOGY:
                # Calculate R&D ratio for tech companies
                rd_expense = income_stmt.get('researchAndDevelopmentExpenses', 0)
                metrics['R&D Ratio'] = rd_expense / revenue if revenue > 0 else 0
                
                # Calculate growth metrics
                growth_data = self._calculate_growth_metrics(ticker)
                if growth_data:
                    metrics['Revenue Growth'] = growth_data.get('revenue_growth', 0)
                
            elif sector_type == SectorType.FINANCIAL:
                # Add financial-specific metrics
                cash_flow = self.api_client.get_cash_flow_statement(ticker)
                if cash_flow:
                    metrics['Cash Flow from Operations'] = cash_flow.get('operatingCashFlow', 0)
                    metrics['Net Interest Margin'] = income_stmt.get('netInterestIncome', 0) / total_assets if total_assets > 0 else 0
                    metrics['ROA'] = income_stmt.get('netIncome', 0) / total_assets if total_assets > 0 else 0
            
            elif sector_type == SectorType.HEALTHCARE:
                # Healthcare specific metrics
                rd_expense = income_stmt.get('researchAndDevelopmentExpenses', 0)
                metrics['R&D Ratio'] = rd_expense / revenue if revenue > 0 else 0
                metrics['Gross Margin'] = income_stmt.get('grossProfit', 0) / revenue if revenue > 0 else 0
            
            return metrics
        
        except Exception as e:
            st.warning(f"Error getting sector metrics for {ticker}: {e}")
            return None
            
    def _calculate_growth_metrics(self, ticker):
        """Calculate growth metrics using multi-year data"""
        try:
            # Get 4 years of income statements
            income_statements = self.api_client.get_income_statement(ticker, 4)
            
            if not income_statements or len(income_statements) < 2:
                return {'revenue_growth': 0}
                
            # Calculate year-over-year revenue growth
            revenue_current = income_statements[0].get('revenue', 0)
            revenue_previous = income_statements[1].get('revenue', 0)
            
            if revenue_previous > 0:
                revenue_growth = (revenue_current - revenue_previous) / revenue_previous
            else:
                revenue_growth = 0
                
            return {'revenue_growth': revenue_growth}
        except Exception as e:
            st.warning(f"Error calculating growth metrics: {e}")
            return {'revenue_growth': 0}

    def _calculate_sector_score(self, metrics, sector):
        """Calculate sector-specific score"""
        try:
            sector_type = SectorType(sector)
        except ValueError:
            sector_type = SectorType.TECHNOLOGY  # Default fallback
            
        score = 50  # Start at neutral
        sector_metrics = self.sector_metrics.BASE_METRICS.copy()
        
        # Add sector-specific metrics if available
        if sector_type in self.sector_metrics.SECTOR_CONFIGS:
            sector_metrics.update(
                self.sector_metrics.SECTOR_CONFIGS[sector_type]['specific_metrics']
            )
        
        total_weight = 0
        weighted_score = 0
        
        for metric_name, config in sector_metrics.items():
            value = metrics.get(metric_name)
            if value is not None and config['threshold'] is not None:
                # Calculate metric score
                metric_score = self._score_metric(value, config['threshold'])
                weighted_score += metric_score * config['weight']
                total_weight += config['weight']
        
        # Calculate final score
        if total_weight > 0:
            final_score = weighted_score / total_weight
            return max(0, min(100, final_score))
        
        return score
        
    def _score_metric(self, value, threshold):
        """Convert metric value to score between 0 and 100"""
        if threshold == 0:
            return 50
        
        relative_performance = value / threshold
        if relative_performance >= 2:
            return 100
        elif relative_performance >= 1:
            return 75 + (relative_performance - 1) * 25
        else:
            return max(0, relative_performance * 75)

    def _generate_recommendation(self, valuation_score, technical_score, sector_score):
        """Generate final recommendation based on multiple scores"""
        # Weight the different scores
        # 60% fundamental valuation, 25% technical, 15% sector
        weighted_score = (valuation_score * 0.6) + (technical_score * 0.25) + (sector_score * 0.15)
        
        # Convert to recommendation
        if weighted_score >= 80:
            return "Strong Buy"
        elif weighted_score >= 60:
            return "Buy"
        elif weighted_score >= 40:
            return "Hold"
        elif weighted_score >= 20:
            return "Sell"
        else:
            return "Strong Sell"

def main():
    st.title("Advanced Stock Screener with Enhanced Analysis")

    # Add description
    st.markdown("""
    This stock screener helps you analyze stocks across different sectors using:
    - 📊 Fundamental Analysis (P/E ratios, margins, growth)
    - 📈 Technical Analysis (RSI, Moving Averages)
    - 🏢 Sector-Specific Metrics
    - ⚠️ Risk Assessment
    """)
    
    # Create tabs for main content and educational content
    main_tab, education_tab = st.tabs(["Stock Analysis", "Learn More"])
    
    # Educational Content
    with education_tab:
        st.header("📚 Understanding Stock Analysis")
        
        with st.expander("Valuation Metrics Explained"):
            st.markdown("""
            ### Valuation Score (0-100)
            - **80+ : Significantly Undervalued** - Stock may be trading well below its fair value
            - **60-79: Moderately Undervalued** - Stock appears somewhat undervalued
            - **40-59: Fairly Valued** - Stock is trading close to its fair value
            - **20-39: Moderately Overvalued** - Stock appears somewhat overvalued
            - **<20: Significantly Overvalued** - Stock may be trading well above its fair value
            
            The valuation score combines multiple factors including:
            - P/E Ratio (Price to Earnings)
            - P/B Ratio (Price to Book)
            - PEG Ratio (Price/Earnings to Growth)
            - DCF Valuation (Discounted Cash Flow)
            """)
            
        with st.expander("Technical Analysis Explained"):
            st.markdown("""
            ### Technical Indicators
            - **RSI (Relative Strength Index)**
                - Above 70: Potentially overbought
                - Below 30: Potentially oversold
                - Between 30-70: Neutral territory
            
            - **Moving Averages**
                - Golden Cross (50-day MA crosses above 200-day MA): Bullish signal
                - Death Cross (50-day MA crosses below 200-day MA): Bearish signal
            """)
            
        with st.expander("Understanding Sector Analysis"):
            st.markdown("""
            ### Sector-Specific Metrics
            Different sectors require different analytical approaches:
            
            - **Technology**: R&D spending, margin trends, innovation metrics
            - **Financial**: Interest margins, loan quality, capital ratios
            - **Healthcare**: Drug pipeline, R&D efficiency, regulatory risks
            - **Consumer**: Brand value, market share, inventory turnover
            """)
    
    # Main Analysis Tab
    with main_tab:
        try:
            api_key = get_api_key()
            screener = StockScreener(api_key)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        
        # Sidebar with enhanced help
        with st.sidebar:
            st.header("Stock Selection")
            
            st.info("💡 How to use this screener:")
            st.markdown("""
            1. Enter a stock symbol (e.g., AAPL for Apple)
            2. Click 'Add to Analysis List'
            3. Add multiple stocks to compare
            4. Click 'Analyze Stocks' to see results
            """)
            
            # Stock search and add functionality
            stock_search = st.text_input(
                "Search for a stock (e.g., AAPL, TD.TO, SHOP.TO)",
                help="Enter ticker symbol. Add .TO for Canadian stocks"
            )
            
            # Initialize session state for watchlist
            if 'watchlist' not in st.session_state:
                st.session_state.watchlist = []
            
            # Add stock button with validation
            if st.button("Add to Analysis List") and stock_search:
                if stock_search not in st.session_state.watchlist:
                    with st.spinner(f"Validating {stock_search}..."):
                        try:
                            if screener.validate_ticker(stock_search):
                                st.session_state.watchlist.append(stock_search)
                                display_stock_info(stock_search, screener)
                            else:
                                st.error("Invalid ticker symbol")
                        except Exception as e:
                            if "API call limit" in str(e).lower():
                                st.error("API call limit reached. Please try again later or upgrade your API plan.")
                            else:
                                st.error(f"Error validating ticker: {str(e)}")
            
            # Display and manage watchlist
            st.subheader("Analysis List")
            for stock in st.session_state.watchlist:
                col1, col2 = st.columns([3, 1])
                col1.write(stock)
                if col2.button("🗑️", key=f"remove_{stock}"):
                    st.session_state.watchlist.remove(stock)
            
            if st.button("Clear List"):
                st.session_state.watchlist = []

            # Analysis button (moved back to sidebar)
            if st.session_state.watchlist:
                if st.button("Analyze Stocks"):
                    st.session_state.analyze_clicked = True
            else:
                st.info("Add stocks to analyze")

        # Main content area (analysis results only)
        if st.session_state.watchlist:
            if getattr(st.session_state, 'analyze_clicked', False):
                with st.spinner('Performing comprehensive analysis...'):
                    results = []
                    progress_bar = st.progress(0)
                    api_limit_reached = False
                    
                    for i, ticker in enumerate(st.session_state.watchlist):
                        try:
                            profile = screener.api_client.get_company_profile(ticker)
                            if profile:
                                sector = profile.get('sector')
                                sector_type = map_sector_to_type(sector)
                                analysis = screener.analyze_stock(ticker, sector_type)
                                if analysis:
                                    results.append(analysis)
                            progress_bar.progress((i + 1) / len(st.session_state.watchlist))
                        except Exception as e:
                            if "API call limit" in str(e).lower():
                                api_limit_reached = True
                                st.error(f"API call limit reached while analyzing {ticker}. Please try again later or upgrade your API plan.")
                                break
                            else:
                                st.error(f"Error analyzing {ticker}: {str(e)}")
                    
                    if api_limit_reached:
                        st.warning("Analysis incomplete due to API limit. Some stocks may not have been analyzed.")
                        if results:  # If we have partial results, show them
                            st.info("Showing partial results from completed analyses.")
                    
                    if results:  # Check if we have any results
                        df = pd.DataFrame(results)
                        
                        # Define the expected columns in the correct order
                        expected_columns = [
                            'Ticker', 'Company Name', 'Sector', 'Current Price', 'Market Cap',
                            'P/E Ratio', 'P/B Ratio', 'PEG Ratio', 'DCF Value',
                            'Valuation Score', 'Technical Score', 'Sector Score',
                            'RSI', 'MA50', 'MA200', 'MACD', 'Signal_Line', 'Volume',
                            '30D Volatility', 'Beta', 'Debt/Equity', 'FCF', 'Dividend Yield',
                            'Recommendation', 'Revenue', 'Operating Margin', 'Asset Turnover', 
                            'Daily_Return', 'Close', 'Risk Category'
                        ]
                        
                        # Reorder columns to match expected order
                        available_columns = [col for col in expected_columns if col in df.columns]
                        df = df[available_columns]
                        
                        # Add risk category if not present
                        if 'Risk Category' not in df.columns:
                            df["Risk Category"] = df.apply(classify_stocks_alpha, axis=1)
                        
                        # Ensure numeric columns are properly formatted
                        numeric_columns = [
                            'Current Price', 'Market Cap', 'P/E Ratio', 'P/B Ratio',
                            'PEG Ratio', 'DCF Value', 'Valuation Score', 'Technical Score',
                            'Sector Score', 'RSI', 'MA50', 'MA200', 'MACD', 'Signal_Line',
                            'Volume', '30D Volatility', 'Beta', 'Debt/Equity', 'FCF',
                            'Dividend Yield', 'Revenue', 'Operating Margin', 'Asset Turnover'
                        ]
                        
                        for col in numeric_columns:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                                # Fill missing values with 0 for numeric columns
                                df[col] = df[col].fillna(0)
                        
                        # Save to CSV with consistent column order
                        csv_path = "classified_stocks.csv"
                        write_header =not os.path.exists(csv_path)
                        df.to_csv(csv_path, mode='a', header=write_header, index=False)
                        st.success("Stocks classified and saved for portfolio generation.")

                        # Create tabs for different views
                        tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            "Summary Dashboard",
                            "Technical Analysis",
                            "Sector Analysis",
                            "Stock Comparison",
                            "Detailed Analysis"
                        ])
                        
                        with tab1:
                            display_summary_dashboard(df)
                        
                        with tab2:
                            display_technical_analysis(df)
                        
                        with tab3:
                            display_sector_analysis(df, screener)
                        
                        with tab4:
                            display_stock_comparison(df)
                        
                        with tab5:
                            display_detailed_analysis(df)
                    else:
                        st.warning("No analysis results were generated. This might be due to API limits or errors in data retrieval.")
        else:
            st.info("👈 Add stocks to your analysis list using the sidebar")

def display_summary_dashboard(df):
    """Display enhanced summary dashboard"""
    st.subheader("📊 Market Overview")
    
    # Enhanced metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_score = df['Valuation Score'].mean()
        delta = avg_score - 50
        st.metric(
            "Average Valuation Score", 
            f"{avg_score:.1f}",
            delta=f"{delta:.1f} vs Neutral",
            help="Scores above 50 indicate undervaluation"
        )
    
    with col2:
        best_stock = df.loc[df['Valuation Score'].idxmax()]
        st.metric(
            "Top Pick", 
            best_stock['Ticker'],
            f"Score: {best_stock['Valuation Score']:.1f}",
            help="Stock with the highest overall score"
        )
    
    with col3:
        avg_tech_score = df['Technical Score'].mean()
        tech_delta = avg_tech_score - 50
        st.metric(
            "Average Technical Score",
            f"{avg_tech_score:.1f}",
            delta=f"{tech_delta:.1f} vs Neutral",
            help="Technical strength indicator"
        )
    
    with col4:
        avg_volatility = df['30D Volatility'].mean()
        st.metric(
            "Average 30D Volatility",
            f"{avg_volatility:.2%}",
            help="Average annualized volatility across selected stocks"
        )
    
    # Market interpretation
    if avg_score > 60:
        st.success("🔥 Overall market segment appears undervalued")
    elif avg_score < 40:
        st.warning("⚠️ Overall market segment appears overvalued")
    else:
        st.info("📊 Overall market segment appears fairly valued")
    
    # Enhanced recommendation distribution
    st.subheader("Investment Recommendations")
    rec_df = df.groupby('Recommendation').size().reset_index(name='Count')
    fig_rec = px.pie(
        rec_df, 
        values='Count', 
        names='Recommendation',
        color='Recommendation',
        color_discrete_map={
            'Strong Buy': '#2E7D32',
            'Buy': '#4CAF50',
            'Hold': '#FFC107',
            'Sell': '#F44336',
            'Strong Sell': '#B71C1C'
        },
        hole=0.4
    )
    st.plotly_chart(fig_rec)
    
    # Key insights
    display_key_insights(df)

def display_technical_analysis(df):
    """Display enhanced technical analysis"""
    st.subheader("Technical Indicators")
    
    # Add technical analysis explanation
    with st.expander("Understanding Technical Indicators"):
        st.markdown("""
        - **RSI (Relative Strength Index)**: Momentum indicator showing overbought/oversold conditions
        - **Moving Averages**: Trend indicators showing short-term vs long-term price movements
        - **Volume**: Trading activity indicator
        """)
    
    # RSI Analysis with interpretation
    st.subheader("RSI Analysis")
    fig_rsi = px.scatter(
        df,
        x='Ticker',
        y='RSI',
        color='Recommendation',
        title='Relative Strength Index (RSI)',
        color_discrete_map={
            'Strong Buy': '#2E7D32',
            'Buy': '#4CAF50',
            'Hold': '#FFC107',
            'Sell': '#F44336',
            'Strong Sell': '#B71C1C'
        }
    )
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    st.plotly_chart(fig_rsi)
    
    # Moving Averages with crossover analysis
    st.subheader("Moving Average Analysis")
    ma_data = df[['Ticker', 'MA50', 'MA200']].melt(
        id_vars=['Ticker'],
        var_name='MA Type',
        value_name='Value'
    )
    fig_ma = px.line(
        ma_data,
        x='Ticker',
        y='Value',
        color='MA Type',
        title='Moving Averages Comparison'
    )
    st.plotly_chart(fig_ma)
    
    # Add MA crossover signals
    for _, row in df.iterrows():
        if row['MA50'] > row['MA200']:
            st.info(f"🔵 {row['Ticker']}: Golden Cross (Bullish Signal)")
        elif row['MA50'] < row['MA200']:
            st.warning(f"🔴 {row['Ticker']}: Death Cross (Bearish Signal)")

def display_stock_comparison(df):
    """Display stock comparison analysis"""
    st.subheader("📊 Stock Comparison")
    
    # Select stocks to compare
    stocks_to_compare = st.multiselect(
        "Select stocks to compare (max 3)",
        df['Ticker'].tolist(),
        max_selections=3
    )
    
    if stocks_to_compare:
        comparison_df = df[df['Ticker'].isin(stocks_to_compare)]
        
        # Radar chart comparison
        metrics = ['Valuation Score', 'Technical Score', 'Sector Score']
        fig = go.Figure()
        
        for ticker in stocks_to_compare:
            stock_data = comparison_df[comparison_df['Ticker'] == ticker]
            fig.add_trace(go.Scatterpolar(
                r=[stock_data[metric].iloc[0] for metric in metrics],
                theta=metrics,
                fill='toself',
                name=ticker
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Comparative Analysis"
        )
        st.plotly_chart(fig)
        
        # Detailed comparison table
        st.subheader("Detailed Comparison")
        comparison_metrics = ['Ticker', 'Current Price', 'Valuation Score', 
                            'Technical Score', 'Sector Score', 'Recommendation']
        st.dataframe(comparison_df[comparison_metrics])

def display_key_insights(df):
    """Display key insights from analysis"""
    st.subheader("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Best Value Opportunities")
        best_value = df.nlargest(3, 'Valuation Score')
        for _, stock in best_value.iterrows():
            st.write(f"**{stock['Ticker']}**: Score {stock['Valuation Score']:.1f}")
            st.write(f"Recommendation: {stock['Recommendation']}")
            st.write("---")
    
    with col2:
        st.markdown("#### Technical Standouts")
        best_tech = df.nlargest(3, 'Technical Score')
        for _, stock in best_tech.iterrows():
            st.write(f"**{stock['Ticker']}**: Score {stock['Technical Score']:.1f}")
            st.write(f"RSI: {stock.get('RSI', 'N/A')}")
            st.write("---")

def display_detailed_analysis(df):
    """Display detailed analysis with filtering"""
    st.subheader("Comprehensive Analysis")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        min_score = st.slider("Minimum Valuation Score", 0, 100, 0)
    with col2:
        selected_recommendations = st.multiselect(
            "Filter by Recommendation",
            df['Recommendation'].unique().tolist(),
            default=df['Recommendation'].unique().tolist()
        )
    
    # Filter DataFrame
    filtered_df = df[
        (df['Valuation Score'] >= min_score) &
        (df['Recommendation'].isin(selected_recommendations))
    ]
    
    # Display filtered results
    display_cols = ['Ticker', 'Company Name', 'Sector', 'Current Price', 
                   'Valuation Score', 'Technical Score', 'Sector Score',
                   '30D Volatility', 'Recommendation']
    
    styled_df = filtered_df[display_cols].style\
        .background_gradient(subset=['Valuation Score', 'Technical Score', 'Sector Score'], cmap='RdYlGn')\
        .background_gradient(subset=['30D Volatility'], cmap='YlOrRd')\
        .applymap(color_recommendation, subset=['Recommendation'])
    
    st.dataframe(styled_df)
    
    # Export functionality
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Analysis Results",
        data=csv,
        file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def map_sector_to_type(sector):
    """Map FMP sectors to SectorType"""
    sector_map = {
        'Technology': SectorType.TECHNOLOGY,
        'Healthcare': SectorType.HEALTHCARE,
        'Financials': SectorType.FINANCIAL,
        'Financial': SectorType.FINANCIAL,
        'Consumer Discretionary': SectorType.CONSUMER_CYCLICAL,
        'Consumer Staples': SectorType.CONSUMER_DEFENSIVE,
        'Industrials': SectorType.INDUSTRIALS,
        'Materials': SectorType.BASIC_MATERIALS,
        'Energy': SectorType.ENERGY,
        'Utilities': SectorType.UTILITIES,
        'Real Estate': SectorType.REAL_ESTATE,
        'Communication Services': SectorType.COMMUNICATION,
        'Telecommunications': SectorType.COMMUNICATION
    }
    return sector_map.get(sector, SectorType.TECHNOLOGY)

def color_recommendation(val):
    """Style function for recommendations"""
    colors = {
        'Strong Buy': 'background-color: #2E7D32; color: white',
        'Buy': 'background-color: #4CAF50; color: white',
        'Hold': 'background-color: #FFC107',
        'Sell': 'background-color: #F44336; color: white',
        'Strong Sell': 'background-color: #B71C1C; color: white'
    }
    return colors.get(val, '')

def display_stock_info(ticker, screener):
    """Show basic info when stock is added"""
    info = screener.api_client.get_company_profile(ticker)
    if info:
        sector = info.get('sector', 'Unknown')
        exchange = info.get('exchange', 'Unknown')
        price = info.get('price', 0)
        
        st.sidebar.markdown(f"""
        **Added: {info['companyName']}**
        - Exchange: {exchange}
        - Sector: {sector}
        - Current Price: ${price:.2f}
        """)

def display_sector_analysis(df, screener):
    """Display enhanced sector analysis"""
    st.subheader("Sector Analysis")
    
    # Get unique sectors
    sectors = df['Sector'].unique().tolist()
    
    # Sector selector
    selected_sector = st.selectbox("Select Sector for Analysis", sectors)
    
    # Filter for selected sector
    sector_df = df[df['Sector'] == selected_sector]
    
    if not sector_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sector metrics
            st.markdown(f"### {selected_sector} Metrics")
            avg_sector_score = sector_df['Sector Score'].mean()
            st.metric(
                "Average Sector Score",
                f"{avg_sector_score:.1f}",
                delta=f"{avg_sector_score - 50:.1f} vs Market",
                help="Sector-specific performance score"
            )
            
            # Get sector-specific metrics
            sector_type = map_sector_to_type(selected_sector)
            if sector_type in screener.sector_metrics.SECTOR_CONFIGS:
                specific_metrics = screener.sector_metrics.SECTOR_CONFIGS[sector_type]['specific_metrics']
                st.write("Key Sector Metrics:")
                
                # Display available metrics for the sector
                metrics_found = False
                for metric, config in specific_metrics.items():
                    # Convert metric name to the format used in your DataFrame
                    df_metric_name = metric.lower()
                    if df_metric_name in sector_df.columns:
                        metrics_found = True
                        avg_value = sector_df[df_metric_name].mean()
                        threshold = config['threshold']
                        if threshold is not None:
                            st.metric(
                                metric.replace('_', ' ').title(),
                                f"{avg_value:.2f}",
                                delta=f"{avg_value - threshold:.2f} vs Threshold"
                            )
                
                if not metrics_found:
                    st.info(f"No specific metrics available for {selected_sector}")
        
        with col2:
            # Sector risk analysis
            st.markdown("### Risk Analysis")
            if sector_type in screener.sector_metrics.SECTOR_CONFIGS:
                risk_factors = screener.sector_metrics.SECTOR_CONFIGS[sector_type]['risk_factors']
                st.write("Key Risk Factors:")
                
                # Display available risk factors
                risks_found = False
                for risk in risk_factors:
                    risk_col = f"{risk.lower()}_risk"
                    if risk_col in sector_df.columns:
                        risks_found = True
                        risk_value = sector_df[risk_col].mean()
                        st.progress(risk_value/100)
                        st.write(f"{risk.replace('_', ' ')}: {risk_value:.1f}%")
                
                if not risks_found:
                    st.info(f"No risk factors available for {selected_sector}")
        
        # Sector performance visualization
        st.subheader("Sector Performance Distribution")
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=sector_df['Valuation Score'],
            name='Valuation Score',
            boxpoints='all'
        ))
        fig.add_trace(go.Box(
            y=sector_df['Technical Score'],
            name='Technical Score',
            boxpoints='all'
        ))
        fig.add_trace(go.Box(
            y=sector_df['Sector Score'],
            name='Sector Score',
            boxpoints='all'
        ))
        fig.update_layout(
            title=f"{selected_sector} Score Distribution",
            yaxis_title="Score",
            showlegend=True
        )
        st.plotly_chart(fig)
    else:
        st.warning("No data available for selected sector")

if __name__ == "__main__":
    main()