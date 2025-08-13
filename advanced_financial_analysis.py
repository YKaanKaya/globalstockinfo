import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedFinancialAnalysis:
    """Advanced financial analysis and screening capabilities."""
    
    def __init__(self):
        pass
    
    def analyze_financial_statements(self, ticker: str) -> Dict:
        """
        Comprehensive financial statement analysis.
        
        Args:
            ticker (str): Stock ticker symbol
        
        Returns:
            Dict: Financial analysis results
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get financial statements
            income_stmt = stock.financials
            quarterly_income = stock.quarterly_financials
            balance_sheet = stock.balance_sheet
            quarterly_balance = stock.quarterly_balance_sheet
            cash_flow = stock.cashflow
            quarterly_cash = stock.quarterly_cashflow
            
            analysis = {
                'ticker': ticker,
                'income_statement_analysis': self._analyze_income_statement(income_stmt, quarterly_income),
                'balance_sheet_analysis': self._analyze_balance_sheet(balance_sheet, quarterly_balance),
                'cash_flow_analysis': self._analyze_cash_flow(cash_flow, quarterly_cash),
                'financial_ratios': self._calculate_financial_ratios(income_stmt, balance_sheet, cash_flow),
                'growth_analysis': self._analyze_growth_trends(income_stmt, quarterly_income),
                'financial_health_score': 0  # Will be calculated based on other metrics
            }
            
            # Calculate overall financial health score
            analysis['financial_health_score'] = self._calculate_financial_health_score(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing financial statements for {ticker}: {e}")
            return {'error': str(e)}
    
    def _analyze_income_statement(self, income_stmt: pd.DataFrame, quarterly: pd.DataFrame) -> Dict:
        """Analyze income statement trends and metrics."""
        analysis = {}
        
        if income_stmt.empty:
            return {'error': 'No income statement data available'}
        
        try:
            # Revenue analysis
            if 'Total Revenue' in income_stmt.index:
                revenues = income_stmt.loc['Total Revenue'].dropna()
                if len(revenues) >= 2:
                    revenue_growth = ((revenues.iloc[0] - revenues.iloc[1]) / revenues.iloc[1] * 100)
                    analysis['revenue_growth_annual'] = revenue_growth
                    analysis['revenue_trend'] = 'growing' if revenue_growth > 0 else 'declining'
                
                # Calculate revenue growth over multiple years
                if len(revenues) >= 3:
                    growth_rates = []
                    for i in range(len(revenues) - 1):
                        growth = ((revenues.iloc[i] - revenues.iloc[i+1]) / revenues.iloc[i+1] * 100)
                        growth_rates.append(growth)
                    analysis['avg_revenue_growth'] = np.mean(growth_rates)
                    analysis['revenue_growth_consistency'] = 1 - (np.std(growth_rates) / np.mean(np.abs(growth_rates))) if np.mean(np.abs(growth_rates)) != 0 else 0
            
            # Profitability analysis
            if 'Net Income' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                net_income = income_stmt.loc['Net Income'].dropna()
                revenues = income_stmt.loc['Total Revenue'].dropna()
                
                # Profit margins
                profit_margins = (net_income / revenues * 100).dropna()
                if not profit_margins.empty:
                    analysis['current_profit_margin'] = profit_margins.iloc[0]
                    if len(profit_margins) >= 2:
                        analysis['profit_margin_trend'] = profit_margins.iloc[0] - profit_margins.iloc[1]
            
            # Operating efficiency
            if 'Operating Income' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                operating_income = income_stmt.loc['Operating Income'].dropna()
                revenues = income_stmt.loc['Total Revenue'].dropna()
                operating_margins = (operating_income / revenues * 100).dropna()
                
                if not operating_margins.empty:
                    analysis['current_operating_margin'] = operating_margins.iloc[0]
            
            # Cost structure analysis
            if 'Cost Of Revenue' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                cost_of_revenue = income_stmt.loc['Cost Of Revenue'].dropna()
                revenues = income_stmt.loc['Total Revenue'].dropna()
                gross_margins = ((revenues - cost_of_revenue) / revenues * 100).dropna()
                
                if not gross_margins.empty:
                    analysis['current_gross_margin'] = gross_margins.iloc[0]
            
            # Quarterly trends (if available)
            if not quarterly.empty and 'Total Revenue' in quarterly.index:
                quarterly_revenues = quarterly.loc['Total Revenue'].dropna()
                if len(quarterly_revenues) >= 2:
                    quarterly_growth = ((quarterly_revenues.iloc[0] - quarterly_revenues.iloc[1]) / quarterly_revenues.iloc[1] * 100)
                    analysis['quarterly_revenue_growth'] = quarterly_growth
            
        except Exception as e:
            logger.error(f"Error in income statement analysis: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _analyze_balance_sheet(self, balance_sheet: pd.DataFrame, quarterly: pd.DataFrame) -> Dict:
        """Analyze balance sheet strength and trends."""
        analysis = {}
        
        if balance_sheet.empty:
            return {'error': 'No balance sheet data available'}
        
        try:
            # Liquidity analysis
            if 'Current Assets' in balance_sheet.index and 'Current Liabilities' in balance_sheet.index:
                current_assets = balance_sheet.loc['Current Assets'].dropna()
                current_liabilities = balance_sheet.loc['Current Liabilities'].dropna()
                
                if not current_assets.empty and not current_liabilities.empty:
                    current_ratio = (current_assets / current_liabilities).iloc[0]
                    analysis['current_ratio'] = current_ratio
                    analysis['liquidity_assessment'] = self._assess_liquidity(current_ratio)
            
            # Cash analysis
            if 'Cash And Cash Equivalents' in balance_sheet.index:
                cash_levels = balance_sheet.loc['Cash And Cash Equivalents'].dropna()
                if len(cash_levels) >= 2:
                    cash_change = cash_levels.iloc[0] - cash_levels.iloc[1]
                    analysis['cash_change'] = cash_change
                    analysis['cash_trend'] = 'increasing' if cash_change > 0 else 'decreasing'
            
            # Debt analysis
            debt_items = ['Total Debt', 'Long Term Debt', 'Short Long Term Debt']
            total_debt = 0
            
            for debt_item in debt_items:
                if debt_item in balance_sheet.index:
                    debt_value = balance_sheet.loc[debt_item].dropna()
                    if not debt_value.empty:
                        total_debt += debt_value.iloc[0]
            
            if 'Total Stockholder Equity' in balance_sheet.index:
                equity = balance_sheet.loc['Total Stockholder Equity'].dropna()
                if not equity.empty and equity.iloc[0] != 0:
                    debt_to_equity = total_debt / equity.iloc[0]
                    analysis['debt_to_equity'] = debt_to_equity
                    analysis['debt_assessment'] = self._assess_debt_level(debt_to_equity)
            
            # Asset efficiency
            if 'Total Assets' in balance_sheet.index:
                total_assets = balance_sheet.loc['Total Assets'].dropna()
                if len(total_assets) >= 2:
                    asset_growth = ((total_assets.iloc[0] - total_assets.iloc[1]) / total_assets.iloc[1] * 100)
                    analysis['asset_growth'] = asset_growth
            
            # Working capital analysis
            if 'Current Assets' in balance_sheet.index and 'Current Liabilities' in balance_sheet.index:
                current_assets = balance_sheet.loc['Current Assets'].dropna()
                current_liabilities = balance_sheet.loc['Current Liabilities'].dropna()
                
                if not current_assets.empty and not current_liabilities.empty:
                    working_capital = current_assets.iloc[0] - current_liabilities.iloc[0]
                    analysis['working_capital'] = working_capital
            
        except Exception as e:
            logger.error(f"Error in balance sheet analysis: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _analyze_cash_flow(self, cash_flow: pd.DataFrame, quarterly: pd.DataFrame) -> Dict:
        """Analyze cash flow patterns and sustainability."""
        analysis = {}
        
        if cash_flow.empty:
            return {'error': 'No cash flow data available'}
        
        try:
            # Operating cash flow analysis
            if 'Operating Cash Flow' in cash_flow.index:
                operating_cf = cash_flow.loc['Operating Cash Flow'].dropna()
                if not operating_cf.empty:
                    analysis['current_operating_cf'] = operating_cf.iloc[0]
                    
                    if len(operating_cf) >= 2:
                        cf_growth = ((operating_cf.iloc[0] - operating_cf.iloc[1]) / abs(operating_cf.iloc[1]) * 100)
                        analysis['operating_cf_growth'] = cf_growth
                        analysis['operating_cf_trend'] = 'improving' if cf_growth > 0 else 'declining'
            
            # Free cash flow calculation
            operating_cf = cash_flow.loc['Operating Cash Flow'].dropna() if 'Operating Cash Flow' in cash_flow.index else pd.Series()
            capex = cash_flow.loc['Capital Expenditures'].dropna() if 'Capital Expenditures' in cash_flow.index else pd.Series()
            
            if not operating_cf.empty and not capex.empty:
                # Capital expenditures are usually negative in yfinance
                free_cash_flow = operating_cf.iloc[0] + capex.iloc[0]  # Adding because capex is negative
                analysis['free_cash_flow'] = free_cash_flow
                
                if len(operating_cf) >= 2 and len(capex) >= 2:
                    prev_fcf = operating_cf.iloc[1] + capex.iloc[1]
                    fcf_growth = ((free_cash_flow - prev_fcf) / abs(prev_fcf) * 100) if prev_fcf != 0 else 0
                    analysis['free_cash_flow_growth'] = fcf_growth
            
            # Cash flow quality
            if 'Operating Cash Flow' in cash_flow.index and 'Net Income' in cash_flow.index:
                # Note: Net Income might not be in cash flow statement, might need to get from income statement
                pass
            
            # Investment analysis
            if 'Capital Expenditures' in cash_flow.index:
                capex_values = cash_flow.loc['Capital Expenditures'].dropna()
                if not capex_values.empty:
                    analysis['capital_expenditures'] = abs(capex_values.iloc[0])  # Make positive for display
                    
                    if len(capex_values) >= 2:
                        capex_change = ((abs(capex_values.iloc[0]) - abs(capex_values.iloc[1])) / abs(capex_values.iloc[1]) * 100)
                        analysis['capex_growth'] = capex_change
            
            # Financing activities
            if 'Financing Cash Flow' in cash_flow.index:
                financing_cf = cash_flow.loc['Financing Cash Flow'].dropna()
                if not financing_cf.empty:
                    analysis['financing_cash_flow'] = financing_cf.iloc[0]
            
        except Exception as e:
            logger.error(f"Error in cash flow analysis: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _calculate_financial_ratios(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame, cash_flow: pd.DataFrame) -> Dict:
        """Calculate comprehensive financial ratios."""
        ratios = {}
        
        try:
            # Get latest values
            latest_income = income_stmt.iloc[:, 0] if not income_stmt.empty else pd.Series()
            latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()
            
            # Profitability ratios
            if 'Net Income' in latest_income.index and 'Total Revenue' in latest_income.index:
                ratios['net_profit_margin'] = (latest_income['Net Income'] / latest_income['Total Revenue']) * 100
            
            if 'Net Income' in latest_income.index and 'Total Assets' in latest_balance.index:
                ratios['return_on_assets'] = (latest_income['Net Income'] / latest_balance['Total Assets']) * 100
            
            if 'Net Income' in latest_income.index and 'Total Stockholder Equity' in latest_balance.index:
                ratios['return_on_equity'] = (latest_income['Net Income'] / latest_balance['Total Stockholder Equity']) * 100
            
            # Efficiency ratios
            if 'Total Revenue' in latest_income.index and 'Total Assets' in latest_balance.index:
                ratios['asset_turnover'] = latest_income['Total Revenue'] / latest_balance['Total Assets']
            
            if 'Total Revenue' in latest_income.index and 'Total Stockholder Equity' in latest_balance.index:
                ratios['equity_multiplier'] = latest_balance['Total Assets'] / latest_balance['Total Stockholder Equity']
            
            # Liquidity ratios (already calculated in balance sheet analysis, but included here for completeness)
            if 'Current Assets' in latest_balance.index and 'Current Liabilities' in latest_balance.index:
                ratios['current_ratio'] = latest_balance['Current Assets'] / latest_balance['Current Liabilities']
            
            # Quick ratio (if inventory data available)
            if 'Current Assets' in latest_balance.index and 'Current Liabilities' in latest_balance.index:
                inventory = latest_balance.get('Inventory', 0)
                quick_assets = latest_balance['Current Assets'] - inventory
                ratios['quick_ratio'] = quick_assets / latest_balance['Current Liabilities']
            
            # Coverage ratios
            if 'Operating Income' in latest_income.index:
                # Interest coverage ratio (if interest expense available)
                interest_expense = latest_income.get('Interest Expense', 0)
                if interest_expense != 0:
                    ratios['interest_coverage'] = latest_income['Operating Income'] / abs(interest_expense)
            
        except Exception as e:
            logger.error(f"Error calculating financial ratios: {e}")
            ratios['error'] = str(e)
        
        return ratios
    
    def _analyze_growth_trends(self, income_stmt: pd.DataFrame, quarterly: pd.DataFrame) -> Dict:
        """Analyze growth trends over time."""
        growth_analysis = {}
        
        try:
            # Revenue growth trend
            if 'Total Revenue' in income_stmt.index:
                revenues = income_stmt.loc['Total Revenue'].dropna()
                if len(revenues) >= 3:
                    growth_rates = []
                    for i in range(len(revenues) - 1):
                        growth = ((revenues.iloc[i] - revenues.iloc[i+1]) / revenues.iloc[i+1] * 100)
                        growth_rates.append(growth)
                    
                    growth_analysis['revenue_growth_rates'] = growth_rates
                    growth_analysis['revenue_growth_trend'] = self._calculate_trend(growth_rates)
                    growth_analysis['revenue_cagr'] = self._calculate_cagr(revenues.iloc[-1], revenues.iloc[0], len(revenues) - 1)
            
            # Earnings growth trend
            if 'Net Income' in income_stmt.index:
                earnings = income_stmt.loc['Net Income'].dropna()
                if len(earnings) >= 3:
                    growth_rates = []
                    for i in range(len(earnings) - 1):
                        if earnings.iloc[i+1] != 0:
                            growth = ((earnings.iloc[i] - earnings.iloc[i+1]) / abs(earnings.iloc[i+1]) * 100)
                            growth_rates.append(growth)
                    
                    if growth_rates:
                        growth_analysis['earnings_growth_rates'] = growth_rates
                        growth_analysis['earnings_growth_trend'] = self._calculate_trend(growth_rates)
            
            # Quarterly growth analysis
            if not quarterly.empty and 'Total Revenue' in quarterly.index:
                quarterly_revenues = quarterly.loc['Total Revenue'].dropna()
                if len(quarterly_revenues) >= 4:
                    # Year-over-year quarterly growth
                    yoy_growth = []
                    for i in range(len(quarterly_revenues) - 4):
                        growth = ((quarterly_revenues.iloc[i] - quarterly_revenues.iloc[i+4]) / quarterly_revenues.iloc[i+4] * 100)
                        yoy_growth.append(growth)
                    
                    if yoy_growth:
                        growth_analysis['quarterly_yoy_growth'] = yoy_growth
                        growth_analysis['recent_quarterly_momentum'] = yoy_growth[0] if yoy_growth else 0
            
        except Exception as e:
            logger.error(f"Error in growth trend analysis: {e}")
            growth_analysis['error'] = str(e)
        
        return growth_analysis
    
    def _calculate_financial_health_score(self, analysis: Dict) -> float:
        """Calculate overall financial health score (0-100)."""
        score = 0
        max_score = 100
        
        try:
            # Profitability score (25 points)
            income_analysis = analysis.get('income_statement_analysis', {})
            profit_margin = income_analysis.get('current_profit_margin', 0)
            if profit_margin > 20: score += 25
            elif profit_margin > 10: score += 20
            elif profit_margin > 5: score += 15
            elif profit_margin > 0: score += 10
            
            # Liquidity score (20 points)
            balance_analysis = analysis.get('balance_sheet_analysis', {})
            current_ratio = balance_analysis.get('current_ratio', 0)
            if current_ratio > 2: score += 20
            elif current_ratio > 1.5: score += 15
            elif current_ratio > 1: score += 10
            elif current_ratio > 0.5: score += 5
            
            # Debt score (20 points)
            debt_to_equity = balance_analysis.get('debt_to_equity', float('inf'))
            if debt_to_equity < 0.3: score += 20
            elif debt_to_equity < 0.5: score += 15
            elif debt_to_equity < 1: score += 10
            elif debt_to_equity < 2: score += 5
            
            # Growth score (20 points)
            growth_analysis = analysis.get('growth_analysis', {})
            revenue_growth = income_analysis.get('revenue_growth_annual', 0)
            if revenue_growth > 20: score += 20
            elif revenue_growth > 10: score += 15
            elif revenue_growth > 5: score += 10
            elif revenue_growth > 0: score += 5
            
            # Cash flow score (15 points)
            cf_analysis = analysis.get('cash_flow_analysis', {})
            operating_cf = cf_analysis.get('current_operating_cf', 0)
            free_cf = cf_analysis.get('free_cash_flow', 0)
            if operating_cf > 0 and free_cf > 0: score += 15
            elif operating_cf > 0: score += 10
            elif free_cf > 0: score += 5
            
        except Exception as e:
            logger.error(f"Error calculating financial health score: {e}")
        
        return min(score, max_score)
    
    def _assess_liquidity(self, current_ratio: float) -> str:
        """Assess liquidity based on current ratio."""
        if current_ratio > 3:
            return "Excellent - May have excess cash"
        elif current_ratio > 2:
            return "Very Good"
        elif current_ratio > 1.5:
            return "Good"
        elif current_ratio > 1:
            return "Adequate"
        elif current_ratio > 0.5:
            return "Poor - Liquidity concerns"
        else:
            return "Critical - Severe liquidity issues"
    
    def _assess_debt_level(self, debt_to_equity: float) -> str:
        """Assess debt level based on debt-to-equity ratio."""
        if debt_to_equity < 0.3:
            return "Conservative - Low debt"
        elif debt_to_equity < 0.5:
            return "Moderate - Reasonable debt level"
        elif debt_to_equity < 1:
            return "Moderate to High"
        elif debt_to_equity < 2:
            return "High - Monitor debt levels"
        else:
            return "Very High - Potential concern"
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values."""
        if not values or len(values) < 2:
            return "Insufficient data"
        
        # Use linear regression to determine trend
        x = np.arange(len(values))
        slope, _, r_value, _, _ = stats.linregress(x, values)
        
        if abs(r_value) < 0.3:
            return "No clear trend"
        elif slope > 0:
            return "Improving"
        else:
            return "Declining"
    
    def _calculate_cagr(self, start_value: float, end_value: float, periods: int) -> float:
        """Calculate Compound Annual Growth Rate."""
        if start_value <= 0 or end_value <= 0 or periods <= 0:
            return 0
        
        return ((end_value / start_value) ** (1 / periods) - 1) * 100
    
    def screen_stocks(self, tickers: List[str], criteria: Dict) -> pd.DataFrame:
        """
        Screen stocks based on financial criteria.
        
        Args:
            tickers (List[str]): List of stock tickers to screen
            criteria (Dict): Screening criteria
        
        Returns:
            pd.DataFrame: Screening results
        """
        results = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Basic info
                result = {
                    'ticker': ticker,
                    'company_name': info.get('longName', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'forward_pe': info.get('forwardPE', 0),
                    'pb_ratio': info.get('priceToBook', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    'beta': info.get('beta', 0),
                    'profit_margin': info.get('profitMargins', 0),
                    'operating_margin': info.get('operatingMargins', 0),
                    'roe': info.get('returnOnEquity', 0),
                    'roa': info.get('returnOnAssets', 0),
                    'debt_to_equity': info.get('debtToEquity', 0),
                    'current_ratio': info.get('currentRatio', 0),
                    'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                    'peg_ratio': info.get('pegRatio', 0),
                    'passes_screen': True  # Will be updated based on criteria
                }
                
                # Apply screening criteria
                for criterion, threshold in criteria.items():
                    if criterion in result:
                        value = result[criterion]
                        if isinstance(threshold, dict):
                            # Range criteria
                            min_val = threshold.get('min', float('-inf'))
                            max_val = threshold.get('max', float('inf'))
                            if not (min_val <= value <= max_val):
                                result['passes_screen'] = False
                                break
                        else:
                            # Simple threshold
                            if value < threshold:
                                result['passes_screen'] = False
                                break
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error screening {ticker}: {e}")
                results.append({
                    'ticker': ticker,
                    'error': str(e),
                    'passes_screen': False
                })
        
        return pd.DataFrame(results)
    
    def compare_companies(self, tickers: List[str]) -> Dict:
        """
        Compare financial metrics across multiple companies.
        
        Args:
            tickers (List[str]): List of company tickers to compare
        
        Returns:
            Dict: Comparison results
        """
        comparison = {'tickers': tickers, 'metrics': {}}
        
        for ticker in tickers:
            try:
                analysis = self.analyze_financial_statements(ticker)
                
                if 'error' not in analysis:
                    comparison['metrics'][ticker] = {
                        'financial_health_score': analysis.get('financial_health_score', 0),
                        'profit_margin': analysis.get('income_statement_analysis', {}).get('current_profit_margin', 0),
                        'revenue_growth': analysis.get('income_statement_analysis', {}).get('revenue_growth_annual', 0),
                        'current_ratio': analysis.get('balance_sheet_analysis', {}).get('current_ratio', 0),
                        'debt_to_equity': analysis.get('balance_sheet_analysis', {}).get('debt_to_equity', 0),
                        'free_cash_flow': analysis.get('cash_flow_analysis', {}).get('free_cash_flow', 0)
                    }
                
            except Exception as e:
                logger.error(f"Error comparing {ticker}: {e}")
                comparison['metrics'][ticker] = {'error': str(e)}
        
        return comparison

# Global instance
financial_analyzer = AdvancedFinancialAnalysis()