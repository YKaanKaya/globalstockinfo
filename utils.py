import streamlit as st

# Alpha Vantage API key from Streamlit secrets
api_key = st.secrets["A_KEY"]

def format_large_number(value):
    """Format large numbers into readable strings with suffixes."""
    if not isinstance(value, (int, float)):
        return 'N/A'
    abs_value = abs(value)
    if abs_value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif abs_value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif abs_value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.0f}"
