#!/usr/bin/env python3
"""
Project structure setup script for Enhanced Stock Analysis Dashboard.
This script creates the necessary files and directories for the project.
"""

import os
import sys
import argparse

def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def create_file(path, content=""):
    """Create file with optional content"""
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write(content)
        print(f"Created file: {path}")
    else:
        print(f"File already exists: {path}")

def setup_project(base_dir, api_key=None):
    """Set up the project structure"""
    # Create base directory
    create_directory(base_dir)
    
    # Create .streamlit directory and secrets.toml
    streamlit_dir = os.path.join(base_dir, '.streamlit')
    create_directory(streamlit_dir)
    
    secrets_content = f'A_KEY = "{api_key or "your_alpha_vantage_api_key"}"'
    create_file(os.path.join(streamlit_dir, 'secrets.toml'), secrets_content)
    
    # Create empty Python files for the modules
    modules = [
        'app.py',
        'data_fetcher.py',
        'technical_analysis.py',
        'visualizations.py',
        'financial_analysis.py',
        'sentiment_analysis.py',
        'recommendation_engine.py',
        'portfolio_tracker.py'
    ]
    
    for module in modules:
        create_file(os.path.join(base_dir, module))
    
    # Create requirements.txt
    requirements = """streamlit>=1.20.0
pandas>=1.5.0
numpy>=1.23.0
yfinance>=0.2.12
plotly>=5.13.0
requests>=2.28.0
textblob>=0.17.1
matplotlib>=3.7.0
scikit-learn>=1.2.0
scipy>=1.10.0
tenacity>=8.2.0
uuid>=1.30
"""
    create_file(os.path.join(base_dir, 'requirements.txt'), requirements)
    
    # Create basic README.md
    readme = """# Enhanced Stock Analysis Dashboard

A comprehensive stock analysis dashboard built with Streamlit.

## Installation

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run app.py
```

## Features

- Stock price visualization
- Technical indicators
- Fundamental analysis
- Sentiment analysis
- Portfolio tracking
"""
    create_file(os.path.join(base_dir, 'README.md'), readme)
    
    print("\nProject structure created successfully!")
    print(f"Navigate to {base_dir} and start adding code to the modules.")
    print("\nTo run the dashboard after implementation:")
    print(f"cd {base_dir}")
    print("pip install -r requirements.txt")
    print("streamlit run app.py")

def main():
    parser = argparse.ArgumentParser(description='Set up Enhanced Stock Analysis Dashboard project structure')
    parser.add_argument('--dir', default='enhanced-stock-dashboard', help='Base directory for the project')
    parser.add_argument('--api-key', help='Alpha Vantage API key')
    
    args = parser.parse_args()
    
    setup_project(args.dir, args.api_key)

if __name__ == '__main__':
    main()
