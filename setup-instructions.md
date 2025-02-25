# Installation and Setup Guide

This guide provides step-by-step instructions to set up and run the Enhanced Stock Analysis Dashboard.

## Prerequisites

- Python 3.8 or higher
- Pip package manager
- Git (optional, for cloning the repository)
- Alpha Vantage API key (free tier available)

## Installation Steps

### 1. Clone or Download the Repository

**Option 1: Using Git**
```bash
git clone https://github.com/yourusername/enhanced-stock-dashboard.git
cd enhanced-stock-dashboard
```

**Option 2: Manual Download**
- Download the ZIP file of the repository
- Extract to your preferred location
- Open a terminal/command prompt and navigate to the extracted folder

### 2. Create a Virtual Environment (Optional but Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Set up API Keys

1. Create a `.streamlit` directory in the project root if it doesn't exist:
```bash
mkdir -p .streamlit
```

2. Create a `secrets.toml` file inside the `.streamlit` directory:
```bash
touch .streamlit/secrets.toml
```

3. Add your Alpha Vantage API key to the `secrets.toml` file:
```toml
A_KEY = "your_alpha_vantage_api_key"
```

### 5. Run the Application

```bash
streamlit run app.py
```

The dashboard should open automatically in your default web browser at `http://localhost:8501`.

## Folder Structure Setup

Ensure your project has the following structure:
```
enhanced-stock-dashboard/
├── .streamlit/
│   └── secrets.toml
├── app.py
├── data_fetcher.py
├── technical_analysis.py
├── visualizations.py
├── financial_analysis.py
├── sentiment_analysis.py
├── recommendation_engine.py
├── portfolio_tracker.py
├── requirements.txt
└── README.md
```

## Getting an Alpha Vantage API Key

1. Go to [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Fill out the form to get a free API key
3. Copy the key into your `secrets.toml` file as shown above

## Troubleshooting Common Issues

### Issue: ModuleNotFoundError
If you see an error like `ModuleNotFoundError: No module named 'xyz'`:
```bash
pip install xyz
```

### Issue: API Key Not Found
If you get an error about missing API keys:
- Verify your `secrets.toml` file is in the correct location
- Check the API key is correctly formatted with no extra spaces
- Restart the Streamlit server

### Issue: Slow Performance
- Consider upgrading to a paid API tier for higher rate limits
- Adjust the caching TTL in `data_fetcher.py` if needed

### Issue: Data Not Loading for Some Stocks
- Some stocks may have limited data availability
- Check if the ticker symbol is correct
- Try a different stock to verify the API connection

## Deployment Options

### Deploy to Streamlit Cloud

1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Select your repository and deploy
5. Add your API keys in the Streamlit Cloud secrets management section

### Deploy using Docker

1. Create a Dockerfile:
```Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py"]
```

2. Build and run the Docker container:
```bash
docker build -t stock-dashboard .
docker run -p 8501:8501 stock-dashboard
```

## Updating the Dashboard

To update the dashboard to the latest version:

1. Pull the latest changes (if using Git):
```bash
git pull origin main
```

2. Update dependencies:
```bash
pip install -r requirements.txt --upgrade
```

3. Restart the application:
```bash
streamlit run app.py
```
