import pandas as pd

class VectorizedBacktester:
    """
    An institutional-grade vectorized backtesting engine designed to demonstrate 
    the intersection of quantitative finance theory and data science engineering.
    
    This engine simulates algorithmic trading strategies using matrix operations 
    via NumPy and Pandas, incorporating realistic friction parameters like transaction 
    costs and slippage, and computing industry-standard risk/reward metrics.
    """
    
    def __init__(self, ticker: str, initial_capital: float = 100000.0, transaction_cost: float = 0.0005):
        """
        Initializes the backtesting framework.
        
        Parameters:
        -----------
        ticker : str
            The identifier of the financial asset being backtested (e.g., 'AAPL', 'BTC/USD').
        initial_capital : float
            The portfolio starting value in base currency (default: $100,000).
        transaction_cost : float
            The combined fee and slippage rate per trade side (e.g., 0.0005 = 0.05% or 5 bps).
        """
        self.ticker = ticker
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.data = pd.DataFrame()
        self.metrics = {}

    def generate_synthetic_market_data(self, days: int = 1260, spot_price: float = 100.0, mu: float = 0.05, sigma: float = 0.25, seed: int = 42):
        """
        Generates synthetic market price pathways using Geometric Brownian Motion (GBM).
        This proves mathematical competency in stochastic modeling to quantitative recruiters.
        
        Parameters:
        -----------
        days : int
            Number of trading days to simulate (~5 years of daily data).
        spot_price : float
            The initial asset price.
        mu : float
            The annualized expected return (drift coefficient).
        sigma : float
            The annualized volatility coefficient.
        seed : int
            Random seed for reproducibility of data generation.
        """
        np.random.seed(seed)
        dt = 1 / 252  # Time step size mapping to single daily increments
        
        # Stochastic calculus formulation: dS = mu * S * dt + sigma * S * dWt
        random_shocks = np.random.normal(0, np.sqrt(dt), days - 1)
        price_path = np.zeros(days)
        price_path[0] = spot_price
        
        for t in range(1, days):
            price_path[t] = price_path[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * random_shocks[t-1])
            
        date_range = pd.date_range(end=pd.Timestamp.today(), periods=days, freq='B')
        
        df = pd.DataFrame(index=date_range)
        df['Close'] = price_path
        # Compute benchmarking ground truth: Base asset log returns
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Modern Pandas practice: Assigning to state using functional copies to avoid warnings
        self.data = df.dropna().copy()

    def implement_bollinger_bands_strategy(self, window: int = 20, num_std: float = 2.0):
        """
        Implements a classic mean-reversion Bollinger Bands quantitative strategy.
        Generates buy/sell matrix signals dynamically using data vectorization.
        
        Parameters:
        -----------
        window : int
            The rolling lookback window size for calculating moving averages.
        num_std : float
            The multiplier for standard deviation to establish outer bands boundaries.
        """
        if self.data.empty:
            raise ValueError("No market data found. Execute data generation or ingestion before strategy math.")
            
        df = self.data.copy()
        
        # Data Science Skill: Rolling window aggregations
        df['MA'] = df['Close'].rolling(window=window).mean()
        df['Std'] = df['Close'].rolling(window=window).std()
        df['Upper_Band'] = df['MA'] + (num_std * df['Std'])
        df['Lower_Band'] = df['MA'] - (num_std * df['Std'])
        
        # Vectorized Generation of Signal logic
        # 1 = Long market stance, -1 = Short market stance, 0 = Cash position
        df['Signal'] = 0
        df.loc[df['Close'] < df['Lower_Band'], 'Signal'] = 1  # Oversold condition -> Long
        df.loc[df['Close'] > df['Upper_Band'], 'Signal'] = -1 # Overbought condition -> Short
        
        # MODERN PANDAS COMPATIBILITY FIX:
        # The 'method' parameter inside NDFrame.replace() is deprecated/removed in recent releases.
        # To carry forward the signals cleanly without lookahead bias, we replace 0 with NaN, 
        # apply the standalone .ffill() method, and shift execution by 1 period to prevent lookahead bias.
        df['Position'] = df['Signal'].replace(0, np.nan).ffill().shift(1)
        df['Position'] = df['Position'].fillna(0)
        
        self.data = df.copy()
        self._calculate_strategy_returns()

    def _calculate_strategy_returns(self):
        """
        Computes portfolio equity path, adjusting for execution lag and transaction costs.
        Internal method mapped automatically by strategy routine executions.
        """
        df = self.data.copy()
        
        # Strategy Return = Position state * Underlying log asset returns
        df['Strategy_Returns_Raw'] = df['Position'] * df['Returns']
        
        # Quant Core Detail: Detect position alterations to assess transaction friction
        df['Trades'] = df['Position'].diff().abs()
        df['Trades'] = df['Trades'].fillna(0)
        
        # Apply performance haircut based on execution friction parameter
        df['Transaction_Cost_Impact'] = df['Trades'] * self.transaction_cost
        df['Strategy_Returns_Net'] = df['Strategy_Returns_Raw'] - df['Transaction_Cost_Impact']
        
        # Convert vector logs back to standard compounding performance paths
        df['Cum_Market_Returns'] = np.exp(df['Returns'].cumsum())
        df['Cum_Strategy_Returns'] = np.exp(df['Strategy_Returns_Net'].cumsum())
        
        # Portfolio value progression over the testing horizon
        df['Portfolio_Value'] = self.initial_capital * df['Cum_Strategy_Returns']
        
        self.data = df.copy()

    def compute_performance_metrics(self):
        """
        Calculates advanced, mathematical performance indicators expected by Quant Funds.
        Evaluates returns data to yield risk management dimensions.
        """
        df = self.data.copy()
        net_returns = df['Strategy_Returns_Net']
        cum_returns = df['Cum_Strategy_Returns']
        
        # 1. Compound Annual Growth Rate (CAGR) calculation
        total_days = len(df)
        years = total_days / 252
        cagr = (cum_returns.iloc[-1]) ** (1 / years) - 1 if total_days > 0 and cum_returns.iloc[-1] > 0 else 0
        
        # 2. Annualized Volatility
        annualized_vol = net_returns.std() * np.sqrt(252)
        
        # 3. Sharpe Ratio (Assuming 0% risk-free benchmark floor for math architecture)
        sharpe_ratio = (net_returns.mean() / net_returns.std()) * np.sqrt(252) if net_returns.std() != 0 else 0
        
        # 4. REFINED INSTITUTIONAL SORTINO RATIO:
        # Standard quant conventions require evaluating downside risk across the entire timeframe, 
        # treating positive days as 0 deviation rather than ignoring them completely.
        downside_returns = net_returns.copy()
        downside_returns[downside_returns > 0] = 0
        downside_vol = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
        sortino_ratio = (net_returns.mean() * 252) / downside_vol if downside_vol != 0 else 0
        
        # 5. Maximum Drawdown (Highest Peak-to-Trough drop path calculation)
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # 6. Calmar Ratio
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        self.metrics = {
            "Initial Capital": f"${self.initial_capital:,.2f}",
            "Final Portfolio Value": f"${df['Portfolio_Value'].iloc[-1]:,.2f}",
            "CAGR": f"{cagr * 100:.2f}%",
            "Annualized Volatility": f"{annualized_vol * 100:.2f}%",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Sortino Ratio": f"{sortino_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown * 100:.2f}%",
            "Calmar Ratio": f"{calmar_ratio:.2f}",
            "Total Trades Executed": int(df['Trades'].sum())
        }
        
        return self.metrics

    def display_report(self):
        """Prints performance dashboard output cleanly to terminal console."""
        print("="*60)
        print(f" COMPREHENSIVE QUANT PERFORMANCE REPORT: {self.ticker} ")
        print("="*60)
        for metric, value in self.metrics.items():
            print(f"{metric:<30}: {value}")
        print("="*60)


# Interactive Execution Routine to validate script readiness
if __name__ == "__main__":
    # Create target backtest instance with 5 basis point transaction costs
    bt = VectorizedBacktester(ticker="AAPL", initial_capital=100000.0, transaction_cost=0.0005)
    
    # Generate 5 Years of Market Data using Geometric Brownian Motion
    print("Executing stochastic path simulation via Geometric Brownian Motion...")
    bt.generate_synthetic_market_data(days=1260, spot_price=150.0, mu=0.07, sigma=0.22)
    
    # Apply strategy framework
    print("Applying vectorized Bollinger Bands strategy rules...")
    bt.implement_bollinger_bands_strategy(window=20, num_std=2.0)
    
    # Extract structural risk evaluation arrays
    print("Extracting performance matrix parameters...")
    bt.compute_performance_metrics()
    
    # Display results pipeline
    bt.display_report()
