import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
# Advanced Financial Machine Learning Backtester
class AdvancedMLBacktester:
    """
    An advanced machine learning backtesting engine executing information-sampled 
    dollar bars, dynamic triple-barrier path labeling, and meta-classification filters.
    """
    
    def __init__(self, ticker: str, dollar_threshold: float = 5000000.0, transaction_cost: float = 0.0001):
        self.ticker = ticker
        self.dollar_threshold = dollar_threshold
        self.transaction_cost = transaction_cost
        
        self.raw_data = pd.DataFrame()
        self.dollar_bars = pd.DataFrame()

        self.labeled_events = pd.DataFrame()
        self.backtest_df = pd.DataFrame()
        self.entry_events = pd.Index([])
        self.features_df = pd.DataFrame()
        self.metrics = {}

    def fetch_intraday_market_data(self):
        print(f"[Data Pipeline] Querying live 1-minute historical nodes for {self.ticker}...")
        try:
            ticker_obj = yf.Ticker(self.ticker)
            df = ticker_obj.history(period="7d", interval="1m")
            
            if df.empty or len(df) < 500:
                raise ValueError("Insufficient intraday tick sequence returned from live stream API.")
                
            self.raw_data = pd.DataFrame(index=df.index)
            self.raw_data['Price'] = df['Close'].astype(float)
            self.raw_data['Volume'] = df['Volume'].astype(float)
            print(f"[Data Pipeline] Ingestion successful. Loaded {len(self.raw_data)} raw intraday intervals.")

            
        except Exception as e:
            print(f"\n[API Notice] Live intraday download bypassed or unavailable: {e}")
            print("Switching seamlessly to internal high-frequency market path simulation...\n")
            
            np.random.seed(42)
            sim_records = 10000
            shocks = np.random.normal(0.0, 0.2, sim_records)
            prices = 5000.0 + np.cumsum(shocks)
            volumes = np.random.randint(100, 2500, size=sim_records)
            dates = pd.date_range(end=pd.Timestamp.today(), periods=sim_records, freq='min')
            
            self.raw_data = pd.DataFrame({'Price': prices, 'Volume': volumes}, index=dates)
            print(f"[Simulation] Synthesized {len(self.raw_data)} high-density tick observations.")

    def construct_dollar_bars(self):
        print(f"[Data Engineering] Chunking series matrix into Transaction Dollar Bars (M = ${self.dollar_threshold:,.2f})...")
        
        timestamps = self.raw_data.index.values

        prices = self.raw_data['Price'].values
        volumes = self.raw_data['Volume'].values
        
        b_time, b_open, b_high, b_low, b_close, b_vol = [], [], [], [], [], []
        
        cum_dollar_value = 0.0
        cur_open = prices[0]
        cur_high = prices[0]
        cur_low = prices[0]
        cur_vol = 0
        
        for i in range(len(self.raw_data)):
            p = prices[i]
            v = volumes[i]
            turnover = p * v
            
            cum_dollar_value += turnover
            cur_vol += v
            

            if p > cur_high: cur_high = p
            if p < cur_low:  cur_low = p
            
            if cum_dollar_value >= self.dollar_threshold:
                b_time.append(timestamps[i])
                b_open.append(cur_open)
                b_high.append(cur_high)
                b_low.append(cur_low)
                b_close.append(p)
                b_vol.append(cur_vol)
                
                cum_dollar_value = 0.0
                if i < len(self.raw_data) - 1:
                    cur_open = prices[i+1]
                    cur_high = prices[i+1]
                    cur_low = prices[i+1]
                    cur_vol = 0
                    
        self.dollar_bars = pd.DataFrame({

            'Open': b_open, 'High': b_high, 'Low': b_low, 'Close': b_close, 'Volume': b_vol
        }, index=pd.to_datetime(b_time))
        
        self.dollar_bars['Returns'] = np.log(self.dollar_bars['Close'] / self.dollar_bars['Close'].shift(1))
        self.dollar_bars.dropna(inplace=True)
        print(f"[Data Engineering] Sampling complete. Generated {len(self.dollar_bars)} standardized information nodes.")

    def engineer_signals_and_features(self, fast_window: int = 10, slow_window: int = 30):
        print("[Features Grid] Instantiating signal logic vectors...")
        df = self.dollar_bars.copy()
        
        df['Fast_MA'] = df['Close'].rolling(window=fast_window).mean()
        df['Slow_MA'] = df['Close'].rolling(window=slow_window).mean()
        df.dropna(inplace=True)
        
        df['Side'] = 0
        df.loc[df['Fast_MA'] > df['Slow_MA'], 'Side'] = 1
        df.loc[df['Fast_MA'] < df['Slow_MA'], 'Side'] = -1
        
        df['Signal_Change'] = df['Side'].diff()
        self.entry_events = df[df['Signal_Change'] != 0].index
        
        df['Volatility'] = df['Returns'].ewm(span=50).std()
        df['Momentum'] = df['Fast_MA'] - df['Slow_MA']
        df['Serial_Corr'] = df['Returns'].rolling(window=5).apply(lambda x: x.autocorr(lag=1), raw=False)
        df['MA_Diff'] = (df['Close'] - df['Slow_MA']) / df['Slow_MA']
        
        self.features_df = df.dropna().copy()
        print(f"[Features Grid] Feature arrays built. Identified {len(self.entry_events)} baseline transaction opportunities.")

    def apply_triple_barrier_labeling(self, pt_multiplier: float = 1.0, sl_multiplier: float = 2.0, vertical_horizon: int = 10):
        print("[Labeling Suite] Computing dynamic paths via Triple Barrier Method bounds...")
        df = self.features_df.copy()
        active_events = self.entry_events.intersection(df.index)
        
        active_events = [e for e in active_events if df.index.get_loc(e) + vertical_horizon < len(df)]
        
        meta_labels = []

        validated_times = []
        
        for event_time in active_events:
            loc = df.index.get_loc(event_time)
            entry_price = df.loc[event_time, 'Close']
            volatility = df.loc[event_time, 'Volatility']
            side = df.loc[event_time, 'Side']
            
            if volatility < 0.0001 or side == 0:
                continue
                
            upper_barrier = entry_price * (1.0 + side * pt_multiplier * volatility)
            lower_barrier = entry_price * (1.0 - side * sl_multiplier * volatility)
            
            forward_look = df.iloc[loc + 1 : loc + 1 + vertical_horizon]
            label_state = 0
            
            for _, row in forward_look.iterrows():
                current_price = row['Close']

                if side == 1:
                    if current_price >= upper_barrier:
                        label_state = 1
                        break
                    elif current_price <= lower_barrier:
                        label_state = 0
                        break
                elif side == -1:
                    if current_price <= upper_barrier:
                        label_state = 1
                        break
                    elif current_price >= lower_barrier:
                        label_state = 0
                        break
                        
            meta_labels.append(label_state)
            validated_times.append(event_time)
            
        self.labeled_events = pd.DataFrame({

            'Side': df.loc[validated_times, 'Side'],
            'Volatility': df.loc[validated_times, 'Volatility'],
            'Momentum': df.loc[validated_times, 'Momentum'],
            'Serial_Corr': df.loc[validated_times, 'Serial_Corr'],
            'MA_Diff': df.loc[validated_times, 'MA_Diff'],
            'Entry_Price': df.loc[validated_times, 'Close'],
            'Meta_Label': meta_labels
        }, index=validated_times)
        
        print(f"[Labeling Suite] Complete. Total labeled records: {len(self.labeled_events)} | Success Hits: {self.labeled_events['Meta_Label'].sum()}")

    def train_meta_classifier(self, train_ratio: float = 0.70, confidence_threshold: float = 0.75):
        df = self.labeled_events.copy()
        if len(df) < 15:
            train_ratio = 0.50
            
        feature_cols = ['Volatility', 'Momentum', 'Serial_Corr', 'MA_Diff']
        X = df[feature_cols]
        y = df['Meta_Label']

        
        split_point = int(len(df) * train_ratio)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        print(f"[ML Model] Tuning Random Forest Ensemble structure across out-of-sample vectors...")
        clf = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train)
        
        test_probabilities = clf.predict_proba(X_test)[:, 1]
        test_df = df.iloc[split_point:].copy()
        test_df['Prob_Success'] = test_probabilities
        test_df['Meta_Prediction'] = np.where(test_df['Prob_Success'] > confidence_threshold, 1, 0)
        
        self.backtest_df = test_df
        self._log_classification_metrics(y_test, test_df['Meta_Prediction'])

    def _log_classification_metrics(self, ground_truth, predictions):
        print("\n" + "="*70)

        print("          OUT-OF-SAMPLE DATA SCIENCE SCORECARD (META-LABELER)    ")
        print("="*70)
        print(classification_report(ground_truth, predictions, zero_division=0))
        print("-"*70)
        print("Confusion Matrix Layout Array:")
        print(confusion_matrix(ground_truth, predictions))
        print("="*70 + "\n")

    def run_comparative_backtest(self, holding_horizon: int = 10):
        print("[Backtester Engine] Assessing portfolio trajectory allocations...")
        df_bars = self.dollar_bars.copy()
        test_events = self.backtest_df.copy()
        
        base_returns = []
        filtered_returns = []
        
        for event_time, row in test_events.iterrows():
            loc = df_bars.index.get_loc(event_time)
            side = row['Side']

            entry_price = row['Entry_Price']
            meta_decision = row['Meta_Prediction']
            
            exit_price = df_bars.iloc[loc + holding_horizon]['Close']
            trade_return = side * (np.log(exit_price / entry_price)) - self.transaction_cost
            base_returns.append(trade_return)
            
            if meta_decision == 1:
                filtered_returns.append(trade_return)
                
        br = np.array(base_returns)
        fr = np.array(filtered_returns) if len(filtered_returns) > 0 else np.array([0.0])
        
        sharpe_base = (br.mean() / br.std() * np.sqrt(252)) if len(br) > 0 and br.std() != 0 else 0
        sharpe_filt = (fr.mean() / fr.std() * np.sqrt(252)) if len(fr) > 0 and fr.std() != 0 else 0
        
        hit_base = (br > 0).sum() / len(br) if len(br) > 0 else 0
        hit_filt = (fr > 0).sum() / len(fr) if len(fr) > 0 else 0
        
        self.metrics = {
            "Total Unfiltered Primary Signals": len(br),
            "Meta-Filtered Executed Trades"  : len(fr),
            "Baseline Unfiltered Hit Ratio"  : f"{hit_base * 100:.2f}%",
            "Meta-Filtered Signal Hit Ratio" : f"{hit_filt * 100:.2f}%",
            "Baseline Average Return / Trade" : f"{br.mean() * 100:.4f}%",
            "Meta-Filtered Avg Return / Trade": f"{fr.mean() * 100:.4f}%",
            "Baseline Portfolio Sharpe Ratio" : f"{sharpe_base:.3f}",
            "Meta-Filtered Portfolio Sharpe"  : f"{sharpe_filt:.3f}"
        }

    def print_performance_dashboard(self):
        print("="*80)
        print("    QUANTITATIVE SYSTEM UPGRADE REPORT: REPO CORE BACKTEST ENGINE STATUS   ")
        print("="*80)
        print(f"Target Underlyer Asset Identifier: {self.ticker:<20} | Strategy Class : FinML Meta-Classifier")
        print(f"Data Aggregation Architecture    : Dollar Sampling Bars | Threshold Limit: ${self.dollar_threshold:,.2f}")
        print(f"Execution Cost Slippage Haircut  : {self.transaction_cost * 10000:.1f} bps per trade side execution")
        print("-" * 80)

        print(f"{'EVALUATION APPRAISAL METRIC PROFILE TYPE':<42} | {'METRIC OUTCOME MATRIX VALUE':<30}")
        print("-" * 80)
        for metric, val in self.metrics.items():
            print(f" • {metric:<40} | {val:<30}")
        print("="*80)
if __name__ == "__main__":
    quant_engine = AdvancedMLBacktester(ticker="SPY", dollar_threshold=5000000.0, transaction_cost=0.0001)
    quant_engine.fetch_intraday_market_data()
    quant_engine.construct_dollar_bars()
    quant_engine.engineer_signals_and_features(fast_window=10, slow_window=30)
    quant_engine.apply_triple_barrier_labeling(pt_multiplier=1.0, sl_multiplier=2.0, vertical_horizon=10)
    quant_engine.train_meta_classifier(train_ratio=0.70, confidence_threshold=0.75)
    quant_engine.run_comparative_backtest(holding_horizon=10)
    quant_engine.print_performance_dashboard()


