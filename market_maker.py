import numpy as np
import pandas as pd
import time

class AvellanedaStoikovMarketMaker:
    """
    An institutional-grade algorithmic market-making engine implementing the
    classical Avellaneda-Stoikov framework for optimal inventory risk management.
    
    This engine computes optimal reservation prices and asymmetric bid-ask quotes
    dynamically as a function of inventory parameters, underlying asset volatility,
    and instantaneous order flow parameters to maximize portfolio utility.
    """
    
    def __init__(self, initial_mid: float = 100.0, asset_vol: float = 0.015, 
                 gamma: float = 0.1, kappa: float = 1.5, inventory_limit: int = 20):
        """
        Initializes the market-making parameter space.
        """
        self.mid_price = initial_mid
        self.volatility = asset_vol
        self.gamma = gamma
        self.kappa = kappa
        self.inventory_limit = inventory_limit
        
        # State tracking parameters
        self.inventory = 0
        self.cash = 100000.0
        self.initial_wealth = self.cash
        self.trades_executed = 0
        
        # Historical logging dataframes
        self.history = []

    def calculate_reservation_price(self, T_minus_t: float) -> float:
        """
        Computes the structural reservation price (r) adjusted for inventory risk.
        r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
        """
        reservation = self.mid_price - (self.inventory * self.gamma * (self.volatility ** 2) * T_minus_t)
        return reservation

    def calculate_optimal_spread(self, T_minus_t: float) -> float:
        """
        Computes the total optimal asymptotic bid-ask spread depth.
        spread = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / kappa)
        """
        inventory_risk_premium = self.gamma * (self.volatility ** 2) * T_minus_t
        liquidity_density_premium = (2 / self.gamma) * np.log(1 + (self.gamma / self.kappa))
        return inventory_risk_premium + liquidity_density_premium

    def generate_asymmetric_quotes(self, T_minus_t: float) -> dict:
        """
        Generates optimal localized bid and ask prices based on reservation price mappings.
        """
        r = self.calculate_reservation_price(T_minus_t)
        total_spread = self.calculate_optimal_spread(T_minus_t)
        
        half_spread = total_spread / 2.0
        
        bid_quote = r - half_spread
        ask_quote = r + half_spread
        
        if self.inventory >= self.inventory_limit:
            bid_quote = 0.0  
        elif self.inventory <= -self.inventory_limit:
            ask_quote = float('inf')  
            
        return {
            "Mid_Price": self.mid_price,
            "Reservation_Price": r,
            "Bid_Price": round(bid_quote, 4),
            "Ask_Price": round(ask_quote, 4),
            "Spread": round(total_spread, 4)
        }

    def simulate_order_flow_hits(self, quotes: dict) -> dict:
        """
        Simulates order flow execution hits via Poisson arrival probabilities.
        """
        delta_bid = quotes["Mid_Price"] - quotes["Bid_Price"]
        delta_ask = quotes["Ask_Price"] - quotes["Mid_Price"]
        
        prob_hit_bid = np.exp(-self.kappa * delta_bid) if quotes["Bid_Price"] > 0 else 0.0
        prob_hit_ask = np.exp(-self.kappa * delta_ask) if quotes["Ask_Price"] < float('inf') else 0.0
        
        bid_filled = np.random.rand() < prob_hit_bid
        ask_filled = np.random.rand() < prob_hit_ask
        
        if bid_filled and ask_filled:
            if np.random.rand() > 0.5:
                bid_filled = False
            else:
                ask_filled = False
                
        return {"Bid_Filled": bid_filled, "Ask_Filled": ask_filled}

    def process_execution_cycle(self, time_step: int, total_steps: int):
        """
        Executes a localized discrete processing step of the market-making loop.
        """
        dt = 1.0 / total_steps
        price_shock = np.random.normal(0, self.volatility * np.sqrt(dt))
        self.mid_price += price_shock
        
        T_minus_t = (total_steps - time_step) / total_steps
        
        quotes = self.generate_asymmetric_quotes(T_minus_t)
        fills = self.simulate_order_flow_hits(quotes)
        
        if fills["Bid_Filled"] and self.inventory < self.inventory_limit:
            self.inventory += 1
            self.cash -= quotes["Bid_Price"]
            self.trades_executed += 1
            
        if fills["Ask_Filled"] and self.inventory > -self.inventory_limit:
            self.inventory -= 1
            self.cash += quotes["Ask_Price"]
            self.trades_executed += 1
            
        portfolio_mtm_value = self.cash + (self.inventory * self.mid_price)
        pnl = portfolio_mtm_value - self.initial_wealth
        
        self.history.append({
            "Step": time_step,
            "Mid": quotes["Mid_Price"],
            "Reservation": quotes["Reservation_Price"],
            "Bid": quotes["Bid_Price"],
            "Ask": quotes["Ask_Price"],
            "Inventory": self.inventory,
            "Cash": self.cash,
            "MTM_Value": portfolio_mtm_value,
            "Net_PnL": pnl
        })

    def compile_backtest_dashboard(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

if __name__ == "__main__":
    print("Initializing Avellaneda-Stoikov algorithmic framework matrix parameters...")
    total_epochs = 500
    
    mm_bot = AvellanedaStoikovMarketMaker(
        initial_mid=100.0, 
        asset_vol=0.012, 
        gamma=0.15, 
        kappa=2.2, 
        inventory_limit=15
    )
    
    print("Running discrete optimization matching loops across simulated order books...")
    for step in range(1, total_epochs + 1):
        mm_bot.process_execution_cycle(time_step=step, total_steps=total_epochs)
        
    results_df = mm_bot.compile_backtest_dashboard()
    final_row = results_df.iloc[-1]
    
    print("\n" + "="*70)
    print("          AKUNA COMPETITION SIMULATOR SCORECARD PREVIEW             ")
    print("="*70)
    print(f"Total Trading Step Epochs processed  : {total_epochs}")
    print(f"Total Order Book Fills Captured      : {mm_bot.trades_executed}")
    print(f"Final Closing Underlying Mid Asset   : ${final_row['Mid']:.4f}")
    print(f"Terminal Bot Position Inventory Stance: {int(final_row['Inventory'])} contracts")
    print(f"Terminal Liquid Free Cash Balance     : ${final_row['Cash']:,.2f}")
    print(f"Net Portfolio Alpha PnL Generated    : ${final_row['Net_PnL']:,.2f}")
    
    max_inv = results_df['Inventory'].abs().max()
    print(f"Maximum Peak Inventory Stress Strain : {max_inv} / {mm_bot.inventory_limit}")
    print("="*70 + "\n")
