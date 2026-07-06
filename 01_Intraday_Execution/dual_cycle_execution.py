import numpy as np
import pandas as pd

class BESSTradingEnv:
    """
    Reinforcement Learning environment skeleton for BESS Intraday Arbitrage.
    Focuses on market-taker execution over continuous 15-min EPEX order books.
    """
    def __init__(self, capacity_mw=13, duration_h=4, max_soc=52):
        self.capacity_mw = capacity_mw
        self.max_soc = max_soc
        self.current_soc = 0.0
        
    def cycle_1_solar_rescue(self, current_price, orderbook_depth):
        """
        Triggered during Abregelung (solar overproduction).
        Absorbs energy when prices hit deep negative thresholds.
        """
        # Regulatory EPEX technical floor is -600 EUR/MWh
        if current_price < -50.0 and self.current_soc < self.max_soc:
            charge_volume = min(self.capacity_mw, orderbook_depth)
            self.current_soc += charge_volume
            return f"Executed Charge: {charge_volume} MW at {current_price} EUR"
        return "Hold"

    def cycle_2_scarcity_arbitrage(self, current_price, orderbook_depth):
        """
        Triggered during evening thermal inertia spikes.
        Dumps capacity into the grid when liquidity allows.
        """
        if current_price > 120.0 and self.current_soc > 0:
            discharge_volume = min(self.capacity_mw, self.current_soc, orderbook_depth)
            self.current_soc -= discharge_volume
            return f"Executed Discharge: {discharge_volume} MW at {current_price} EUR"
        return "Hold"

# Note: Full PPO/SAC training loops and proprietary reward functions are omitted for confidentiality.
