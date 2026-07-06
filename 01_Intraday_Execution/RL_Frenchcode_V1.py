# ==============================================================================
# BESS INTRADAY TRADING — V11
# Corrections vs V10 :
#   [1] n_steps=672 dans PPO (aligné avec épisode hebdomadaire → GAE propre)
#   [2] Signal XGBoost normalisé par sa std réelle (pas /50 hardcodé)
#   [3] Features XGBoost enrichies + régularisation subsample
#   [4] R² XGBoost imprimé (sanity check que le ML prédit quelque chose)
#   [5] TqdmCallback : pbar dans _on_training_start, pas dans __init__
# ==============================================================================

# ==============================================================================
# 1. IMPORTS & SETUP
# ==============================================================================
# !pip install -q stable-baselines3[extra] gymnasium matplotlib numpy pandas tqdm requests scikit-learn xgboost

import warnings
import os
import logging
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("gymnasium").setLevel(logging.ERROR)

print("=======================================================================", flush=True)
print("     PHASE 1 : DATA ENGINEERING (AR1) & XGBOOST ENRICHI                ", flush=True)
print("=======================================================================", flush=True)

# ------------------------------------------------------------------------------
# 1.1 Extraction 6 mois EPEX Day-Ahead DE-LU
# ------------------------------------------------------------------------------
try:
    print(">>> Téléchargement de 6 mois d'historique EPEX...", flush=True)
    url = "https://api.energy-charts.info/price?bzn=DE-LU&start=2024-01-01&end=2024-06-30"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame({
        'Datetime': pd.to_datetime(data['unix_seconds'], unit='s', utc=True).map(
            lambda x: x.tz_convert('Europe/Berlin')
        ),
        'DayAhead_Price': data['price']
    }).set_index('Datetime')

    df = df.resample('15min').ffill()

except Exception as e:
    raise Exception(f"Erreur fatale API : {e}")

# ------------------------------------------------------------------------------
# 1.2 Processus AR(1) + choc solaire (structure temporelle exploitable par XGB)
# ------------------------------------------------------------------------------
print(">>> Génération AR(1) + choc solaire...", flush=True)
np.random.seed(42)
ar1 = 0.85
da_prices = df['DayAhead_Price'].values
id_prices = np.zeros(len(df))
id_prices[0] = da_prices[0]
hours = df.index.hour.values

for i in range(1, len(df)):
    solar_effect = (
        -20 * np.sin((hours[i] - 13) * np.pi / 6)
        if 8 <= hours[i] <= 18 else 0
    )
    id_prices[i] = (
        ar1 * id_prices[i - 1]
        + (1 - ar1) * da_prices[i]
        + solar_effect
        + np.random.normal(0, 5)
    )

df['Intraday_Price'] = id_prices

# ------------------------------------------------------------------------------
# 1.3 Feature Engineering — [FIX 3] features enrichies + régularisation XGB
# ------------------------------------------------------------------------------
df['price_lag_1']      = df['Intraday_Price'].shift(1)   # T-15min
df['price_lag_4']      = df['Intraday_Price'].shift(4)   # T-1h
df['price_lag_8']      = df['Intraday_Price'].shift(8)   # T-2h  ← NEW
df['rolling_mean_16']  = df['Intraday_Price'].rolling(16).mean()
df['rolling_std_16']   = df['Intraday_Price'].rolling(16).std()   # ← NEW : volatilité récente
df['hour']             = df.index.hour
df['day_of_week']      = df.index.dayofweek                       # ← NEW : saisonnalité hebdo
df['TARGET_T4']        = df['Intraday_Price'].shift(-4)

df = df.dropna()

# ------------------------------------------------------------------------------
# 1.4 Train / Test split (70 / 30)
# ------------------------------------------------------------------------------
split_idx = int(len(df) * 0.7)
train_df  = df.iloc[:split_idx].copy()
test_df   = df.iloc[split_idx:].copy()

print(f">>> Split : Train ({len(train_df)} steps) | Test OOS ({len(test_df)} steps)", flush=True)

# ------------------------------------------------------------------------------
# 1.5 Entraînement XGBoost
# ------------------------------------------------------------------------------
features = [
    'price_lag_1', 'price_lag_4', 'price_lag_8',
    'rolling_mean_16', 'rolling_std_16',
    'hour', 'day_of_week'
]

xgb_model = XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,          # [FIX 3] régularisation
    colsample_bytree=0.8,   # [FIX 3] régularisation
    random_state=42
)
xgb_model.fit(train_df[features], train_df['TARGET_T4'])

# [FIX 4] Diagnostic R² — si R² < 0.05 le signal est du bruit, arrêter là
r2_train = r2_score(train_df['TARGET_T4'], xgb_model.predict(train_df[features]))
r2_test  = r2_score(test_df['TARGET_T4'],  xgb_model.predict(test_df[features]))
print(f">>> XGBoost R² — Train : {r2_train:.4f} | Test OOS : {r2_test:.4f}", flush=True)
if r2_test < 0.05:
    print("[!] AVERTISSEMENT : R² OOS < 5%. Le signal ML est très faible.", flush=True)

# [FIX 2] Normalisation du signal par sa std réelle (calculée sur train uniquement)
train_df['ML_Raw'] = xgb_model.predict(train_df[features]) - train_df['Intraday_Price']
test_df['ML_Raw']  = xgb_model.predict(test_df[features])  - test_df['Intraday_Price']

SIGNAL_STD = train_df['ML_Raw'].std()   # référence fixée sur train, jamais sur test
train_df['ML_Signal'] = train_df['ML_Raw'] / (SIGNAL_STD + 1e-8)
test_df['ML_Signal']  = test_df['ML_Raw']  / (SIGNAL_STD + 1e-8)

print(f">>> Signal normalisé — std train : {SIGNAL_STD:.2f} €/MWh", flush=True)

# Seuils dynamiques pour la baseline (train uniquement)
P_LOW  = np.percentile(train_df['Intraday_Price'], 20)
P_HIGH = np.percentile(train_df['Intraday_Price'], 80)
print(f">>> Baseline seuils : Achat < {P_LOW:.1f} € | Vente > {P_HIGH:.1f} €", flush=True)

# ==============================================================================
# 2. ENVIRONNEMENT DE TRADING
# ==============================================================================
class EpexVppEnv(gym.Env):
    def __init__(self, df_data, episode_length=672, random_start=True):
        super().__init__()
        self.capacity      = 40.0
        self.max_power     = 20.0
        self.deg_cost      = 31.90
        self.half_deg_cost = self.deg_cost / 2.0
        self.epex_fee      = 0.10

        self.action_space      = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

        self.prices         = df_data['Intraday_Price'].values
        self.signals        = df_data['ML_Signal'].values
        self.max_steps      = len(self.prices)
        self.episode_length = episode_length
        self.random_start   = random_start
        self.current_step   = 0
        self.end_step       = 0
        self.soc            = 0.0

    def _get_obs(self):
        # Prix normalisé /200 | SoC ∈ [0,1] | Signal déjà normalisé par std
        norm_price  = self.prices[self.current_step]  / 200.0
        norm_soc    = self.soc / self.capacity
        norm_signal = self.signals[self.current_step]          # [FIX 2] std=1 garanti
        return np.array([norm_price, norm_soc, norm_signal], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.soc = 0.0
        if self.random_start and self.max_steps > self.episode_length:
            self.current_step = np.random.randint(0, self.max_steps - self.episode_length)
        else:
            self.current_step = 0
        self.end_step = min(self.current_step + self.episode_length, self.max_steps)
        return self._get_obs(), {}

    def step(self, action):
        current_price = self.prices[self.current_step]
        reward        = 0.0
        trading_pnl   = 0.0
        energy_volume = self.max_power * 0.25   # MWh par quart d'heure
        trade_type    = "IDLE"
        penalty_hit   = 0

        if action == 1:   # CHARGE
            if self.soc + energy_volume <= self.capacity:
                self.soc    += energy_volume
                cost         = (current_price + self.half_deg_cost + self.epex_fee) * energy_volume
                reward       = -cost
                trading_pnl  = -cost
                trade_type   = "CHARGE"
            else:
                reward      = -5000.0
                penalty_hit = 1

        elif action == 2:   # DISCHARGE
            if self.soc - energy_volume >= 0:
                self.soc    -= energy_volume
                revenue      = (current_price - self.half_deg_cost - self.epex_fee) * energy_volume
                reward       = revenue
                trading_pnl  = revenue
                trade_type   = "DISCHARGE"
            else:
                reward      = -5000.0
                penalty_hit = 1

        self.current_step += 1
        terminated = bool(self.current_step >= self.end_step)
        obs        = self._get_obs() if not terminated else np.zeros(3, dtype=np.float32)

        info = {
            "real_pnl":    trading_pnl,
            "trade_type":  trade_type,
            "penalty_hit": penalty_hit,
            "volume":      energy_volume
        }
        return obs, reward, terminated, False, info

# ==============================================================================
# 3. BACKTEST
# ==============================================================================
def evaluate_agent(name, env, policy_func):
    obs, _ = env.reset()
    cumulative_pnl = 0.0
    total_volume   = 0.0
    penalties      = 0

    while True:
        action = policy_func(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if info["trade_type"] in ("CHARGE", "DISCHARGE") and info["penalty_hit"] == 0:
            total_volume += info["volume"]
        if info["penalty_hit"] == 1:
            penalties += 1

        cumulative_pnl += info["real_pnl"]
        if terminated:
            break

    return {
        "Name":            name,
        "OOS Net PnL (€)": round(cumulative_pnl, 2),
        "Vol. (MWh)":      round(total_volume, 2),
        "Cycles":          round(total_volume / env.capacity / 2, 1),
        "Fines":           penalties
    }

# [FIX 5] TqdmCallback propre : pbar dans _on_training_start
class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="[RL] Apprentissage (épisodes hebdo)",
            unit=" steps"
        )

    def _on_step(self):
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()

# ==============================================================================
# 4. ENTRAÎNEMENT PPO
# ==============================================================================
print("\n=======================================================================", flush=True)
print("     PHASE 2 : ENTRAÎNEMENT PPO (500k STEPS / ÉPISODES 7J)             ", flush=True)
print("=======================================================================", flush=True)

train_env = EpexVppEnv(train_df, episode_length=672, random_start=True)

model = PPO(
    "MlpPolicy",
    train_env,
    verbose=0,
    learning_rate=0.0003,
    ent_coef=0.02,
    batch_size=256,
    n_steps=672,   # [FIX 1] aligné avec épisode → 1 collecte = 1 semaine exacte → GAE propre
    device='cpu'
)

total_steps = 500_000
model.learn(total_timesteps=total_steps, callback=TqdmCallback(total_steps))

# ==============================================================================
# 5. ÉVALUATION OOS
# ==============================================================================
print("\n=======================================================================", flush=True)
print("     PHASE 3 : EVALUATION OUT-OF-SAMPLE (2 MOIS TEST INCONNUS)         ", flush=True)
print("=======================================================================", flush=True)

test_env = EpexVppEnv(test_df, episode_length=len(test_df), random_start=False)

def naive_policy(obs):
    price_real = obs[0] * 200.0
    soc_real   = obs[1] * 40.0
    if price_real < P_LOW  and soc_real < 40.0: return 1
    if price_real > P_HIGH and soc_real > 0.0:  return 2
    return 0

res_baseline = evaluate_agent("Baseline Dynamique",   test_env, naive_policy)
res_rl       = evaluate_agent("Agent RL + XGBoost V11", test_env,
                               lambda obs: model.predict(obs, deterministic=True)[0])

df_res = pd.DataFrame([res_baseline, res_rl]).set_index("Name")
print(df_res.to_string())

alpha = res_rl["OOS Net PnL (€)"] - res_baseline["OOS Net PnL (€)"]
print(f"\n>>> ALPHA OOS : {alpha:+.2f} €")
print(f">>> XGBoost R² OOS : {r2_test:.4f}")
