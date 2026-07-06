import numpy as np
import pandas as pd

def simulate_mrjd_additive(F_t, kappa, sigma, prob_jump, mu_j, sigma_j=50, n_paths=1000, seed=42):
    """
    Monte Carlo simulation using Additive Ornstein-Uhlenbeck Jump Diffusion.
    Models residual grid noise over deterministic baselines.
    """
    np.random.seed(seed)
    n_hours = len(F_t)
    all_paths = np.zeros((n_paths, n_hours))

    for i in range(n_paths):
        X = np.zeros(n_hours)
        for t in range(1, n_hours):
            diffusion = sigma * np.random.normal()
            jump = np.random.normal(mu_j, sigma_j) if (F_t[t] < 50 and np.random.rand() < prob_jump) else 0
            
            # Euler-Maruyama Discretization
            X[t] = X[t-1] - kappa * X[t-1] + diffusion + jump

        # Map back to Spot Price with EPEX regulatory bounds (-600 to 1000 EUR/MWh)
        prices = np.clip(F_t + X, -600.0, 1000.0)
        all_paths[i, :] = prices

    return all_paths

if __name__ == "__main__":
    # Dummy execution for public demonstration
    # Simulating a flattened 24h HPFC curve dipping slightly at midday
    dummy_hpfc = np.array([80, 75, 70, 65, 60, 50, 40, 20, 0, -10, -15, -20, -25, -10, 10, 40, 80, 120, 150, 130, 100, 90, 85, 80])
    
    # Run MLE calibrated parameters
    simulated_paths = simulate_mrjd_additive(
        F_t=dummy_hpfc,
        kappa=0.105, 
        sigma=12.5,
        prob_jump=0.15,
        mu_j=-150,
        n_paths=100
    )
    print("Simulation Complete. Output Shape:", simulated_paths.shape)
