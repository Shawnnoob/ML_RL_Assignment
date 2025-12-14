from DQN_v2 import SmartVentilationEnv, evaluate_model, BoltzmannDQN, DoubleDQN, BoltzmannDoubleDQN, debug_evaluate_model
from stable_baselines3 import DQN
import os

env_eval = SmartVentilationEnv()

# Standard values
LR_DEF, LR_LOW, LR_HIGH = 1e-4, 1e-5, 1e-3  # Learning rate
GAM_DEF, GAM_LOW, GAM_HIGH = 0.99, 0.90, 0.999  # Discount factor

configs = [
    # --- 1.1 Epsilon Greedy (Standard DQN) ---
    {"name": "Eps_Default", "class": DQN, "lr": LR_DEF, "gamma": GAM_DEF,
        "kwargs": {"exploration_fraction": 0.5}},
    {"name": "Eps_LrLow_GamLow", "class": DQN, "lr": LR_LOW, "gamma": GAM_LOW,
        "kwargs": {"exploration_fraction": 0.5}},
    {"name": "Eps_LrLow_GamHigh", "class": DQN, "lr": LR_LOW, "gamma": GAM_HIGH,
        "kwargs": {"exploration_fraction": 0.5}},
    {"name": "Eps_LrHigh_GamLow", "class": DQN, "lr": LR_HIGH, "gamma": GAM_LOW,
        "kwargs": {"exploration_fraction": 0.5}},
    {"name": "Eps_LrHigh_GamHigh", "class": DQN, "lr": LR_HIGH, "gamma": GAM_HIGH,
        "kwargs": {"exploration_fraction": 0.5}},

    # --- 1.2 Random Exploration (Baseline DQN) ---
    {"name": "Rnd_Default", "class": DQN, "lr": LR_DEF, "gamma": GAM_DEF,
        "kwargs": {"exploration_initial_eps": 1.0, "exploration_final_eps": 1.0}},

    # --- 1.3 Entropy-Based (Boltzmann DQN) ---
    {"name": "Ent_Default", "class": BoltzmannDQN, "lr": LR_DEF, "gamma": GAM_DEF, "kwargs": {"temperature": 1.0}},
    {"name": "Ent_LrLow_GamLow", "class": BoltzmannDQN, "lr": LR_LOW, "gamma": GAM_LOW,
        "kwargs": {"temperature": 1.0}},
    {"name": "Ent_LrLow_GamHigh", "class": BoltzmannDQN, "lr": LR_LOW, "gamma": GAM_HIGH,
        "kwargs": {"temperature": 1.0}},
    {"name": "Ent_LrHigh_GamLow", "class": BoltzmannDQN, "lr": LR_HIGH, "gamma": GAM_LOW,
        "kwargs": {"temperature": 1.0}},
    {"name": "Ent_LrHigh_GamHigh", "class": BoltzmannDQN, "lr": LR_HIGH, "gamma": GAM_HIGH,
        "kwargs": {"temperature": 1.0}},

# --- 2.1 Epsilon Greedy (Standard DDQN) ---
    {"name": "DDQN_Eps_Default", "class": DoubleDQN, "lr": LR_DEF, "gamma": GAM_DEF,
        "kwargs": {"exploration_fraction": 0.5}},
    {"name": "DDQN_Eps_LrLow_GamLow", "class": DoubleDQN, "lr": LR_LOW, "gamma": GAM_LOW,
        "kwargs": {"exploration_fraction": 0.5}},
    {"name": "DDQN_Eps_LrLow_GamHigh", "class": DoubleDQN, "lr": LR_LOW, "gamma": GAM_HIGH,
        "kwargs": {"exploration_fraction": 0.5}},
    {"name": "DDQN_Eps_LrHigh_GamLow", "class": DoubleDQN, "lr": LR_HIGH, "gamma": GAM_LOW,
        "kwargs": {"exploration_fraction": 0.5}},
    {"name": "DDQN_Eps_LrHigh_GamHigh", "class": DoubleDQN, "lr": LR_HIGH, "gamma": GAM_HIGH,
        "kwargs": {"exploration_fraction": 0.5}},

    # --- 2.2 Random Exploration (Baseline DDQN) ---
    {"name": "DDQN_Rnd_Default", "class": DoubleDQN, "lr": LR_DEF, "gamma": GAM_DEF,
        "kwargs": {"exploration_initial_eps": 1.0, "exploration_final_eps": 1.0}},

    # --- 2.3 Entropy-Based (Boltzmann DDQN) ---
    # Uses our custom class
    {"name": "DDQN_Ent_Default", "class": BoltzmannDoubleDQN, "lr": LR_DEF, "gamma": GAM_DEF,
        "kwargs": {"temperature": 1.0}},
    {"name": "DDQN_Ent_LrLow_GamLow", "class": BoltzmannDoubleDQN, "lr": LR_LOW, "gamma": GAM_LOW,
        "kwargs": {"temperature": 1.0}},
    {"name": "DDQN_Ent_LrLow_GamHigh", "class": BoltzmannDoubleDQN, "lr": LR_LOW, "gamma": GAM_HIGH,
        "kwargs": {"temperature": 1.0}},
    {"name": "DDQN_Ent_LrHigh_GamLow", "class": BoltzmannDoubleDQN, "lr": LR_HIGH, "gamma": GAM_LOW,
        "kwargs": {"temperature": 1.0}},
    {"name": "DDQN_Ent_LrHigh_GamHigh", "class": BoltzmannDoubleDQN, "lr": LR_HIGH, "gamma": GAM_HIGH,
        "kwargs": {"temperature": 1.0}},
    ]

# ==========================================================================
# 4. Final Model Loading & Evaluation
# ==========================================================================
print("\n" + "=" * 80)
print("FINAL BENCHMARK: LOADING SAVED MODELS")
print("=" * 80)
print(f"{'Model Name':<30} | {'Reward':<10} | {'TASVT':<8} | {'Avg CO2':<10} | {'Avg PM':<10}")
print("-" * 80)

# We use env_eval (which resets to seed 42) to benchmark them all fairly
for config in configs:
    name = config["name"]
    model_path = os.path.join("models", f"model_{name}.zip")

    if not os.path.exists(model_path):
        print(f"{name:<30} | {'[NOT FOUND]':<45}")
        continue

    # Identify the class to load
    model_class = config["class"]

    try:
        # Load the model
        loaded_model = model_class.load(model_path, env=env_eval)

        # Evaluate using the standard function (Deterministic=True)
        stats = evaluate_model(loaded_model, env_eval, label=name, print_logs=False)

        # Print row
        print(
            f"{name:<30} | {stats['Reward']:<10.0f} | {stats['TASVT']:<8} | {stats['Avg_CO2']:<10.1f} | {stats['Avg_PM']:<10.1f}")

    except Exception as e:
        print(f"{name:<30} | [ERROR LOADING]: {e}")

print("-" * 80)
print("[INFO] Benchmark Complete.")

#debug_evaluate_model(model=DQN.load("models/model_Rnd_Default.zip"), env=env_eval, not_random_model=False)