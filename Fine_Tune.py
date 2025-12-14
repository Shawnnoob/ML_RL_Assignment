import os
import json
import pandas as pd  # Optional, for pretty printing tables
from stable_baselines3 import DQN
from DQN_v2 import SmartVentilationEnv, evaluate_model

# Define directories for this specific tuning phase
TUNED_MODELS_DIR = "tuned_models"
TUNED_LOGS_DIR = "tuned_logs"
os.makedirs(TUNED_MODELS_DIR, exist_ok=True)
os.makedirs(TUNED_LOGS_DIR, exist_ok=True)


def run_tuning_session(tuning_grid, total_steps, eval_freq, model_class=DQN):
    """
    Orchestrates the tuning process.
    Args:
        tuning_grid (list): List of config dictionaries.
        total_steps (int): Total training steps per model.
        eval_freq (int): How often to evaluate.
        model_class: Class to use (DQN or DoubleDQN). Defaults to DQN.
    """
    summary_results = []

    print(f"\n" + "=" * 80)
    print(f"STARTING HYPERPARAMETER TUNING SESSION ({len(tuning_grid)} Configurations)")
    print(f"Algorithm: {model_class.__name__}")
    print("=" * 80)

    for i, config in enumerate(tuning_grid):
        name = config["name"]
        env_params = config.get("env_params", {})
        model_params = config.get("model_params", {})
        seed = config.get("seed", 42)

        print(f"\n>>> [{i + 1}/{len(tuning_grid)}] Training: {name}")
        print(f"    Env Weights: {env_params}")
        print(f"    Hyperparams: {model_params}")

        # 1. Create CUSTOM Environments for this specific run
        # We must re-create them because Reward Weights (alpha, beta...) change per config
        env_train = SmartVentilationEnv(**env_params)
        env_eval = SmartVentilationEnv(**env_params)

        # 2. Initialize Model
        # EXTRACT KEY HYPERPARAMETERS DIRECTLY FROM GRID
        lr = config.get("learning_rate", 1e-4)  # Default if missing
        gamma = config.get("gamma", 0.999)  # Default if missing

        model = model_class(
            "MlpPolicy",
            env_train,
            learning_rate=lr,
            gamma=gamma,
            seed=seed,
            verbose=0,
            **model_params  # Unpack remaining params (buffer_size, etc.)
        )

        # 3. Training Loop
        history = []
        cycles = total_steps // eval_freq
        current_step = 0

        for cycle in range(cycles):
            model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)
            current_step += eval_freq

            # Evaluate
            stats = evaluate_model(model, env_eval, label=name, print_logs=False)
            history.append({"step": current_step, **stats})

            # Simple inline progress print
            print(f"    Step {current_step}: TASVT={stats['TASVT']} | Reward={stats['Reward']:.0f}")

        # 4. Save Artifacts
        model_path = os.path.join(TUNED_MODELS_DIR, f"{name}.zip")
        log_path = os.path.join(TUNED_LOGS_DIR, f"{name}.json")

        model.save(model_path)
        with open(log_path, "w") as f:
            json.dump(history, f)

        # 5. Record Summary for Comparison
        # Get final metrics
        final_stats = history[-1]

        summary_entry = {
            "Model": name,
            "LR": lr,
            "Gamma": gamma,
            # Flatten weights for display
            "α (VOC)": env_params.get("alpha", 1.0),
            "β (CO2)": env_params.get("beta", 1.0),
            "γ (PM)": env_params.get("gamma", 1.0),
            "δ (Eng)": env_params.get("delta", 1.0),
            # Metrics
            "Final Reward": final_stats["Reward"],
            "Unsafe Mins": final_stats["TASVT"],
            "Avg CO2": final_stats["Avg_CO2"],
            "Avg PM": final_stats["Avg_PM"]
        }
        summary_results.append(summary_entry)

    return summary_results


def display_tuning_results(results):
    """
    Prints a clean, formatted table of the tuning results.
    """
    print("\n" + "=" * 100)
    print("HYPERPARAMETER TUNING FINAL RESULTS")
    print("=" * 100)

    # Try using Pandas for a pretty table if available
    try:
        df = pd.DataFrame(results)
        # Reorder columns for readability
        cols = ["Model", "Unsafe Mins", "Final Reward", "α (VOC)", "γ (PM)", "δ (Eng)", "LR", "Avg CO2", "Avg PM"]
        # Filter only existing columns
        cols = [c for c in cols if c in df.columns]

        print(df[cols].to_string(index=False))
    except ImportError:
        # Fallback to manual printing
        header = f"{'Model':<25} | {'Unsafe':<8} | {'Reward':<10} | {'Weights (a,g,d)':<15} | {'LR':<8}"
        print(header)
        print("-" * 80)
        for r in results:
            w_str = f"{r['α (VOC)']},{r['γ (PM)']},{r['δ (Eng)']}"
            print(f"{r['Model']:<25} | {r['Unsafe Mins']:<8} | {r['Final Reward']:<10.0f} | {w_str:<15} | {r['LR']:<8}")
    print("=" * 100)


# ==============================================================================
# MAIN TUNING BLOCK
# ==============================================================================
if __name__ == "__main__":
    # Settings
    DAYS = 100  # Shorter duration for tuning speed
    STEPS = DAYS * 1440
    FREQ = 25 * 1440

    # --- HYPERPARAMETER GRID ---
    # We mix Physics (Reward Weights) with Brains (Learning Rate)
    # alpha=(VOC), beta=(CO2) gamma=(PM), delta=(energy)
    tuning_grid = [
        # GROUP A: BASELINE (Standard Weights)
        {
            "name": "DQN_Base_HighLR",
            "learning_rate": 1e-3,
            "gamma": 0.999,
            #"env_params": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0},
            "env_params": {"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25},
            "model_params": {"buffer_size": 50000, "exploration_fraction": 0.5},
        },
        {
            "name": "DQN_Base_LowLR",
            "learning_rate": 1e-4,
            "gamma": 0.999,
            #"env_params": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0},
            "env_params": {"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25},
            "model_params": {"buffer_size": 100000, "exploration_fraction": 0.5},
        },

        # GROUP B: SAFETY FOCUSED (High Penalties for Pollution)
        {
            "name": "DQN_Safety_LowLR",
            "learning_rate": 1e-4,
            "gamma": 0.999,
            #"env_params": {"alpha": 2.0, "beta": 1.0, "gamma": 2.0, "delta": 0.5},
            "env_params": {"alpha": 0.35, "beta": 0.20, "gamma": 0.35, "delta": 0.10},
            "model_params": {"exploration_fraction": 0.4},
        },

        # GROUP C: ENERGY FOCUSED (High Penalty for Energy)
        # alpha=1.0, gamma=1.0, delta=3.0 (Energy is expensive!)
        {
            "name": "DQN_Energy_LowLR",
            "learning_rate": 1e-4,
            "gamma": 0.999,
            #"env_params": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 3.0},
            "env_params": {"alpha": 0.20, "beta": 0.10, "gamma": 0.20, "delta": 0.50},
            "model_params": {"exploration_fraction": 0.4},
        },

        # GROUP D: AGGRESSIVE EXPLORATION (Longer epsilon decay)
        {
            "name": "DQN_LongExpl_LowLR",
            "learning_rate": 1e-4,
            "gamma": 0.999,
            #"env_params": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0},
            "env_params": {"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25},
            "model_params": {"exploration_fraction": 0.8},  # Explore for 80% of time
        },
    ]

    # Run Tuning
    results = run_tuning_session(tuning_grid, STEPS, FREQ, model_class=DQN)

    # Display
    display_tuning_results(results)