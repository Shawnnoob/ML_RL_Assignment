from DQN_v2 import SmartVentilationEnv, BoltzmannDQN, evaluate_model, debug_evaluate_model
from stable_baselines3 import DQN
import os

def load_and_evaluate_models(model_names):
    print("\n================ FINAL EVALUATION RESULTS ================\n")

    # Fixed benchmark environment
    env_eval = SmartVentilationEnv()

    results = []

    for name in model_names:
        model_path = f"models/model_{name}.zip"

        if not os.path.exists(model_path):
            print(f"[SKIP] Model not found: {model_path}")
            continue

        # Decide which class to load
        if name.startswith("Ent_"):
            model = BoltzmannDQN.load(model_path, env=env_eval)
        else:
            model = DQN.load(model_path, env=env_eval)

        stats = debug_evaluate_model(
            model,
            env_eval,
            #label=name,
            #print_logs=True
        )

        results.append((name, stats))

    # ---- Summary Table ----
    print("\n================ SUMMARY TABLE =================")
    print(f"{'Model':<28} | {'Reward':>8} | {'TASVT':>8} | {'Avg CO2':>8} | {'Avg PM':>8}")
    print("-" * 75)

    for name, s in results:
        print(
            f"{name:<28} | "
            f"{s['Reward']:>8.0f} | "
            f"{s['TASVT']:>8} | "
            f"{s['Avg_CO2']:>8.0f} | "
            f"{s['Avg_PM']:>8.1f}"
        )

    print("=" * 75)


# --------------------------------------------------
# Run evaluation on all trained models
# --------------------------------------------------

MODEL_NAMES = [
    # Epsilon-greedy
    "Eps_Default",
    "Eps_LrLow_GamLow",
    "Eps_LrLow_GamHigh",
    "Eps_LrHigh_GamLow",
    "Eps_LrHigh_GamHigh",

    # Random baseline
    "Rnd_Default",

    # Entropy-based (Boltzmann)
    "Ent_Default",
    "Ent_LrLow_GamLow",
    "Ent_LrLow_GamHigh",
    "Ent_LrHigh_GamLow",
    "Ent_LrHigh_GamHigh",
]

load_and_evaluate_models(MODEL_NAMES)