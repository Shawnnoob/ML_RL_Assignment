import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn.functional as F
import os
import json

class SmartVentilationEnv(gym.Env):
    MAX_STEPS = 1440

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # --- Constants ---
        self.V = 82.72  # Kitchen Volume (m³)

        # --- Pollutant Safe Limits ---
        self.voc_safe = 500.0
        self.pm_safe = 35.0
        self.co2_safe = 2000.0
        self.co2_background = 400.0

        # --- Observation Space (Normalized for model) ---
        # 1. Norm_VOC, 2. Norm_PM, 3. Norm_CO2, 4. Norm_Activity, 5. Norm_AirFlow
        obs_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([np.inf, np.inf, np.inf, 1.0, 1.0], dtype=np.float32)
        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.action_space = Discrete(6) # Inactive + 5 actions
        self.action_space_map = {
            0: 0.0,
            1: -0.20, 2: -0.10, 3: +0.00, 4: +0.10, 5: +0.20
        }

        # --- Fan Specs ---
        self.fan_low_watt = 40.8
        self.fan_high_watt = 52.9
        self.fan_slope = 13.44
        self.fan_intercept = 39.46
        self.Q_MAX_ON = 4.2
        self.Q_MIN_ON = 2.5

        # --- VOC EMISSION ---
        # 1 ppb VOC = 4.5 ug/m3
        # We calculate rates assuming pollutants accumulated over a fixed cooking duration.
        # Derivation of Duration from PM2.5 Data:
        # PM Mass = Conc (92.9) * Vol (82.72) = 7684.7 ug
        # Duration = Mass (7684.7) / PM_Rate (596) = 12.9 ≈ 13 minutes
        voc_conc_factor = 4.5
        kitchen_vol = 82.72
        cooking_duration_mins = 13

        # Formula: Rate = (Concentration_Increase * Volume) / Duration
        voc_factor = (voc_conc_factor * kitchen_vol) / cooking_duration_mins
        self.EMISSION_VOC_RATE_MAP = {
            0: 0.0,
            1: 20.0 * voc_factor,  # Air-frying
            2: 30.0 * voc_factor,  # Boiling
            3: 110.0 * voc_factor,  # Stir-frying
            4: 230.0 * voc_factor,  # Deep-frying
            5: 260.0 * voc_factor  # Pan-frying
        }

        # --- PM EMISSION (Mass Rates) ---
        self.EMISSION_PM_RATE_MAP = {
            0: 0.0,
            1: 1.1, # Air-frying
            2: 1.1, # Boiling
            3: 178.0, # Stir-frying
            4: 47.0, # Deep-frying
            5: 596.0 # Pan-frying
        }

        # --- CO2 EMISSION ---
        # Calculation: 299 ppm increase over 45-60 min
        # Rate: Random range between 4.98 and 6.65 ppm/min
        self.CO2_EMISSION_MIN = 4.98
        self.CO2_EMISSION_MAX = 6.65

        # --- Init variables ---
        self.voc = 0.0
        self.pm = 0.0
        self.co2 = 400.0
        self.fan_speed = 0.0
        self.current_step = 0
        self.activity_index = 0
        self.activity_list = ["none", "air", "boil", "stir", "deep", "pan"]
        # Cooking related settings
        self.steps_remaining_in_activity = 0
        self.meal_tracker = {"breakfast": False, "lunch": False, "dinner": False}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.voc = self.np_random.uniform(10.0, self.voc_safe / 2.0)
        self.pm = self.np_random.uniform(1.0, self.pm_safe / 2.0)
        self.co2 = self.np_random.uniform(400.0, 800.0)
        self.fan_speed = 0.0 # Fan turned OFF
        self.activity_index = 0 # No cooking activity
        self.current_step = 0 # Start of the day
        self.steps_remaining_in_activity = 0 # Ensure the meals are done once per day
        self.meal_tracker = {"breakfast": False, "lunch": False, "dinner": False}

        return self._get_obs(), {"initial_activity": self.activity_list[self.activity_index]}

    def _get_obs(self):
        # 1. Normalize Pollutants by their Safe Limit (Neural Networks learn better)
        # Value 1.0 means "Exactly at threshold", value 2.0 means "Double the limit"
        # 0.5 - 50% of limit, 1.0 - 100% of limit, 2.0 - 200% of limit
        n_voc = self.voc / self.voc_safe
        n_pm = self.pm / self.pm_safe
        n_co2 = self.co2 / self.co2_safe

        # 2. Normalize Activity (0 to 5) -> (0.0 to 1.0)
        n_act = self.activity_index / 5.0

        # 3. Normalize Airflow (0 to 4.2) -> (0.0 to 1.0)
        current_airflow = self._get_fan_airflow()
        n_flow = current_airflow / self.Q_MAX_ON

        return np.array([n_voc, n_pm, n_co2, n_act, n_flow], dtype=np.float32)

    def _update_fan_speed(self, action: int):
        if action == 0:
            self.fan_speed = 0.0
        else:
            change = self.action_space_map.get(action)
            self.fan_speed = np.clip(self.fan_speed + change, 0.0, 1.0)

    def _get_fan_airflow(self) -> float:
        if self.fan_speed == 0.0: return 0.0
        # Linear: y = 1.89x + 2.31
        return np.clip(1.89 * self.fan_speed + 2.31, 2.5, 4.2)

    def _update_pollutants(self):
        # 1. Get emission rates
        base_voc = self.EMISSION_VOC_RATE_MAP.get(self.activity_index, 0.0)
        base_pm = self.EMISSION_PM_RATE_MAP.get(self.activity_index, 0.0)
        if self.activity_index > 0:
            # Noise level
            noise_scale = 0.05
            # Clip at 0.0 to prevent negative emissions
            E_VOC = max(0.0, np.random.normal(base_voc, base_voc * noise_scale))
            E_PM = max(0.0, np.random.normal(base_pm, base_pm * noise_scale))
            E_CO2_rate = self.np_random.uniform(self.CO2_EMISSION_MIN, self.CO2_EMISSION_MAX)
        else:
            E_VOC = 0.0
            E_PM = 0.0
            E_CO2_rate = 0.0

        # 2. Get ventilation
        Q_flow = self._get_fan_airflow()
        removal = Q_flow / self.V  # Removal fraction per minute

        # 3. Update pollutants
        self.voc += (E_VOC / self.V) - (self.voc * removal)
        self.pm += (E_PM / self.V) - (self.pm * removal)
        self.co2 += E_CO2_rate - ((self.co2 - self.co2_background) * removal)

        # 4. Check boundary
        self.voc = max(0.0, self.voc)
        self.pm = max(0.0, self.pm)
        self.co2 = max(self.co2_background, self.co2)

    def _get_fan_power(self, speed):
        if speed == 0:
            return 0.0
        else:
            return 13.44 * speed + 39.46

    def _reward_func(self, current, limit):
        if current <= limit:
            return 1.0
        else:
            ratio = current / limit
            # --- EXPONENTIAL PENALTY ---
            # Steepness (k): Controls how fast the penalty grows.
            # k=5.0 means at 50% overflow (1.5 ratio), penalty is ~ -11.0
            steepness = 1.0

            # We subtract 1.0 so that at ratio=1.0, the penalty is exactly 0.0
            penalty = -(np.exp((ratio - 1.0) * steepness) - 1.0)

            # Apply hard cap to prevent mathematical explosion
            return penalty

    def _calculate_total_reward(self):
        R_voc = self._reward_func(self.voc, self.voc_safe)
        R_pm = self._reward_func(self.pm, self.pm_safe)
        R_co2 = self._reward_func(self.co2, self.co2_safe)

        # Energy Reward: +1 if Off, negative based on power if On
        power = self._get_fan_power(self.fan_speed)
        if power == 0:
            R_energy = 1.0
        else:
            R_energy = -(power / self.fan_high_watt)

        total = (self.alpha * R_voc + self.beta * R_co2 +
                 self.gamma * R_pm + self.delta * R_energy)

        return {
            "R_total": total,
            "R_voc": R_voc,
            "R_pm": R_pm,
            "R_co2": R_co2,
            "R_energy": R_energy
        }

    def _update_activity(self):
        """
        Manages the daily schedule (Breakfast, Lunch, Dinner) with
        randomized activity types and durations (20-45 min).
        """
        step = self.current_step

        # 1. If currently cooking, continue cooking and countdown
        if self.steps_remaining_in_activity > 0:
            self.steps_remaining_in_activity -= 1
            # If time runs out, stop cooking
            if self.steps_remaining_in_activity <= 0:
                self.activity_index = 0  # Turn off activity
            return self.activity_index

        # 2. Define Time Windows (Step counts)
        # Breakfast: 7:00 AM - 9:00 AM (420 - 540)
        breakfast_start, breakfast_end = 420, 540
        # Lunch: 12:00 PM - 2:00 PM (720 - 840)
        lunch_start, lunch_end = 720, 840
        # Dinner: 6:00 PM - 8:00 PM (1080 - 1200)
        dinner_start, dinner_end = 1080, 1200

        # 3. Check windows and trigger new activities

        # --- Breakfast Logic ---
        if breakfast_start <= step < breakfast_end and not self.meal_tracker['breakfast']:
            self._start_random_activity()
            self.meal_tracker['breakfast'] = True

        # --- Lunch Logic ---
        elif lunch_start <= step < lunch_end and not self.meal_tracker['lunch']:
            self._start_random_activity()
            self.meal_tracker['lunch'] = True

        # --- Dinner Logic ---
        elif dinner_start <= step < dinner_end and not self.meal_tracker['dinner']:
            self._start_random_activity()
            self.meal_tracker['dinner'] = True

        else:
            # No activity scheduled, ensure index is 0
            self.activity_index = 0

        return self.activity_index

    def _start_random_activity(self):
        """Helper to pick random activity and duration"""
        # Pick random activity from 1 to 5 (exclude 0/None)
        # 1: Air, 2: Boil, 3: Stir, 4: Deep, 5: Pan
        self.activity_index = self.np_random.integers(1, 6)

        # Pick random duration between 20 and 45 minutes
        duration = self.np_random.integers(20, 46)
        self.steps_remaining_in_activity = duration

    def step(self, action: int):
        self._update_fan_speed(action)
        self._update_pollutants()
        self.current_step += 1

        # Update activity logic (check schedule or continue cooking)
        self._update_activity()

        # Extract scalar 'R_total', full dict is available if needed
        reward_details = self._calculate_total_reward()
        reward = reward_details["R_total"]

        observation = self._get_obs()
        terminated = False
        truncated = self.current_step >= self.MAX_STEPS

        info = {
            "activity": self.activity_list[self.activity_index],
            "fan_power_W": self._get_fan_power(self.fan_speed),
            "reward_details": reward_details,
            "voc": self.voc,
            "pm": self.pm,
            "co2": self.co2
        }

        return observation, reward, terminated, truncated, info

class BoltzmannDQN(DQN):
    """
    Subclass of DQN that replaces Epsilon-Greedy with Boltzmann (Softmax) Exploration.
    This fulfills the 'Entropy-based exploration' requirement by selecting actions
    proportionally to their Q-values.
    """
    def __init__(self, *args, temperature=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def _sample_action(self, learning_starts: int, action_noise=None, n_envs: int = 1):
        # 1. Select action randomly if we haven't started learning yet
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
            return unscaled_action, unscaled_action

        # 2. Get Q-values from the policy
        with torch.no_grad():
            obs_tensor = self.policy.obs_to_tensor(self._last_obs)[0]
            q_values = self.policy.q_net(obs_tensor) # Shape: [n_envs, n_actions]

        # 3. Apply Softmax (Boltzmann)
        # Prob(a) = exp(Q(a)/T) / sum(exp(Q/T))
        probs = F.softmax(q_values / self.temperature, dim=1).cpu().numpy()

        # 4. Sample action based on probabilities
        actions = []
        for i in range(n_envs):
            action = np.random.choice(self.action_space.n, p=probs[i])
            actions.append(action)

        return np.array(actions), np.array(actions)


def evaluate_model(model, env, label="Model", print_logs=False):
    # Evaluates on a FIXED Benchmark (Seed 42) for fair comparison
    obs, _ = env.reset(seed=42)  # Force same day scenario

    cumulative_reward = 0.0
    tasvt_steps = 0
    total_co2 = 0.0
    total_pm = 0.0

    is_random = "Random" in label
    deterministic = not is_random

    for _ in range(env.MAX_STEPS):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, _, trunc, info = env.step(action.item())

        cumulative_reward += reward
        total_co2 += info['co2']
        total_pm += info['pm']
        if info['voc'] > env.voc_safe:
            tasvt_steps += 1
        if trunc: break

    avg_co2 = total_co2 / env.MAX_STEPS
    avg_pm = total_pm / env.MAX_STEPS

    if print_logs:
        print(f"[{label}] R: {cumulative_reward:.0f} | Unsafe: {tasvt_steps} | CO2: {avg_co2:.0f} | PM: {avg_pm:.0f}")

    return {"Reward": cumulative_reward, "TASVT": tasvt_steps, "Avg_CO2": avg_co2, "Avg_PM": avg_pm}


def debug_evaluate_model(model, env, seed=42):
    """
    Prints step-by-step logs including instantaneous Step Reward and Cumulative Reward.
    """
    print(f"\n[DEBUG] Detailed Evaluation on Seed {seed}...")

    # 1. Update Header to include 'StepRew'
    print(
        f"{'Step':<6} | {'Time':<6} | {'Activity':<10} | {'Fan':<5} | {'VOC':<8} | {'PM2.5':<6} | {'CO2':<6} | {'StepRew':<8} | {'CumRew':<8} | {'TASVT':<5}")
    print("-" * 100)

    obs, _ = env.reset(seed=seed)

    total_unsafe = 0
    cumulative_reward = 0.0

    while True:
        action, _ = model.predict(obs, deterministic=True)

        # 2. Capture the single-step reward
        obs, reward, _, truncated, info = env.step(action.item())

        cumulative_reward += reward
        step = env.current_step

        if info['voc'] > env.voc_safe:
            total_unsafe += 1

        # 3. Print logic
        if (info['activity'] != "none") or (env.fan_speed > 0) or (info['voc'] > env.voc_safe):
            time_s = f"{(step // 60) % 24:02d}:{step % 60:02d}"
            voc_s = f"{info['voc']:.1f}" + ("!" if info['voc'] > env.voc_safe else "")

            # 4. Add 'reward' to the print statement
            print(
                f"{step:<6} | {time_s:<6} | {info['activity']:<10} | {env.fan_speed:<5.2f} | {voc_s:<8} | {info['pm']:<6.1f} | {info['co2']:<6.1f} | {reward:<8.1f} | {cumulative_reward:<8.0f} | {total_unsafe:<5}")

        if truncated: break

    print("-" * 100)
    print(f"[DEBUG] Finished. Final Reward: {cumulative_reward:.0f} | Total Unsafe Minutes: {total_unsafe}\n")


def run_experiment(config, env_train, env_eval, total_steps, eval_freq):
    name = config["name"]
    print(f"\n>>> Starting Experiment: {name}")

    # 1. Initialize Model on Training Environment
    model = config["class"](
        "MlpPolicy",
        env_train,
        learning_rate=config["lr"],
        gamma=config["gamma"],
        seed=42,
        verbose=0,
        **config["kwargs"]
    )

    history = []
    cycles = total_steps // eval_freq
    current_step = 0

    for cycle in range(cycles):
        # 2. Train (on continuous random days)
        model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)
        current_step += eval_freq

        # 3. Evaluate (on fixed benchmark day)
        stats = evaluate_model(model, env_eval, label=f"{name} (Step {current_step})", print_logs=True)
        history.append({"step": current_step, **stats})

    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Define paths
    model_path = os.path.join("models", f"model_{name}.zip")
    log_path = os.path.join("logs", f"log_{name}.json")

    # Save
    model.save(model_path)
    with open(log_path, "w") as f:
        json.dump(history, f)

    print(f"[INFO] Saved: {model_path} and {log_path}")
    return history, model


def plot_from_logs(log_filenames, metric_key="Reward", title=None, save_name=None,
                   use_symlog=False, symlog_scale=5000, y_axis_ticks=None):
    """
    Args:
        use_symlog (bool): If True, applies symmetric log scaling to handle massive negative drops.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Formatting
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    linestyles = ['-', '--', '-.', ':']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    found_any = False

    for i, filename in enumerate(log_filenames):
        if not os.path.exists(filename):
            print(f"[WARN] File not found: {filename}")
            continue

        found_any = True
        with open(filename, 'r') as f:
            history = json.load(f)

        # 1440 minutes = 1 Day
        steps = [entry['step'] for entry in history]
        days = [step / 1440.0 for step in steps]

        values = [entry[metric_key] for entry in history]

        clean_label = filename.replace("log_", "").replace(".json", "")

        # Plot using 'days' on X-axis instead of 'steps'
        ax.plot(days, values, label=clean_label,
                marker=markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)],
                color=colors[i % len(colors)],
                linewidth=2, markersize=5, alpha=0.8)

    if not found_any: return

    # --- THE SCALING FIX ---
    if use_symlog:
        # 1. Enable Symlog
        ax.set_yscale('symlog', linthresh=symlog_scale)

        # 2. Force Readable Labels (No 10^x scientific notation)
        # This forces the axis to use plain numbers (e.g. 5000, -100000)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)  # Turn off 1e6 notation
        ax.yaxis.get_major_formatter().set_useOffset(False)

        # 3. Manually set ticks if auto-detection fails
        # Uncomment this only if the axis is still blank
        if y_axis_ticks is not None:
            # y_axis_ticks = [int(tick) for tick in y_axis_ticks]
            ax.set_yticks(y_axis_ticks)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    final_title = title if title else f"Comparison: {metric_key}"
    plt.title(final_title, fontsize=16)
    plt.xlabel("Training Days", fontsize=12)
    plt.ylabel(metric_key, fontsize=12)
    plt.legend(fontsize=10, loc='best')

    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"[INFO] Plot saved to {save_name}")

    plt.show()


if __name__ == "__main__":
    # Create TWO Separate Environments
    env_train = SmartVentilationEnv()
    env_eval = SmartVentilationEnv()

    # Define Experiments
    DAYS_TO_TRAIN = 300
    TOTAL_STEPS = DAYS_TO_TRAIN * 1440
    RECORD_FREQUENCY = 15  # No. of days before a single record
    EVAL_FREQ = RECORD_FREQUENCY * 1440  # Evaluate every n days

    # Standard values
    LR_DEF, LR_LOW, LR_HIGH = 1e-4, 1e-5, 1e-3  # Learning rate
    GAM_DEF, GAM_LOW, GAM_HIGH = 0.99, 0.90, 0.999  # Discount factor

    configs = [
        # --- 1. Epsilon Greedy (Standard DQN) ---
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

        # --- 2. Random Exploration (Baseline) ---
        # Fixed epsilon = 1.0 means pure random
        {"name": "Rnd_Default", "class": DQN, "lr": LR_DEF, "gamma": GAM_DEF,
         "kwargs": {"exploration_initial_eps": 1.0, "exploration_final_eps": 1.0}},

        # --- 3. Entropy-Based (Boltzmann DQN) ---
        # Uses our custom class

        {"name": "Ent_Default", "class": BoltzmannDQN, "lr": LR_DEF, "gamma": GAM_DEF, "kwargs": {"temperature": 1.0}},
        {"name": "Ent_LrLow_GamLow", "class": BoltzmannDQN, "lr": LR_LOW, "gamma": GAM_LOW,
         "kwargs": {"temperature": 1.0}},
        {"name": "Ent_LrLow_GamHigh", "class": BoltzmannDQN, "lr": LR_LOW, "gamma": GAM_HIGH,
         "kwargs": {"temperature": 1.0}},
        {"name": "Ent_LrHigh_GamLow", "class": BoltzmannDQN, "lr": LR_HIGH, "gamma": GAM_LOW,
         "kwargs": {"temperature": 1.0}},
        {"name": "Ent_LrHigh_GamHigh", "class": BoltzmannDQN, "lr": LR_HIGH, "gamma": GAM_HIGH,
         "kwargs": {"temperature": 1.0}},

    ]

    all_results = {}

    # C. Run Loop
    for config in configs:
        history, model = run_experiment(config, env_train, env_eval, TOTAL_STEPS, EVAL_FREQ)
        all_results[config["name"]] = history

    os.makedirs("plots", exist_ok=True)

    # my_logs = [
    #     "logs/log_Eps_Default.json",
    #     "logs/log_Eps_LrHigh_GamHigh.json",
    #     "logs/log_Eps_LrLow_GamLow.json",
    #     "logs/log_Rnd_Default.json"
    # ]
    #
    # plot_from_logs(
    #     my_logs,
    #     metric_key="Reward",
    #     title="Epsilon DQN Reward Curve (Log Scale)",
    #     save_name="plots/Epsilon_DQN_Reward.png",
    #     use_symlog=True,
    #     symlog_scale=5000,
    #     y_axis_ticks=[-3000, -2000, -1000, 0, 1000, 2000, 3000, 4000, 5000]
    # )
    #
    # plot_from_logs(
    #     my_logs,
    #     metric_key="TASVT",
    #     title="Epsilon DQN Unsafe Minutes",
    #     save_name="plots/Epsilon_DQN_Unsafe.png",
    #     use_symlog=False
    # )
    #
    # my_logs = [
    #     "logs/log_Ent_Default.json",
    #     "logs/log_Ent_LrHigh_GamHigh.json",
    #     "logs/log_Ent_LrLow_GamLow.json",
    #     "logs/log_Rnd_Default.json"
    # ]
    #
    # plot_from_logs(
    #     my_logs,
    #     metric_key="Reward",
    #     title="Entropy DQN Reward Curve (Log Scale)",
    #     save_name="plots/Entropy_DQN_Reward.png",
    #     use_symlog=True,
    #     symlog_scale=6000,
    #     y_axis_ticks=[-1350000, 0, 1000, 2000, 3000, 4000, 5000]
    # )
    #
    # plot_from_logs(
    #     my_logs,
    #     metric_key="TASVT",
    #     title="Entropy DQN Unsafe Minutes",
    #     save_name="plots/Entropy_DQN_Unsafe.png",
    #     use_symlog=False
    # )
    #
    # my_logs = [
    #     "logs/log_Eps_Default.json",
    #     "logs/log_Eps_LrHigh_GamHigh.json",
    #     "logs/log_Ent_Default.json",
    #     "logs/log_Ent_LrHigh_GamHigh.json"
    # ]
    #
    # plot_from_logs(
    #     my_logs,
    #     metric_key="Reward",
    #     title="Summary DQN Reward Curve (Log Scale)",
    #     save_name="plots/Summary_DQN_Reward.png",
    #     use_symlog=True,
    #     symlog_scale=6000,
    #     y_axis_ticks=[1000, 2000, 3000, 4000, 5000, 5500]
    # )
    #
    # plot_from_logs(
    #     my_logs,
    #     metric_key="TASVT",
    #     title="Summary DQN Unsafe Minutes",
    #     save_name="plots/Summary_DQN_Unsafe.png",
    #     use_symlog=False
    # )