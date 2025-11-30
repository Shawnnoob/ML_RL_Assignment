# !pip install stable-baselines3 gymnasium
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import DQN


class SmartVentilationEnv(gym.Env):
    MAX_STEPS = 1440
    MAX_PENALTY_COMPONENT = -50.0

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
            # penalty = -((ratio - 1.0) ** 2)
            penalty = -((ratio - 1.0) * 10.0)
            return max(self.MAX_PENALTY_COMPONENT, penalty)

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


# =========================================================================
# 2. RL Agent Training & Testing
# =========================================================================

if __name__ == '__main__':
    # Increase training time slightly to ensure it learns the new normalized state
    TRAIN_STEPS = 144000

    env = SmartVentilationEnv()
    model = DQN("MlpPolicy", env,
                learning_rate=1e-3,
                learning_starts=5000,
                verbose=1,
                seed=42)

    print("--- Training with Normalized States ---")
    model.learn(total_timesteps=TRAIN_STEPS)

    print("\n--- Evaluation ---")
    test_env = SmartVentilationEnv()
    obs, _ = test_env.reset()
    episode_reward = 0.0

    # Header for cleaner output
    print(f"{'Step':<10} | {'Activity':<8} | {'Fan':<5} | {'VOC':<8} | {'PM2.5':<8} | {'CO2':<8} | {'Reward':<6}")
    print("-" * 80)

    for step in range(test_env.MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = test_env.step(action.item())
        episode_reward += reward

        # Print every hour OR when fan is ON OR when Cooking
        should_print = (step % 60 == 0) or (info['activity'] != "none") or (test_env.fan_speed > 0)

        if should_print:
            print(f"{step:<10} | {info['activity']:<8} | {test_env.fan_speed:.2f}  | "
                  f"{info['voc']:<8.1f} | {info['pm']:<8.1f} | {info['co2']:<8.0f} | {reward:.2f}")

        if trunc: break

    print("-" * 80)
    print(f"Evaluation Finished. Total Cumulative Reward: {episode_reward:.2f}")