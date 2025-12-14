import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

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

        clean_label = filename.replace("logs/log_", "").replace(".json", "")
        clean_label = clean_label.replace("_LrLow", " lr=1e-5").replace("_LrHigh", " lr=1e-3")
        clean_label = clean_label.replace("_GamLow", " γ=0.90").replace("_GamHigh", " γ=0.999")

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


os.makedirs("plots", exist_ok=True)

# # --- DQN EPSILON VS RANDOM ---
# my_logs = [
#     "logs/log_Eps_Default.json",
#     "logs/log_Eps_LrLow_GamLow.json",
#     "logs/log_Eps_LrLow_GamHigh.json",
#     "logs/log_Eps_LrHigh_GamLow.json",
#     "logs/log_Eps_LrHigh_GamHigh.json",
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
#     y_axis_ticks=[-300000, -5000, -2500, 0, 1000, 2000, 3000, 4000, 5000]
# )
#
# plot_from_logs(
#     my_logs,
#     metric_key="TASVT",
#     title="Epsilon DQN Unsafe VOC Minutes",
#     save_name="plots/Epsilon_DQN_Unsafe.png",
#     use_symlog=False
# )
#
# # --- DQN ENTROPY VS RANDOM ---
# my_logs = [
#     "logs/log_Ent_Default.json",
#     "logs/log_Ent_LrLow_GamLow.json",
#     "logs/log_Ent_LrLow_GamHigh.json",
#     "logs/log_Ent_LrHigh_GamLow.json",
#     "logs/log_Ent_LrHigh_GamHigh.json",
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
#     y_axis_ticks=[-1400000, 0, 1000, 2000, 3000, 4000, 5000]
# )
#
# plot_from_logs(
#     my_logs,
#     metric_key="TASVT",
#     title="Entropy DQN Unsafe VOC Minutes",
#     save_name="plots/Entropy_DQN_Unsafe.png",
#     use_symlog=False
# )

# --- DQN EPSILON VS ENTROPY ---
my_logs = [
    "logs/log_Eps_LrLow_GamHigh.json",
    "logs/log_Eps_LrHigh_GamHigh.json",
    #"logs/log_Ent_Default.json",
    "logs/log_Ent_LrHigh_GamHigh.json",
    "logs/log_Rnd_Default.json"
]

plot_from_logs(
    my_logs,
    metric_key="Reward",
    title="Summary DQN Reward Curve (Log Scale)",
    save_name="plots/Summary_DQN_Reward.png",
    use_symlog=True,
    symlog_scale=6000,
    y_axis_ticks=[1000, 2000, 3000, 4000, 5000, 5500]
)

plot_from_logs(
    my_logs,
    metric_key="TASVT",
    title="Summary DQN Unsafe VOC Minutes",
    save_name="plots/Summary_DQN_Unsafe.png",
    use_symlog=False
)

# # --- DQN Pure Random ---
# my_logs = [
#     "logs/log_Rnd_Default.json"
# ]
#
# plot_from_logs(
#     my_logs,
#     metric_key="Reward",
#     title="Random DQN Reward Curve",
#     save_name="plots/Random_DQN_Reward.png",
#     use_symlog=True,
#     symlog_scale=5000,
#     y_axis_ticks=[0, 1000, 2000, 3000, 4000, 5000]
# )
#
# plot_from_logs(
#     my_logs,
#     metric_key="TASVT",
#     title="Random DQN Unsafe VOC Minutes",
#     save_name="plots/Random_DQN_Unsafe.png",
#     use_symlog=False
# )
#
# # --- DDQN EPSILON VS RANDOM ---
# my_logs = [
#     "logs/log_DDQN_Eps_Default.json",
#     "logs/log_DDQN_Eps_LrLow_GamLow.json",
#     "logs/log_DDQN_Eps_LrLow_GamHigh.json",
#     "logs/log_DDQN_Eps_LrHigh_GamLow.json",
#     "logs/log_DDQN_Eps_LrHigh_GamHigh.json",
#     "logs/log_DDQN_Rnd_Default.json"
# ]
#
# plot_from_logs(
#     my_logs,
#     metric_key="Reward",
#     title="Epsilon DDQN Reward Curve (Log Scale)",
#     save_name="plots/Epsilon_DDQN_Reward.png",
#     use_symlog=True,
#     symlog_scale=5000,
#     y_axis_ticks=[-300000, -5000, -2500, 0, 1000, 2000, 3000, 4000, 5000]
# )
#
# plot_from_logs(
#     my_logs,
#     metric_key="TASVT",
#     title="Epsilon DDQN Unsafe VOC Minutes",
#     save_name="plots/Epsilon_DDQN_Unsafe.png",
#     use_symlog=False
# )
#
# # --- DDQN ENTROPY VS RANDOM ---
# my_logs = [
#     "logs/log_DDQN_Ent_Default.json",
#     "logs/log_DDQN_Ent_LrLow_GamLow.json",
#     "logs/log_DDQN_Ent_LrLow_GamHigh.json",
#     "logs/log_DDQN_Ent_LrHigh_GamLow.json",
#     "logs/log_DDQN_Ent_LrHigh_GamHigh.json",
#     "logs/log_DDQN_Rnd_Default.json"
# ]
#
# plot_from_logs(
#     my_logs,
#     metric_key="Reward",
#     title="Entropy DDQN Reward Curve (Log Scale)",
#     save_name="plots/Entropy_DDQN_Reward.png",
#     use_symlog=True,
#     symlog_scale=6000,
#     y_axis_ticks=[-1400000, 0, 1000, 2000, 3000, 4000, 5000]
# )
#
# plot_from_logs(
#     my_logs,
#     metric_key="TASVT",
#     title="Entropy DDQN Unsafe VOC Minutes",
#     save_name="plots/Entropy_DDQN_Unsafe.png",
#     use_symlog=False
# )

# --- DDQN EPSILON VS ENTROPY ---
my_logs = [
    "logs/log_DDQN_Eps_LrLow_GamHigh.json",
    "logs/log_DDQN_Eps_LrHigh_GamHigh.json",
    "logs/log_DDQN_Ent_Default.json",
    "logs/log_DDQN_Ent_LrHigh_GamHigh.json",
    "logs/log_DDQN_Rnd_Default.json"
]

plot_from_logs(
    my_logs,
    metric_key="Reward",
    title="Summary DDQN Reward Curve (Log Scale)",
    save_name="plots/Summary_DDQN_Reward.png",
    use_symlog=True,
    symlog_scale=6000,
    y_axis_ticks=[1000, 2000, 3000, 4000, 5000, 5500]
)

plot_from_logs(
    my_logs,
    metric_key="TASVT",
    title="Summary DDQN Unsafe VOC Minutes",
    save_name="plots/Summary_DDQN_Unsafe.png",
    use_symlog=False
)

# # --- DDQN Pure Random ---
# my_logs = [
#     "logs/log_DDQN_Rnd_Default.json"
# ]
#
# plot_from_logs(
#     my_logs,
#     metric_key="Reward",
#     title="Random DDQN Reward Curve",
#     save_name="plots/Random_DDQN_Reward.png",
#     use_symlog=True,
#     symlog_scale=5000,
#     y_axis_ticks=[0, 1000, 2000, 3000, 4000, 5000]
# )
#
# plot_from_logs(
#     my_logs,
#     metric_key="TASVT",
#     title="Random DDQN Unsafe VOC Minutes",
#     save_name="plots/Random_DDQN_Unsafe.png",
#     use_symlog=False
# )