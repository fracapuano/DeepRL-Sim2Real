import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ALGS = [
    "reinforce_baseline",
    "reinforce_standard",
    "reinforce_togo"
] 

seedless_df_baseline_rewards = pd.read_csv(f"seedless_{ALGS[0]}_rewards.txt")
seedless_df_standard_rewards = pd.read_csv(f"seedless_{ALGS[1]}_rewards.txt")
seedless_df_togo_rewards = pd.read_csv(f"seedless_{ALGS[2]}_rewards.txt")

seedless_df_baseline_actions = pd.read_csv(f"seedless_{ALGS[0]}_actions.txt")
seedless_df_standard_actions = pd.read_csv(f"seedless_{ALGS[1]}_actions.txt")
seedless_df_togo_actions = pd.read_csv(f"seedless_{ALGS[2]}_actions.txt")

fig1 = plt.figure(figsize=(24,12))

plot = sns.lineplot(x=seedless_df_baseline_rewards.iloc[:, 0], y=seedless_df_baseline_rewards.iloc[:, 1], data=seedless_df_baseline_rewards)
plot = sns.lineplot(x=seedless_df_standard_rewards.iloc[:, 0], y=seedless_df_standard_rewards.iloc[:, 1], data=seedless_df_standard_rewards)
plot = sns.lineplot(x=seedless_df_togo_rewards.iloc[:, 0], y=seedless_df_togo_rewards.iloc[:, 1], data=seedless_df_togo_rewards)

plot.set_xlabel("Episode")
plot.set_ylabel("Episode Return")
plot.set(title=f"Return per episode")
plot.lines[0].set_linestyle("--")
plot.lines[0].set_label("Baseline")
plot.lines[1].set_linestyle("dotted")
plot.lines[1].set_label("Standard")
plot.lines[2].set_linestyle("dashdot")
plot.lines[2].set_label("To-Go")
plot.collections[0].set_label(None)
plot.collections[1].set_label(None)
plot.collections[2].set_label(None)
# plt.legend(title="reinforces", labels=[alg for alg in ALGS])
plt.legend()
plt.savefig("reward_per_episode.png")
plt.close(fig1)

fig2 = plt.figure(figsize=(24,12))

action_plot = sns.lineplot(x=seedless_df_baseline_actions.iloc[:, 0], y=seedless_df_baseline_actions.iloc[:, 1], data=seedless_df_baseline_actions)
action_plot = sns.lineplot(x=seedless_df_standard_actions.iloc[:, 0], y=seedless_df_standard_actions.iloc[:, 1], data=seedless_df_standard_actions)
action_plot = sns.lineplot(x=seedless_df_togo_actions.iloc[:, 0], y=seedless_df_togo_actions.iloc[:, 1], data=seedless_df_togo_actions)

action_plot.set_xlabel("Episode")
action_plot.set_ylabel("ActionMeasure")
action_plot.set(title="ActionMeasure per episode")

action_plot.lines[0].set_linestyle("--")
action_plot.lines[0].set_label("Baseline")
action_plot.lines[1].set_linestyle("dotted")
action_plot.lines[1].set_label("Standard")
action_plot.lines[2].set_linestyle("dashdot")
action_plot.lines[2].set_label("To-Go")
action_plot.collections[0].set_label(None)
action_plot.collections[1].set_label(None)
action_plot.collections[2].set_label(None)

plt.legend()
plt.savefig("action_measure_per_episode.png")
plt.close(fig2)
