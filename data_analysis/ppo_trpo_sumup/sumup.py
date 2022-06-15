import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.chdir("ppo_trpo_sumup/")

ALGS = [
    "ppo",
    "trpo"
]

seedless_df_ppo_rewards = pd.read_csv(f"seedless_{ALGS[0]}_rewards.txt")
seedless_df_trpo_rewards = pd.read_csv(f"seedless_{ALGS[1]}_rewards.txt")

seedless_df_ppo_actions = pd.read_csv(f"seedless_{ALGS[0]}_actions.txt")
seedless_df_trpo_actions = pd.read_csv(f"seedless_{ALGS[1]}_actions.txt")

fig1 = plt.figure(figsize=(24,12))

plot = sns.lineplot(x=seedless_df_ppo_rewards.iloc[:, 0], y=seedless_df_ppo_rewards.iloc[:, 1], data=seedless_df_ppo_rewards)
plot = sns.lineplot(x=seedless_df_trpo_rewards.iloc[:, 0], y=seedless_df_trpo_rewards.iloc[:, 1], data=seedless_df_trpo_rewards)

plot.set_xlabel("Episode", fontsize=12)
plot.set_xlim(1, seedless_df_trpo_rewards.iloc[:, 0].max())
plot.set_ylabel("Return", fontsize=12)
plot.set_title("Return per episode for PPO and TRPO algorithms", fontsize=12, fontweight="bold")

plot.lines[0].set_linestyle("--")
plot.lines[0].set_label("PPO")
plot.lines[1].set_linestyle("dotted")
plot.lines[1].set_label("TRPO")
plot.collections[0].set_label(None)
plot.collections[1].set_label(None)
plt.legend()
plt.savefig("return_per_episode_ppovstrpo.svg", format="svg")
plt.close(fig1)


fig2 = plt.figure(figsize=(24,12))

action_plot = sns.lineplot(x=seedless_df_ppo_actions.iloc[:, 0], y=seedless_df_ppo_actions.iloc[:, 1], data=seedless_df_ppo_actions)
action_plot = sns.lineplot(x=seedless_df_trpo_actions.iloc[:, 0], y=seedless_df_trpo_actions.iloc[:, 1], data=seedless_df_trpo_actions)

action_plot.set_xlabel("Episode", fontsize=12)
action_plot.set_xlim(1, seedless_df_trpo_rewards.iloc[:, 0].max())
action_plot.set_ylabel("ActionMeasure", fontsize=12)
action_plot.set_title("ActionMeasure per episode for PPO and TRPO algorithms", fontsize=12, fontweight="bold")

action_plot.lines[0].set_linestyle("--")
action_plot.lines[0].set_label("PPO")
action_plot.lines[1].set_linestyle("dotted")
action_plot.lines[1].set_label("TRPO")
action_plot.collections[0].set_label(None)
action_plot.collections[1].set_label(None)

plt.legend()
plt.savefig("action_measure_per_episode_ppovstrpo.svg", format="svg")
plt.close(fig2)