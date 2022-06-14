import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.chdir("a2c_reinforce_sumup/")

ALGS = [
    "actorcritic",
    "reinforce"
]

seedless_df_a2c_rewards = pd.read_csv(f"seedless_{ALGS[0]}_rewards.txt")
seedless_df_reinforce_rewards = pd.read_csv(f"seedless_{ALGS[1]}_rewards.txt")

seedless_df_a2c_actions = pd.read_csv(f"seedless_{ALGS[0]}_actions.txt")
seedless_df_reinforce_actions = pd.read_csv(f"seedless_{ALGS[1]}_actions.txt")

fig1 = plt.figure(figsize=(24,12))

plot = sns.lineplot(x=seedless_df_a2c_rewards.iloc[:, 0], y=seedless_df_a2c_rewards.iloc[:, 1], data=seedless_df_a2c_rewards)
plot = sns.lineplot(x=seedless_df_reinforce_rewards.iloc[:, 0], y=seedless_df_reinforce_rewards.iloc[:, 1], data=seedless_df_reinforce_rewards)

plot.set_xlabel("Episode", fontsize=12)
plot.set_ylabel("Return", fontsize=12)
plot.set_title("Return per episode for ActorCritic and REINFORCE with baseline", fontsize=12, fontweight="bold")
plot.lines[0].set_linestyle("--")
plot.lines[0].set_label("ActorCritic")
plot.lines[1].set_linestyle("dotted")
plot.lines[1].set_label("REINFORCE with Baseline")
plot.collections[0].set_label(None)
plot.collections[1].set_label(None)
plt.legend()
plt.savefig("return_per_episode_a2cvsreinforce.svg", format="svg")
plt.close(fig1)


fig2 = plt.figure(figsize=(24,12))

action_plot = sns.lineplot(x=seedless_df_a2c_actions.iloc[:, 0], y=seedless_df_a2c_actions.iloc[:, 1], data=seedless_df_a2c_actions)
action_plot = sns.lineplot(x=seedless_df_reinforce_actions.iloc[:, 0], y=seedless_df_reinforce_actions.iloc[:, 1], data=seedless_df_reinforce_actions)

action_plot.set_xlabel("Episode", fontsize=12)
action_plot.set_ylabel("ActionMeasure", fontsize=12)
action_plot.set_title("ActionMeasure per episode for ActorCritic and REINFORCE with Baseline", fontsize=12, fontweight="bold")

action_plot.lines[0].set_linestyle("--")
action_plot.lines[0].set_label("ActorCritic")
action_plot.lines[1].set_linestyle("dotted")
action_plot.lines[1].set_label("REINFORCE with Baseline")
action_plot.collections[0].set_label(None)
action_plot.collections[1].set_label(None)

plt.legend()
plt.savefig("action_measure_per_episode_a2cvsreinforce.svg", format="svg")
plt.close(fig2)