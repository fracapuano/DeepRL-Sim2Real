import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.chdir("reinforce_sumup/")

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

fig1, ax = plt.subplots(figsize = (8, 6), nrows = 3)

freq = 250

plot = sns.lineplot(x = seedless_df_standard_rewards.iloc[:, 0][::freq], 
                    y = seedless_df_standard_rewards.iloc[:,1][::freq], 
                    data = seedless_df_standard_rewards, ci = "sd",
                    ax = ax[0], linewidth = 0.5)
plot = sns.lineplot(x = seedless_df_togo_rewards.iloc[:, 0][::freq],
                    y = seedless_df_togo_rewards.iloc[:,1][::freq],
                    data = seedless_df_togo_rewards, ci = "sd",
                    ax = ax[1], linewidth = 0.5)
plot = sns.lineplot(x = seedless_df_baseline_rewards.iloc[:, 0][::freq],
                    y = seedless_df_baseline_rewards.iloc[:,1][::freq],
                    data = seedless_df_baseline_rewards, ci = "sd",
                    ax = ax[2], linewidth = 0.5)

ax[0].set_title("Reinforce standard return per episode", fontweight = "bold")
ax[0].set_xlabel("Episode")
ax[0].set_ylabel("Return per episode")
ax[0].grid()

ax[1].set_title("Reinforce to-go return per episode", fontweight = "bold")
ax[1].set_xlabel("Episode")
ax[1].set_ylabel("Return per episode")
ax[1].grid()

ax[2].set_title("Reinforce with baseline return per episode", fontweight = "bold")
ax[2].set_xlabel("Episode")
ax[2].set_ylabel("Return per episode")
ax[2].grid()

fig1.tight_layout()

fig1.savefig("return_per_episode_reinforces.svg")

fig2 = plt.figure(figsize=(24,12))

action_plot = sns.lineplot(x=seedless_df_baseline_actions.iloc[:, 0], y=seedless_df_baseline_actions.iloc[:, 1], data=seedless_df_baseline_actions)
action_plot = sns.lineplot(x=seedless_df_standard_actions.iloc[:, 0], y=seedless_df_standard_actions.iloc[:, 1], data=seedless_df_standard_actions)
action_plot = sns.lineplot(x=seedless_df_togo_actions.iloc[:, 0], y=seedless_df_togo_actions.iloc[:, 1], data=seedless_df_togo_actions)

action_plot.set_xscale("log")
action_plot.set_xlabel("Episode", fontsize=12)
action_plot.set_ylabel("ActionMeasure", fontsize=12)
action_plot.set_title("ActionMeasure per episode for REINFORCE with three different return implementations", fontsize=12, fontweight="bold")

action_plot.lines[0].set_linestyle("--")
action_plot.lines[0].set_label("REINFORCE with Baseline")
action_plot.lines[1].set_linestyle("dotted")
action_plot.lines[1].set_label(" REINFORCE Standard")
action_plot.lines[2].set_linestyle("dashdot")
action_plot.lines[2].set_label("REINFORCE with reward To-Go")
action_plot.collections[0].set_label(None)
action_plot.collections[1].set_label(None)
action_plot.collections[2].set_label(None)

plt.legend()
plt.savefig("action_measure_per_episode_reinforces.svg", format="svg")
plt.close(fig2)
