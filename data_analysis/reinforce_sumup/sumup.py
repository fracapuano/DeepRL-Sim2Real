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
action1_baseline = sns.lineplot(x=seedless_df_baseline_actions.iloc[:, 4], y=seedless_df_baseline_actions.iloc[:, 1], data=seedless_df_baseline_actions)
plt.savefig("action1_per_timestep_baseline.png")
plt.close(fig2)

fig3 = plt.figure(figsize=(24,12))
action2_baseline = sns.lineplot(x=seedless_df_baseline_actions.iloc[:, 4], y=seedless_df_baseline_actions.iloc[:, 2], data=seedless_df_baseline_actions)
plt.savefig("action2_per_timestep_baseline.png")
plt.close(fig3)

fig4 = plt.figure(figsize=(24,12))
action3_baseline = sns.lineplot(x=seedless_df_baseline_actions.iloc[:, 4], y=seedless_df_baseline_actions.iloc[:, 3], data=seedless_df_baseline_actions)
plt.savefig("action3_per_timestep_baseline.png")
plt.close(fig4)

fig5 = plt.figure(figsize=(24,12))
action1_standard = sns.lineplot(x=seedless_df_standard_actions.iloc[:, 4], y=seedless_df_standard_actions.iloc[:, 1], data=seedless_df_standard_actions)
plt.savefig("action1_per_timestep_standard.png")
plt.close(fig5)

fig6 = plt.figure(figsize=(24,12))
action2_standard = sns.lineplot(x=seedless_df_standard_actions.iloc[:, 4], y=seedless_df_standard_actions.iloc[:, 2], data=seedless_df_standard_actions)
plt.savefig("action2_per_timestep_standard.png")
plt.close(fig6)

fig7 = plt.figure(figsize=(24,12))
action3_standard = sns.lineplot(x=seedless_df_standard_actions.iloc[:, 4], y=seedless_df_standard_actions.iloc[:, 3], data=seedless_df_standard_actions)
plt.savefig("action3_per_timestep_standard.png")
plt.close(fig7)

fig8 = plt.figure(figsize=(24,12))
action1_togo = sns.lineplot(x=seedless_df_togo_actions.iloc[:, 4], y=seedless_df_togo_actions.iloc[:, 1], data=seedless_df_togo_actions)
plt.savefig("action3_per_timestep.png")
plt.close(fig8)

fig9 = plt.figure(figsize=(24,12))
action2_togo = sns.lineplot(x=seedless_df_togo_actions.iloc[:, 4], y=seedless_df_togo_actions.iloc[:, 2], data=seedless_df_togo_actions)
plt.savefig("action3_per_timestep.png")
plt.close(fig9)

fig10 = plt.figure(figsize=(24,12))
action3_togo = sns.lineplot(x=seedless_df_togo_actions.iloc[:, 4], y=seedless_df_togo_actions.iloc[:, 3], data=seedless_df_togo_actions)
plt.savefig("action3_per_timestep.png")
plt.close(fig10)