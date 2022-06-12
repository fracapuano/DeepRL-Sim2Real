import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

plot.set_xlabel("Timestep")
plot.set_ylabel("Reward")
plot.set(title=f"Reward per timesteps")
plot.lines[0].set_linestyle("--")
plot.lines[0].set_label("ppo")
plot.lines[1].set_linestyle("dotted")
plot.lines[1].set_label("trpo")
plot.collections[0].set_label(None)
plot.collections[1].set_label(None)
plt.legend()
plt.savefig("reward_per_timestep.png")
plt.close(fig1)

fig2 = plt.figure(figsize=(24,12))
action1_ppo = sns.lineplot(x=seedless_df_ppo_actions.iloc[:, 4], y=seedless_df_ppo_actions.iloc[:, 1], data=seedless_df_ppo_actions)
plt.savefig("action1_per_timestep_ppo.png")
plt.close(fig2)

fig3 = plt.figure(figsize=(24,12))
action2_ppo = sns.lineplot(x=seedless_df_ppo_actions.iloc[:, 4], y=seedless_df_ppo_actions.iloc[:, 2], data=seedless_df_ppo_actions)
plt.savefig("action2_per_timestep_ppo.png")
plt.close(fig3)

fig4 = plt.figure(figsize=(24,12))
action3_ppo = sns.lineplot(x=seedless_df_ppo_actions.iloc[:, 4], y=seedless_df_ppo_actions.iloc[:, 3], data=seedless_df_ppo_actions)
plt.savefig("action3_per_timestep_ppo.png")
plt.close(fig4)

fig5 = plt.figure(figsize=(24,12))
action1_trpo = sns.lineplot(x=seedless_df_trpo_actions.iloc[:, 4], y=seedless_df_trpo_actions.iloc[:, 1], data=seedless_df_trpo_actions)
plt.savefig("action1_per_timestep_trpo.png")
plt.close(fig5)

fig6 = plt.figure(figsize=(24,12))
action2_trpo = sns.lineplot(x=seedless_df_trpo_actions.iloc[:, 4], y=seedless_df_trpo_actions.iloc[:, 2], data=seedless_df_trpo_actions)
plt.savefig("action2_per_timestep_trpo.png")
plt.close(fig6)

fig7 = plt.figure(figsize=(24,12))
action3_trpo = sns.lineplot(x=seedless_df_trpo_actions.iloc[:, 4], y=seedless_df_trpo_actions.iloc[:, 3], data=seedless_df_trpo_actions)
plt.savefig("action3_per_timestep_trpo.png")
plt.close(fig7)