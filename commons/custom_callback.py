from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CustomCallback(BaseCallback):

    def __init__(self, reward_file, action_file, verbose=0):
        super(CustomCallback, self).__init__(verbose)

        self.reward_file = reward_file
        self.action_file = action_file
        self.nos = 0
        self.episodes_counter = 0
        self.actions= np.array([])

        with open(f"{self.reward_file}", "w") as r_file:
            r_file.write("EpisodeID,Reward,Timestep\n")

        with open(f"{self.action_file}", "w") as a_file:
            a_file.write("EpisodeID,ActionMeasure\n")

    def append_to_file(self, file, content):
        with open(f"{file}", "a") as cb_file:
            cb_file.write(content)

    def _on_step(self) -> bool:

        self.nos += 1
        report = self.locals

        action_t = report['actions']
        reward_t = report['rewards']
        done = report['dones'].item()

        if not done:
            self.actions = np.append(self.actions, action_t)
        else:
            self.episodes_counter += 1
            action_derivative = np.diff(self.actions.reshape(3, -1))
            action_measure = np.abs(action_derivative).max(axis = 0) - np.abs(action_derivative).min(axis = 0)
            action_measure = action_measure.max() - action_measure.min()

            self.append_to_file(self.reward_file, f"{self.episodes_counter},{reward_t.item()},{self.nos}\n")
            self.append_to_file(self.action_file, f"{self.episodes_counter},{action_measure}\n")