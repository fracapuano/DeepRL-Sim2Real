from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug. 

    This class is used so to store informations on the training process. 

    """
    def __init__(self, path = ".", verbose=0, actionfile_name = "training_actions.txt", rewardfile_name = "training_rewards.txt"):
        """
        :path: (str) Path where to write the files containing the elements obtained using the _on_step method
        """
        super(CustomCallback, self).__init__(verbose)
        self.path = path
        self.episode_setup()
        self.episode_counter = 0

        self.actionfile_name = actionfile_name
        self.rewardfile_name = rewardfile_name

        with open(self.path + "/" + self.actionfile_name, "w") as action_file:
            action_file.write("episodeID,action_measure\n")
        
        with open(self.path + "/" + self.rewardfile_name, "w") as reward_file:
            reward_file.write("episodeID,episode_reward,number_of_steps\n")
    
    def init_actions(self):
        self.actions = np.array([])
    def init_rewards(self): 
        self.rewards = np.array([])
    def init_number_of_steps(self): 
        self.nos = 0

    def episode_setup(self):
        self.init_actions()
        self.init_rewards()
        self.init_number_of_steps()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        report = self.locals
        done = report["dones"].item()

        action_t = report["actions"]
        reward_t = report["rewards"]

        if not done: 
            self.actions = np.append(self.actions, action_t)
            self.rewards = np.append(self.rewards, reward_t)
            self.nos += 1

        else:
            episode_reward = self.rewards.sum()
            episode_action_derivative = np.diff(self.actions.reshape(3, -1)) / self.nos
            episode_action_sum_of_moduli = np.abs(episode_action_derivative).mean()
            number_of_steps = self.nos

            with open(self.path + "/" + self.rewardfile_name, "a") as reward_file: 
                reward_file.write(f"{self.episode_counter},{episode_reward},{number_of_steps}\n")
            
            with open(self.path + "/" + self.actionfile_name, "a") as action_file: 
                action_file.write(f"{self.episode_counter},{episode_action_sum_of_moduli}\n")
            
            # reinitializing the actions and rewards arrays
            self.episode_setup() 
            self.episode_counter += 1

        return True