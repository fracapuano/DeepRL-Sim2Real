import numpy as np

class Rewards(): 
    def __init__(self): 
        self.total_reward = None

    def discount_rewards(self, r, gamma):
        """
        This function returns the total discounted return (G) considering the rewards contained in "r" with discount factor "gamma". 
        Paramethers: 
            r: np.array of shape (len_episode, )
            gamma: scalar
        Return: 
            G: scalar
        """
        # casting r to a row vector
        len_ep = r.reshape(-1,).shape[0]

        exponents = np.arange(0, len_ep)
        discounted_gamma = gamma ** exponents
        print(discounted_gamma.shape)

        # total discounted return
        
        G = r @ discounted_gamma

        self.total_reward = G
        return G
    
    def discount_rewards_truncated(self, r, gamma):
        """
        This function overwrites the previously defined function so that all the rewards posterior to index: 
            gamma*100 + 1
        Example: when gamma = 0.99 we tipically are not interested in all the rewards coming after the 100th step
        Paramethers: 
            r: np.array of shape (len_episode, )
            gamma: scalar
        Return: 
            G: scalar
        """
        truncation_index = int(gamma * 100) + 1
        # truncating r
        r = r[:truncation_index]

        # casting r to a row vector
        len_ep = r.reshape(-1,).shape[0]

        exponents = np.arange(0, len_ep)
        discounted_gamma = gamma ** exponents

        # total discounted return
        G = r @ discounted_gamma
        
        self.total_reward = G
        return G

    def get_G(self): 
        return self.total_reward

class Warhol():
    def __init__(self, figure_path="./"):
        self.figure_path = figure_path

class FileSaver():
    def __init__(self, file_name, path='./'):
        self.path = path
        self.file_name = file_name

    def write_header(self, header):
         with open(f"{self.file_name}", "w") as wf:
            wf.write(header)


    def append_content(self, content):
        with open(f"{self.file_name}", "a") as wf:
            wf.write(content)

    def write_content(self, content):
        with open(f"{self.file_name}", "w") as wf:
            wf.write(content)
