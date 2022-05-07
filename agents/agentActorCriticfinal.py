import torch

class Agent(object):
    def __init__(self, policy, net_type,device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)

        if net_type == 'tibNET':
            self.policy = policy
        elif net_type == 'fraNET':
            self.policy.actor_network()
            self.policy.critic_network()
        else:
            raise Exception("Invalid policy architecture selected.")

        self.policy.init_weights()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

        self.I = 1
        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy(self):
        """
        This function updates the current parameters of the policy leveraging the 
        experience collected following that policy (i.e. with that specific set of 
        parameters)
        """
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        numberOfSteps = len(states)
        advantages = []
        current_state_values = []

        for timestep in range(numberOfSteps):
            _, current_value = self.policy(states[timestep]) 
            _, onestep_value = self.policy(next_states[timestep])
            current_state_values.append(current_value)
    
            advantages.append(rewards[timestep] + self.gamma * onestep_value - current_value)

        advantages = torch.stack(advantages, dim = 0)

        actor_loss = []
        critic_loss = []
            
        for log_prob, advantage in zip(action_log_probs, advantages):
            #actor_loss.append(log_prob * self.I * advantage)
            actor_loss.append(log_prob * advantage)

        actor_loss = torch.tensor(actor_loss).mean()

        for value_state, advantage in zip(current_state_values, advantages):
            #critic_loss.append(value_state * self.I * advantage)
            critic_loss.append(value_state * advantage)

        critic_loss = torch.stack(critic_loss).mean()

        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        self.I = self.gamma * self.I

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist, _ = self.policy.forward(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) =
            # log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2]))]

            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)            