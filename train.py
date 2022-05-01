import torch
import gym
import argparse

from env.custom_hopper import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--agent-type', default='reinforce', type=str, help='Select the agent that has to perform training reinforce - actorCritic - PPO - TRPO')
    parser.add_argument('--domain-type', default='source', type=str, help='source / target')
    return parser.parse_args()

args = parse_args()

def main():

    if args.domain_type.lower() == 'source':
        env = gym.make('CustomHopper-source-v0')
    elif args.domain_type.lower() == 'target':
        env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    if args.agent_type.lower() == 'reinforce':

        import agentReinforce
        policy = agentReinforce.Policy(observation_space_dim, action_space_dim)
        agent = agentReinforce.Agent(policy, device=args.device)
        actorCriticCheck=False

    elif args.agent_type.lower() == 'actorcritic':

        import agentActorCritic
        actor = agentActorCritic.Actor(observation_space_dim, action_space_dim)
        critic = agentActorCritic.Critic(observation_space_dim)
        agent = agentActorCritic.Agent(actor=actor, critic=critic, device=args.device)

        actorCriticCheck = True

    elif args.agent_type.lower() == 'ppo':
        actorCriticCheck=False
        from stable_baselines.common.policies import MlpPolicy 
        from stable_baselines import PPO1

        agent = PPO1(MlpPolicy, env, verbose=1)
        agent.learn(total_timesteps=args.n_episodes)
        agent.save('ppo-model.mdl')
        return
    
    elif args.agent_type.lower() == 'trpo':
        actorCriticCheck=False
        from stable_baselines.common.policies import MlpPolicy
        from stable_baselines import TRPO

        agent = TRPO(MlpPolicy, env, verbose=1)
        agent.learn(total_timesteps=args.n_episodes)
        agent.save('trpo-model.mdl')
        return

    for episode in range(args.n_episodes):
        done = False
        train_reward = 0
        state = env.reset()

        while not done:

            action, action_probabilities = agent.get_action(state)
            previous_state = state
            
            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(previous_state, state, action_probabilities, reward, done)

            train_reward += reward
            
            if actorCriticCheck:
                agent.update_policy()
                print(episode)
            else:
                continue
            
        if not actorCriticCheck:
            agent.update_policy()
        else:
            pass
        
        if (episode+1)%args.print_every == 0:
            print('Training episode:', episode)
            print('Episode return:', train_reward)

    if actorCriticCheck:
        torch.save(agent.actor.state_dict(), "actorCritic-model.mdl")
    else:
        torch.save(agent.policy.state_dict(), "reinforce-model.mdl")

if __name__ == '__main__':
    main()