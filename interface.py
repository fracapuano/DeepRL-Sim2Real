### FUNDAMENTAL PACKAGES ###
import torch
import gym
from env.custom_hopper import *

### UTILITIES ###
import argparse
import sys
import os

### COMMONS ###
from commons import trainModel, testModel, saveModel, makeEnv, utils

MODELS_PATH = 'models/'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', default='train', type=str, help='Select to either train an agent or test a trained agent')
    parser.add_argument('--model', default=None, type=str, help='Trained model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device <cpu, cuda>')
    parser.add_argument('--render', default=True, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')
    parser.add_argument('--agent-type', default='reinforce', type=str, help='Agent type to be tested with existing trained model <reinforce - actorCritic - ppo - trpo>')
    parser.add_argument('--domain-type', default='source', type=str, help='Select which env to build: either source or target')
    parser.add_argument('--batch-size', default=10, type=int, help='Batch size!')
    parser.add_argument('--print-every', default=10, type=int, help='number of episodes to pass before printing an output')
    # argument to be removed later, this is just for debugging
    parser.add_argument('--net-type', default='tibNET', type=str, help='String to select the NN used: <tiboniNET, fraNET>')
    return parser.parse_args()

args = parse_args()

def main():

    env = makeEnv.make_environment(args.domain_type)

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    if args.net_type == 'tibNET':
        from policies.tibNET import ReinforcePolicy, ActorCriticPolicy
    elif args.net_type == 'fraNET':
        from policies.fraNET import ReinforcePolicy, ActorCriticPolicy
    else:
        raise Exception("Invalid network architecture selcted. Try <fraNET, tibNET>")

    if args.agent_type.lower() == 'reinforce':

        import agents.agentReinforce

        policy = ReinforcePolicy(observation_space_dim, action_space_dim)
        if args.op == 'test':
            policy.load_state_dict(torch.load(MODELS_PATH + str(args.model)), strict=False)
        agent = agents.agentReinforce.Agent(policy, device=args.device)
        actorCriticCheck=False

    elif args.agent_type.lower() == 'actorcritic':

        import agents.agentActorCritic
        
        batch_size = args.batch_size
        policy = ActorCriticPolicy(observation_space_dim, action_space_dim)
        if args.op == 'test':
            policy.load_state_dict(torch.load(MODELS_PATH + str(args.model)), strict=False)
        agent = agents.agentActorCritic.Agent(policy, args.net_type)
        actorCriticCheck = True

    elif args.agent_type.lower() == 'ppo':

        actorCriticCheck=False
        from stable_baselines3 import PPO

        agent = PPO('MlpPolicy', env, verbose=1)
    
    elif args.agent_type.lower() == 'trpo':

        actorCriticCheck=False
        from sb3_contrib.trpo.trpo import TRPO

        agent = TRPO('MlpPolicy', env, verbose=1)
    
    else:
        raise Exception("Invalid Agent selected. Select one among <reinforce - actorCritic - ppo - trpo>")

    if args.op == 'train' and (args.agent_type != 'ppo' and args.agent_type != 'trpo'):

        trainModel.train(
            agent=agent,
            agent_type=args.agent_type,
            env=env,
            actorCriticCheck=actorCriticCheck,
            batch_size=args.batch_size,
            episodes=args.episodes,
            print_every=args.print_every
            )

        saveModel.save_model(
            agent=agent,
            agent_type=args.agent_type,
            folder_path=MODELS_PATH
            )

    elif args.op == 'train' and (args.agent_type == 'ppo' or args.agent_type == 'trpo'):

        agent.learn(total_timesteps=args.episodes)

        saveModel.save_model(
            agent=agent,
            agent_type=args.agent_type,
            folder_path=MODELS_PATH
            )

    elif args.op == 'test':
        
        testModel.test(
            agent=agent,
            agent_type=args.agent_type,
            env=env,
            episodes=args.episodes,
            render_bool=args.render,
            model_info=MODELS_PATH+args.model
            )

    else:
        raise Exception("Invalid action selected! Try train or test.")

if __name__ == '__main__':
    main()