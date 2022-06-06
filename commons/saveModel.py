import torch

def save_model(agent, agent_type, folder_path='./'):
    if agent_type.lower() == 'reinforce':
        torch.save(agent.policy.state_dict(), folder_path + "reinforce-model.mdl")
    elif agent_type.lower() == 'actorcritic':
        torch.save(agent.policy.state_dict(), folder_path + "actorCritic-model.mdl")
    elif agent_type.lower() == 'ppo':
        agent.save(folder_path + "ppo-model.mdl")
    elif agent_type.lower() == 'trpo':
        agent.save(folder_path + "trpo-model.mdl")
    else:
        raise Exception("Policy type not well specified. Please check input arguments.")