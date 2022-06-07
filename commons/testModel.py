import numpy as np
from tqdm import tqdm

def test(agent, agent_type, env, episodes, render_bool, model_info='models/'):
    if agent_type.lower() == 'reinforce' or agent_type.lower() == 'actorcritic':
        episode_r = np.zeros(episodes)
        for episode in tqdm(range(episodes)):
            rewards=[]
            done = False
            test_reward = 0
            state = env.reset()

            while (not done):

                action, _ = agent.get_action(state, evaluation=True)
            
                state, reward, done, info = env.step(action.detach().cpu().numpy())
                rewards.append(reward)

                if render_bool:
                    env.render()

                test_reward += reward

            episode_r[episode] = test_reward

    elif agent_type == 'ppo' or agent_type == 'trpo':
        model = agent.load(model_info)
        obs = env.reset()
        episode_r = np.zeros(episodes)
        for episode in range(episodes):
            done=False
            rewards=[]
            while (not done):
                if render_bool:
                    env.render()
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                rewards.append(reward)

                if done:
                    env.reset()
                    
            episode_r[episode] = np.array(rewards).sum()
                
    return episode_r.mean()