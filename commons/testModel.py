import numpy as np
from tqdm import tqdm

def test(agent,
agent_type,
env,
episodes,
render_bool,
model_info='models/'):
    """
    This function performs the testing of the agent in a specific environment for a certain number of episodes.
    agent: the agent object employed in the task.
    agent_type: string used to discriminate between agents' different implementations.
    env: specific environment object retrived by makeEnvironment.
    episodes: number of episodes to test the agent in the environment.
    render_bool: simple boolean that indicates wheather the mujoco wrapper has to be rendered or not.
    model_info: path to load the file .mdl used by the agent.
    """
    episode_r = np.zeros(episodes)
    if agent_type.lower() == 'reinforce' or agent_type.lower() == 'actorcritic':
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
                
    return episode_r.mean(), episode_r.std()