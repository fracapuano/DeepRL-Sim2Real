import numpy as np

def test(agent, agent_type, env, episodes, model_info='/', render_bool):
    if agent_type.lower() == 'reinforce' or agent_type.lower() == 'actorcritic':
        episode_r = np.zeros(episodes)
        for episode in range(episodes):
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

            gammas = agent.gamma**np.arange(len(rewards))
            episode_return = gammas @ np.array(rewards)
            episode_r[episode] = episode_return

        print(f"Average return: {episode_r.mean()}")

    elif agent_type == 'ppo' or agent_type == 'trpo':
        model = agent.load(model_info)
        obs = env.reset()
        for episode in range(episodes):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()
            if dones:
                env.reset()
                
    return episode_r.mean()