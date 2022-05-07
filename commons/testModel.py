def test(agent, agent_type, env, episodes, model_info, render_bool):
    if agent_type.lower() == 'reinforce' or agent_type.lower() == 'actorcritic':
        for episode in range(episodes):
            done = False
            test_reward = 0
            state = env.reset()

            while (not done):

                action, _ = agent.get_action(state, evaluation=True)
            
                state, reward, done, info = env.step(action.detach().cpu().numpy())

                if render_bool:
                    env.render()

                test_reward += reward

        print(f"Episode: {episode} | Return: {test_reward}")

    elif agent_type == 'ppo' or agent_type == 'trpo':
        model = agent.load(model_info)
        obs = env.reset()
        for episode in range(episodes):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()
            if dones:
                env.reset()