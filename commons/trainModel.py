from tqdm import tqdm

def train(agent, env, actorCriticCheck, batch_size, episodes, print_every):
    for episode in tqdm(range(episodes)):
        batch_counter = 0
        done = False
        train_reward = 0
        state = env.reset()

        while not done: 
            batch_counter += 1
            action, action_probabilities = agent.get_action(state)
            previous_state = state
            
            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(previous_state, state, action_probabilities, reward, done)

            train_reward += reward

            if actorCriticCheck and batch_counter == batch_size: 
                agent.update_policy()
                batch_counter = 0
                continue
            
        if not actorCriticCheck: 
            agent.update_policy()
        
        #if (episode+1)%print_every == 0:
            #print('Training episode:', episode)
            #print('Episode return:', train_reward)