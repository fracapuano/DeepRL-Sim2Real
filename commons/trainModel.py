from tqdm import tqdm

import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from commons.utils import FileSaver

def train(agent, env, actorCriticCheck, batch_size, episodes, print_every, file_name='reinforce', print_bool=False, save_to_file_bool=True, info_file_path='./'):
   
    if save_to_file_bool:
        files = [
            f'{file_name}_reward_file.txt',
            f'{file_name}_action_file.txt'
            ]
        fs_reward = FileSaver(file_name=files[0], path=info_file_path)
        fs_action = FileSaver(file_name=files[1], path=info_file_path)

        fs_reward.write_header("EpisodeID,Reward\n")
        fs_action.write_header("EpisodeID,ActionMeasure1, ActionMeasure2, ActionMeasure3\n")
   
    episodes_counter = 0
    for episode in tqdm(range(episodes)):
        episodes_counter += 1
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


            fs_reward.append_content(f"{episodes_counter},{reward}\n")
            fs_action.append_content(f"{episodes_counter},{action[0]},{action[1]},{action[2]}\n")


            if actorCriticCheck and batch_counter == batch_size: 
                agent.update_policy()
                batch_counter = 0
                continue
            
        if not actorCriticCheck: 
            agent.update_policy()
        if print_bool:
            if (episode+1)%print_every == 0:
                print('Training episode:', episode)
                print('Episode return:', train_reward)
