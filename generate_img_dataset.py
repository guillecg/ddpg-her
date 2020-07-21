import os

from pathlib import Path

import numpy as np
import gym

import json
import piexif
import piexif.helper

from PIL import Image

import torch

from agents.agent_td3 import AgentTD3

# Needed for rendering in RGB, see:
# https://github.com/openai/mujoco-py/issues/390
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)


SEED = 42
SCORE_TARGET = -6.5
SCORE_WINDOW = 100

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

ENV_NAME = 'FetchReach-v1'


def generate_img_dataset(
        env: gym.Env,
        agent=None,
        n_img: int = 1000,
        data_dir: str = './data/'
) -> None:
    # Initialize environment, agent and variables
    episode = 1
    i_img = 1
    env_info = env.reset()  # reset the environment

    if agent:
        agent.reset()
        data_dir = os.path.join(data_dir, 'agent')

    else:
        data_dir = os.path.join(data_dir, 'random')

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    for i in range(n_img):
        if agent:
            # Concatenation for HER
            state = np.concatenate(
                [env_info['observation'], env_info['desired_goal']],
                axis=0
            )
            action = agent.select_action(np.array([state]))[0]
        else:
            action = np.random.rand(*env.action_space.shape)

        # Get status after performing random step
        env_info, reward, done, info = env.step(action)

        # JSONify the numpy.arrays and floats
        env_info = {key: value.tolist() for key, value in env_info.items()}
        reward = float(reward)
        info = {key: float(value) for key, value in info.items()}

        # Create metadata
        metadata = {
            'obs': env_info,
            'reward': reward,
            'done': done,
            'info': info
        }

        img = env.render(mode='rgb_array')
        img = Image.fromarray(img)

        img_path = os.path.join(data_dir, f'episode_{episode}_img_{i_img}.jpeg')
        img.save(img_path)

        # %% Write out exif data
        # load existing exif data from image
        exif_dict = piexif.load(img_path)
        # insert custom data in usercomment field
        exif_dict['Exif'][piexif.ExifIFD.UserComment] = \
            piexif.helper.UserComment.dump(
                json.dumps(metadata),
                encoding='unicode'
            )
        # insert mutated data (serialised into JSON) into image
        piexif.insert(
            piexif.dump(exif_dict),
            img_path
        )

        i_img += 1

        if done:
            env_info = env.reset()  # reset the environment
            episode += 1
            i_img = 1

            if agent:
                agent.reset()  # reset agent

        if (i + 1) % 100 == 0:
            print(f'[+] {i + 1} images generated.')


if __name__ == '__main__':
    env = gym.make(ENV_NAME, reward_type='sparse')

    # reset the environment
    env_info = env.reset()

    # size of each action
    action_size = env.action_space.shape[0]
    print('Action space:', action_size)

    # examine the state space
    state_size = env.observation_space['observation'].shape[0]
    print('Observation space:', state_size)

    # examine the goal space
    goal_size = len(env_info['achieved_goal'])
    print('Goal space:', goal_size)

    agent = AgentTD3(
        state_size=state_size + goal_size,
        action_size=action_size,
        hyperparams=dict(),
        device=DEVICE,
        seed=SEED
    )

    # Load existing agent
    training_date = '05-05-2020_00h07m'
    experiment_folder = f'experiments/{ENV_NAME}_{training_date}'
    i_episode = 3262

    agent.actor_local.load_state_dict(
        torch.load(f'{experiment_folder}/weights_actor_episode_{i_episode}.pth')
    )
    agent.critic_local.load_state_dict(
        torch.load(f'{experiment_folder}/weights_critic_episode_{i_episode}.pth')
    )

    generate_img_dataset(
        env=env,
        agent=agent,
        n_img=1000
    )

    # # Visualize a random image
    # img_path = 'data/random/episode_1_img_1.jpeg'
    #
    # # %% Read in exif data
    # exif_dict = piexif.load(img_path)
    # # Extract the serialized data
    # user_comment = piexif.helper.UserComment.load(
    #     exif_dict["Exif"][piexif.ExifIFD.UserComment]
    # )
    # # Deserialize
    # d = json.loads(user_comment)
    # print("Read in exif data: %s" % d)
