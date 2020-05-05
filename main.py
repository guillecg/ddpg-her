import datetime
from pathlib import Path

import numpy as np

import torch

from collections import deque

import matplotlib.pyplot as plt

import gym

from agents.agent_td3 import AgentTD3


SEED = 42
SCORE_TARGET = -6.5
SCORE_WINDOW = 100

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

# create folder architecture
PROJECT = '.'
START_TIME = datetime.datetime.now().strftime('%m-%d-%Y_%Hh%Mm')
EXPERIMENT_FOLDER = f'{PROJECT}/experiments/{START_TIME}'
Path(EXPERIMENT_FOLDER).mkdir(parents=True, exist_ok=True)

ENV_NAME = 'FetchReach-v1'


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

    # define the agent
    agent = AgentTD3(
        state_size=state_size + goal_size,
        action_size=action_size,
        hyperparams=dict(),
        device=DEVICE,
        seed=SEED
    )

    # training hyperparameters
    n_episodes = 100  # maximum number of training episodes
    max_t = 1000       # maximum number of timesteps per episode

    scores = []                                 # scores for each episode
    scores_window = deque(maxlen=SCORE_WINDOW)  # last 100 scores
    scores_window_means = []                    # average max scores for each episode

    # training loop
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset()  # reset the environment
        agent.reset()           # initialize agent
        score = 0               # initialize scores

        # Concatenate goal to state
        state = np.concatenate(
            [env_info['observation'], env_info['desired_goal']],
            axis=0
        )

        for t in range(max_t):
            action = agent.select_action(np.array([state]))[0]  # get the action from the agent
            env_info, reward, done, info = env.step(action)     # send the action to the environment

            # Concatenate goal to state
            next_state = np.concatenate(
                [env_info['observation'], env_info['desired_goal']],
                axis=0
            )

            # save experience tuple into replay buffer
            agent.memory.add(data=(
                state,
                action,
                reward,
                next_state,
                done,
                env_info['achieved_goal'],
                env_info['desired_goal']
            ))

            state = next_state  # roll over states to next time step
            score += reward      # update the scores

            # Train each agent
            agent.learn_batch(timestep=t)

            if done:
                break

        # Score is updated for each agent, therefore use mean as an estimate
        score = np.mean(score)

        scores.append(score)
        scores_window.append(score)

        window_score_mean = np.mean(scores_window)  # save mean of window scores
        scores_window_means.append(window_score_mean)

        print(
            '\rEpisode {}\tEpisode total score: {:.2f}\tWindow Score: {:.2f}'
            .format(i_episode, score, window_score_mean),
            end=""
        )

        if i_episode % 100 == 0:
            print(
                '\rEpisode {}\tWindow Score: {:.2f}'
                .format(i_episode, window_score_mean)
            )

        if window_score_mean >= SCORE_TARGET:
            print(
                '\nEnvironment solved in {:d} episodes!\tWindow Score: {:.2f}'
                .format(i_episode, window_score_mean)
            )

            print(f'Saving weights into {EXPERIMENT_FOLDER} folder...')
            torch.save(
                agent.actor_local.state_dict(),
                f'{EXPERIMENT_FOLDER}/weights_actor_episode_{i_episode}.pth'
            )
            torch.save(
                agent.critic_local.state_dict(),
                f'{EXPERIMENT_FOLDER}/weights_critic_episode_{i_episode}.pth'
            )

            break

    # evaluation loop
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset()  # reset the environment
        agent.reset()           # initialize agent
        score = 0               # initialize scores

        # Concatenate goal to state
        state = np.concatenate(
            [env_info['observation'], env_info['desired_goal']],
            axis=0
        )

        for t in range(max_t):
            action = agent.select_action(np.array([state]))[0]  # get the action from the agent
            env_info, reward, done, info = env.step(action)     # send the action to the environment

            # Concatenate goal to state
            next_state = np.concatenate(
                [env_info['observation'], env_info['desired_goal']],
                axis=0
            )

            state = next_state  # roll over states to next time step
            score += reward      # update the scores

            env.render()

            if done:
                break

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores, label='Episode scores')
    plt.plot(np.arange(1, len(scores) + 1), scores_window_means, label='Window mean')
    plt.legend()
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    # save figure to file
    print(f'Saving figure into {EXPERIMENT_FOLDER} folder...')
    fig.savefig(f'{EXPERIMENT_FOLDER}/scores.png')

    # close the environment
    env.close()
