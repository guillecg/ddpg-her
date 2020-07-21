import numpy as np

import torch
import torch.nn.functional as F

from models.fc import ActorFC, CriticFC
from models.utils import HERReplayBuffer, OUNoise


class AgentTD3(object):
    """Agent class that handles the training of the networks and provides
    outputs as actions

    Keywords:
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        hyperparams (dict): hyperparameters for training the agent
        device (device): cuda or cpu to process tensors
        seed (int): random seed

    """

    def __init__(self, action_size, state_size, hyperparams, device, seed=42):
        self.seed = seed
        self.device = device

        # Environment parameters definition
        self.action_size = action_size
        self.state_size = state_size

        # Networks parameters
        self.actor_fc1_units = 64
        self.actor_fc2_units = 32

        self.critic_fc1_units = 64
        self.critic_fc2_units = 32

        # Hyperparameters definition
        self.batch_size = 128        # minibatch size
        self.buffer_size = int(1e6)  # replay buffer size
        self.learn_every = 20        # how often to update the actor
        self.learn_iterations = 20   # number of update iterations
        self.gamma = 0.995           # discount factor
        self.tau = 0.001             # for soft update of target parameters
        self.actor_lr = 1e-4         # actor learning rate
        self.critic_lr = 1e-3        # critic learning rate

        # Update defined hyperparameters
        self.__dict__.update(hyperparams)

        # Initialize actor networks
        self.actor_local = ActorFC(
            state_size=self.state_size,
            action_size=self.action_size,
            fc1_units=self.actor_fc1_units,
            fc2_units=self.actor_fc2_units,
            seed=self.seed
        ).to(self.device)

        self.actor_target = ActorFC(
            state_size=self.state_size,
            action_size=self.action_size,
            fc1_units=self.actor_fc1_units,
            fc2_units=self.actor_fc2_units,
            seed=self.seed
        ).to(self.device)

        # Initialize critic networks
        self.critic_local = CriticFC(
            state_size=self.state_size,
            action_size=self.action_size,
            fc1_units=self.actor_fc1_units,
            fc2_units=self.actor_fc2_units,
            seed=self.seed
        ).to(self.device)

        self.critic_target = CriticFC(
            state_size=self.state_size,
            action_size=self.action_size,
            fc1_units=self.actor_fc1_units,
            fc2_units=self.actor_fc2_units,
            seed=self.seed
        ).to(self.device)

        # Initialize targets weights from local networks
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.critic_target.load_state_dict(self.critic_local.state_dict())

        # Define optimizer for local networks
        self.actor_optimizer = torch.optim.Adam(
            self.actor_local.parameters(),
            lr=self.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_local.parameters(),
            lr=self.critic_lr
        )

        # Noise process
        self.noise = OUNoise(
            size=self.action_size,
            seed=self.seed
        )

        # Replay memory
        self.memory = HERReplayBuffer(
            buffer_size=self.buffer_size,
            device=self.device,
            seed=self.seed
        )

    def select_action(self, state, add_noise=True, agent_size=1):
        """Returns actions for given state as per current policy.

        NOTE: set add_noise=False when not training.

        """
        state = torch.from_numpy(state).float().to(self.device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            for i in range(agent_size):
                action[i] += self.noise.sample()

        # Clip actions to avoid interference while learning
        action = np.clip(action, -1, 1)

        return action

    def reset(self):
        self.noise.reset()

    def learn_batch(self, timestep, gamma=None, tau=None):
        gamma = gamma or self.gamma
        tau = tau or self.tau

        if (timestep + 1) % self.learn_every == 0:
            if len(self.memory) > self.batch_size:
                for _ in range(self.learn_iterations):
                    experiences = self.memory.sample(self.batch_size)
                    self.learn(experiences=experiences, gamma=gamma, tau=tau)

    def learn(self, experiences, gamma, tau):
        """Update policy and value parameters using given batch of sarsd tuples.

        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))

        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Keywords:
            experiences (Tuple[torch.Tensor]): ((s, a, r, s', done))
            gamma (float): discount factor

        """
        # Sample replay buffer
        states, actions, rewards, next_states, dones = experiences

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q1_targets, Q2_targets = self.critic_target(next_states, actions_next)

        # For TD3, get the minimum of both values returned by the twin networks
        Q_targets_next = torch.min(Q1_targets, Q2_targets)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # --------------------------- update critic -------------------------- #
        # Compute critic loss
        Q1_expected, Q2_expected = self.critic_local(states, actions)
        critic_loss = \
            F.mse_loss(Q1_expected, Q_targets) + \
            F.mse_loss(Q2_expected, Q_targets)

        # Minimize critic loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # --------------------------- update actor --------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local.Q1(states, actions_pred).mean()

        # Minimize actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks --------------------- #
        # Update the frozen critic
        self.soft_update(
            local_model=self.critic_local,
            target_model=self.critic_target,
            tau=tau
        )

        # Update the frozen actor
        self.soft_update(
            local_model=self.actor_local,
            target_model=self.actor_target,
            tau=tau
        )

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Keywords:
            local_model (PyTorch model): from which weights will be copied
            target_model (PyTorch model): to which weights will be copied
            tau (float): interpolation parameter

        """
        iterable = zip(target_model.parameters(), local_model.parameters())

        for target_param, local_param in iterable:
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
