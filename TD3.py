import numpy as np
import torch
import copy
from utils import Feedforward, Memory, QFunction


class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate=3e-4):
        super(Critic, self).__init__()
        self.Q1 = QFunction(observation_dim=state_dim, action_dim=action_dim, hidden_sizes=[256, 256],
                            learning_rate=learning_rate)
        self.Q2 = QFunction(observation_dim=state_dim, action_dim=action_dim, hidden_sizes=[256, 256],
                            learning_rate=learning_rate)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate,
                                          eps=0.000001)

    def forward(self, state, action):
        return self.Q1.Q_value(state, action), self.Q2.Q_value(state, action)

    def fit(self, state, action, target_Q):
        self.train()
        self.optimizer.zero_grad()
        current_Q1, current_Q2 = self.forward(state, action)
        loss = self.Q1.loss(current_Q1, target_Q) + self.Q2.loss(current_Q2, target_Q)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Actor(Feedforward):
    def __init__(self, observation_dim, action_dim, max_action, hidden_sizes=[256, 256], learning_rate=3e-4):
        super(Actor, self).__init__(input_size=observation_dim, output_size=action_dim, hidden_sizes=hidden_sizes)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate,
                                          eps=0.000001)
        self.loss = torch.nn.MSELoss()  # -self.critic.Q1(state, self.actor(state)).mean()
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * torch.tanh(super(Actor, self).forward(x))

    def fit(self, critic, state):
        self.train()
        self.optimizer.zero_grad()
        loss = -critic.Q1.Q_value(state, self(state)).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class TD3:
    def __init__(self, state_dim, action_dim, max_action, **config):
        self._config = {
            "discount": 0.99,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_freq": 2,
            "buffer_size": int(1e6),
            "batch_size": 100
        }
        self._config.update(config)

        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)

        self.max_action = max_action

        self.train_iter = 0
        self.buffer = Memory(max_size=self._config["buffer_size"])

    def _update_target_net(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self._config["tau"] * param.data + (1 - self._config["tau"]) * target_param.data)

    def act(self, state):
        return self.actor.predict(state)

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def train(self):
        self.train_iter += 1

        # Sample replay buffer
        data = self.buffer.sample(self._config["batch_size"])
        state = np.stack(data[:, 0])  # s_t
        action = np.stack(data[:, 1]) # a_t for player 1 # TODO
        action = action[:, :int(action.shape[-1]/2)]
        reward = np.stack(data[:, 2])[:, None]  # rew  (batchsize,1)
        next_state = np.stack(data[:, 3])  # s_t+1
        done = np.stack(data[:, 4])[:, None]  # done signal  (batchsize,1)

        state = torch.from_numpy(state).float()
        action = torch.from_numpy(action).float()

        next_state = torch.from_numpy(next_state).float()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self._config["policy_noise"]
            ).clamp(-self._config["noise_clip"], self._config["noise_clip"])


            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1. - done) * self._config["discount"] * target_Q.numpy()

        self.critic.fit(state, action, torch.from_numpy(target_Q).float())
        if self.train_iter % self._config["policy_freq"] == 0:
            self.actor.fit(self.critic, state)
            # Update the frozen target models
            self._update_target_net(self.critic, self.critic_target)
            self._update_target_net(self.actor, self.actor_target)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic.optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor.optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic.optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor.optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
