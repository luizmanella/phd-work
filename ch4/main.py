import gymnasium as gym
import numpy as np
import random
from collections import deque
import copy
import threading
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import ot
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from gymnasium.envs.classic_control.cartpole import CartPoleEnv

class CustomCartPoleEnv(CartPoleEnv):
    def __init__(self, pole_length=0.5, **kwargs):
        super().__init__(**kwargs)
        self.length = pole_length
        self.polemass_length = self.masspole * self.length

import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="CustomCartPole-v0",
    entry_point="__main__:CustomCartPoleEnv",
)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, action_dim, bias=False)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.vstack(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.vstack(next_states), np.array(dones, dtype=bool))

    def __len__(self):
        return len(self.buffer)

def FedWB(models, reg):
    all_min_values, all_sum_values = [], []
    for model in models:
        model_mins, model_sums = [], []
        for p in model.parameters():
            m = float(p.data.min())
            model_mins.append(m)
            p.data.add_(m)
            s = float(p.data.sum())
            model_sums.append(s)
            p.data.div_(s)
        all_min_values.append(model_mins)
        all_sum_values.append(model_sums)

        ref_model = models[0]

    # Collect parameter groups by layer (excluding bias terms)
    param_groups = list(zip(*[m.parameters() for m in models]))
    bary_weights = []

    for layer_params in param_groups:
        if layer_params[0].dim() == 1:
            continue  # skip biases if they exist

        # Stack flattened weights from all models
        A = np.stack([p.data.numpy().ravel() for p in layer_params], axis=0)  # (num_models, n)

        # Normalize
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-8
        A /= row_sums

        n = A.shape[1]
        x = np.arange(n, dtype=np.float64).reshape((n, 1))
        M = ot.dist(x, x, metric='sqeuclidean')

        A_T = A.T  # shape: (n, num_models)
        weights = np.ones(A.shape[0]) / A.shape[0]  # uniform weights

        bary = ot.bregman.barycenter(A_T, M, reg, weights=weights)  # shape (n,)
        bary_weights.append(torch.tensor(bary.reshape(layer_params[0].shape), dtype=torch.float32))

    mins_per_layer = list(zip(*all_min_values))
    sums_per_layer = list(zip(*all_sum_values))
    avg_min = [sum(vals)/len(vals) for vals in mins_per_layer]
    avg_sum = [sum(vals)/len(vals) for vals in sums_per_layer]

    adjusted = []
    for i, b in enumerate(bary_weights):
        adjusted.append(b * avg_sum[i] - avg_min[i])

    new_model = copy.deepcopy(ref_model)
    for p, w in zip(new_model.parameters(), adjusted): p.data.copy_(w)
    return new_model

class Agent:
    def __init__(self, agent_id, pole_length,
                 buffer_capacity, batch_size, gamma,
                 lr):
        self.agent_id = agent_id
        self.env = gym.make("CustomCartPole-v0", pole_length=pole_length)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n
        self.policy_net = QNetwork(obs_dim, act_dim, hidden_dim)
        self.target_net = QNetwork(obs_dim, act_dim, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.steps_done = 0
        self.loss_fn = nn.SmoothL1Loss()
        self.episode_durations = []

    def select_action(self, state, eps):
        if random.random() < eps:
            return self.env.action_space.sample()
        with torch.no_grad():
            return self.policy_net(torch.from_numpy(state).float().unsqueeze(0)).argmax().item()

    def train(self, num_episodes, eps_start, eps_end, eps_decay, C_steps, agg_callback):
        for ep in range(1, num_episodes+1):
            print(f"\n=== Agent {self.agent_id} | Local Epoch: {ep} ===")
            # state, done = self.env.reset(), False
            state, _ = self.env.reset(); done = False
            steps = 0
            while not done:
                print(f"Agent {self.agent_id} | Update: still balancing pole...")
                eps = eps_end + (eps_start-eps_end)*np.exp(-1.*self.steps_done/eps_decay)
                action = self.select_action(state, eps)
                # next_state, reward, done, _ = self.env.step(action)
                next_state, reward, terminated, truncated, _ = self.env.step(action); done = terminated or truncated

                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                if len(self.replay_buffer) >= self.batch_size:
                    self._optimize_model()
                if self.steps_done > 0 and self.steps_done % C_steps == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    agg_callback(sync_targets=True)
                self.steps_done += 1
                steps += 1
            self.episode_durations.append(steps)

        print(f'Agent {self.agent_id} | Update: Finished training. Storing current model.')
        torch.save(self.policy_net.state_dict(), f'model_state/{self.agent_id}.state_dict.pth')
        return self.policy_net

    def _optimize_model(self):
        print(f"Agent {self.agent_id} | Update: Optimizing local model.")
        s,a,r,s2,d = self.replay_buffer.sample(self.batch_size)
        s,a,r,s2,d = map(lambda x: torch.from_numpy(x).float() if x.dtype!=bool else torch.from_numpy(x).unsqueeze(1),
                         (s,a,r,s2,d))
        q = self.policy_net(s).gather(1, a.long())
        q_next = self.target_net(s2).max(1)[0].detach().unsqueeze(1)
        target = r.unsqueeze(1) + self.gamma * q_next * (1-d)
        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Server:
    def __init__(self, num_agents, base_length,
                 rounds, episodes_per_agent,
                 eps_start, eps_end, eps_decay,
                 C_steps, hidden_dim,
                 buffer_capacity, batch_size,
                 gamma, lr, reg):
        self.agents = [Agent(i, (i/num_agents)*base_length,
                              buffer_capacity, batch_size,
                              gamma, lr)
                       for i in range(1, num_agents+1)]
        self.rounds = rounds
        self.episodes = episodes_per_agent
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.C_steps = C_steps
        self.lock = threading.Lock()
        self.reg = reg
        self.episode_durations = []
        self.round_avg_durations = []

    def aggregate_and_distribute(self, sync_targets=False):
        with self.lock:
            if sync_targets:
                for ag in self.agents:
                    ag.target_net.load_state_dict(ag.policy_net.state_dict())
            models = [ag.policy_net for ag in self.agents]
            new_model = FedWB(models, self.reg)
            sd = new_model.state_dict()
            for ag in self.agents:
                ag.policy_net.load_state_dict(sd)
                ag.target_net.load_state_dict(sd)

    def train_federated(self):
        def run_agent(agent):
            agent.train(
                self.episodes,
                self.eps_start,
                self.eps_end,
                self.eps_decay,
                self.C_steps,
                self.aggregate_and_distribute
            )
        self.episode_durations = []
        self.round_avg_durations = []
        for r in range(1, self.rounds + 1):
            print(f"\n=== Federated Round {r}/{self.rounds} ===")
            threads = []

            for agent in self.agents:
                t = threading.Thread(target=run_agent, args=(agent,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # Aggregate durations from all agents
            round_averages = []
            for agent in self.agents:
                print('ep: ' + str(agent.episode_durations[-1]))
                round_averages.append(agent.episode_durations[-1])
            self.episode_durations.append(np.mean(round_averages))

            print('Server: Starting FedWB aggregation.')
            self.aggregate_and_distribute(sync_targets=True)
            print("Server: FedWB aggregation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", default=100, type=int)
    parser.add_argument("--base_length", default=0.5, type=float)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--episodes_per_agent", default=10, type=int) 
    parser.add_argument("--eps_start", default=0.9, type=float)
    parser.add_argument("--eps_end", default=0.05, type=float)
    parser.add_argument("--eps_decay", default=1000, type=int)
    parser.add_argument("--C_steps", default=100, type=int) 
    parser.add_argument("--hidden_dim", default=128, type=int) 
    parser.add_argument("--buffer_capacity", default=10000, type=int)
    parser.add_argument("--batch_size", default=256, type=int) 
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--reg", default=1e-1, type=float)


    args = parser.parse_args()

    # expose hidden_dim for Agent QNetwork
    global hidden_dim
    hidden_dim = args.hidden_dim

    server = Server(
        args.num_agents,
        args.base_length,
        args.epochs,
        args.episodes_per_agent,
        args.eps_start,
        args.eps_end,
        args.eps_decay,
        args.C_steps,
        args.hidden_dim,
        args.buffer_capacity,
        args.batch_size,
        args.gamma,
        args.lr,
        args.reg
    )
    server.train_federated()


    with open('results.json', 'w') as f:
        json.dump(server.episode_durations, f)