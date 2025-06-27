import gym
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

# ----- DQN Network Definition -----
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ----- Replay Buffer -----
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.vstack(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.vstack(next_states), np.array(dones, dtype=np.uint8))

    def __len__(self):
        return len(self.buffer)

# ----- FedWB Aggregation -----
def FedWB(models, reg):
    all_min_values, all_sum_values = [], []
    for model in models:
        model_mins, model_sums = [], []
        for p in model.parameters():
            m = float(p.data.min()); model_mins.append(m)
            p.data.add_(m)
            s = float(p.data.sum()); model_sums.append(s)
            p.data.div_(s)
        all_min_values.append(model_mins)
        all_sum_values.append(model_sums)

    barycenters = []
    per_layer = list(zip(*[m.parameters() for m in models]))
    for layer_params in per_layer:
        A = np.stack([p.data.cpu().numpy().ravel() for p in layer_params], axis=0)
        A /= A.sum(axis=1, keepdims=True)
        n = A.shape[1]
        x = np.arange(n, dtype=np.float64).reshape((n,1))
        M = ot.dist(x, x)
        b = ot.bregman.sinkhorn_barycenter(A, M, reg, weights=None)
        shape = layer_params[0].data.shape
        barycenters.append(torch.tensor(b.reshape(shape), device=layer_params[0].device))

    mins_per_layer = list(zip(*all_min_values))
    sums_per_layer = list(zip(*all_sum_values))
    avg_min = [sum(vals)/len(vals) for vals in mins_per_layer]
    avg_sum = [sum(vals)/len(vals) for vals in sums_per_layer]

    adjusted = []
    for i, b in enumerate(barycenters):
        adjusted.append(b * avg_sum[i] - avg_min[i])

    new_model = copy.deepcopy(models[0])
    for p, w in zip(new_model.parameters(), adjusted): p.data.copy_(w)
    return new_model

class Agent:
    def __init__(self, agent_id, pole_length, env_name,
                 buffer_capacity, batch_size, gamma,
                 lr, target_update_freq):
        self.agent_id = agent_id
        self.env = gym.make(env_name)
        self.env.env.length = pole_length
        self.env.env.polemass_length = self.env.env.masspole * pole_length
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
        self.target_update_freq = target_update_freq
        self.steps_done = 0
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state, eps):
        if random.random() < eps:
            return self.env.action_space.sample()
        with torch.no_grad():
            return self.policy_net(torch.from_numpy(state).float().unsqueeze(0)).argmax().item()

    def train(self, num_episodes, eps_start, eps_end, eps_decay,
              C_steps, agg_callback):
        for ep in range(1, num_episodes+1):
            state, done = self.env.reset(), False
            while not done:
                eps = eps_end + (eps_start-eps_end)*np.exp(-1.*self.steps_done/eps_decay)
                action = self.select_action(state, eps)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                if len(self.replay_buffer) >= self.batch_size:
                    self._optimize_model()
                if self.steps_done % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                if self.steps_done > 0 and self.steps_done % C_steps == 0:
                    agg_callback(sync_targets=True)
                self.steps_done += 1
        return self.policy_net

    def _optimize_model(self):
        s,a,r,s2,d = self.replay_buffer.sample(self.batch_size)
        s,a,r,s2,d = map(lambda x: torch.from_numpy(x).float() if x.dtype!=np.uint8 else torch.from_numpy(x).unsqueeze(1),
                         (s,a,r,s2,d))
        q = self.policy_net(s).gather(1, a.long())
        q_next = self.target_net(s2).max(1)[0].detach().unsqueeze(1)
        target = r.unsqueeze(1) + self.gamma * q_next * (1-d)
        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

class Server:
    def __init__(self, num_agents, base_length,
                 rounds, episodes_per_agent,
                 eps_start, eps_end, eps_decay,
                 C_steps, hidden_dim,
                 buffer_capacity, batch_size,
                 gamma, lr, target_update_freq, reg):
        self.agents = [Agent(i, (i/num_agents)*base_length, env_name,
                              buffer_capacity, batch_size,
                              gamma, lr, target_update_freq)
                       for i in range(1, num_agents+1)]
        self.rounds = rounds
        self.episodes = episodes_per_agent
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.C_steps = C_steps
        self.lock = threading.Lock()
        self.reg = reg

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
            print("Server: FedWB aggregation complete.")

    def train_federated(self):
        for r in range(1, self.rounds+1):
            print(f"\n=== Federated Round {r}/{self.rounds} ===")
            threads = []
            for agent in self.agents:
                t = threading.Thread(
                    target=agent.train,
                    args=(self.episodes, self.eps_start, self.eps_end,
                          self.eps_decay, self.C_steps,
                          self.aggregate_and_distribute)
                )
                t.start(); threads.append(t)
            for t in threads: t.join()
            self.aggregate_and_distribute(sync_targets=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, required=True)
    parser.add_argument("--base_length", type=float, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--episodes_per_agent", type=int, required=True)
    parser.add_argument("--eps_start", type=float, required=True)
    parser.add_argument("--eps_end", type=float, required=True)
    parser.add_argument("--eps_decay", type=int, required=True)
    parser.add_argument("--C_steps", type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, required=True)
    parser.add_argument("--buffer_capacity", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--target_update_freq", type=int, required=True)
    parser.add_argument("--reg", type=float, required=True)
    parser.add_argument("--env_name", type=str, required=True)
    args = parser.parse_args()

    # expose hidden_dim for Agent QNetwork
    global hidden_dim, env_name
    hidden_dim = args.hidden_dim
    env_name = args.env_name

    server = Server(
        args.num_agents,
        args.base_length,
        args.rounds,
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
        args.target_update_freq,
        args.reg
    )
    server.train_federated()
