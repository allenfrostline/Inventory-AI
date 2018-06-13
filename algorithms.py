import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F


CONSISTENT_ERROR = True
RUNNING_WINDOW = 100


class Algo:

    def __init__(self):
        np.random.seed(1)
        torch.manual_seed(1)
        self.error_his = []
        self.score_his = []
        self.log_status = 'w+'

    def plot_his(self):
        error_his = running_mean(self.error_his)
        score_his = running_mean(self.score_his)
        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(111)
        c1 = 'red'
        ax1.plot(range(RUNNING_WINDOW, self.n_episodes + 1), error_his, color=c1)
        ax1.set_xlabel('episode')
        ax1.set_ylabel('error', color=c1)
        ax1.tick_params(axis='y', labelcolor=c1)
        ax2 = ax1.twinx()
        c2 = 'deepskyblue'
        ax2.plot(range(RUNNING_WINDOW, self.n_episodes + 1), score_his, color=c2)
        ax2.set_ylabel('score', color=c2)
        ax2.tick_params(axis='y', labelcolor=c2)
        plt.tight_layout()
        plt.savefig('logs/{}L{}U_{}.jpg'.format(
            self.n_lines,
            int(self.utilization * 10),
            self.name
        ), bbox_inches='tight')

    def plot_policy(self):  # only for 2d systems
        S = sorted(set(self.state_space[:, 0]))
        df = pd.DataFrame(columns=S, index=S)
        for state in self.state_space:
            if self.name == 'sS':
                self.running = False
            action = self.choose_action(state)
            if self.name == 'NAF':
                self.agent.target_i, self.agent.target_q = action
                self.agent.running = False
                action = self.agent.choose_action(state)
            else:
                action = int(action)
            df.loc[state[0], state[1]] = action
        df.to_csv('results/policy_{}.csv'.format(self.name))

    def log(self, s):
        print(s)
        with open('logs/{}L{}U_{}.log'.format(
            self.n_lines,
            int(self.utilization * 10),
            self.name
        ), self.log_status) as f:
            f.write(s + '\n')
        self.log_status = 'a'

    def report(self):
        self.log('---------')
        self.log('score mean last 100 = {:.2f}'.format(np.mean(self.score_his[-100:])))
        self.log('score std last 100 = {:.2f}'.format(np.std(self.score_his[-100:])))
        if self.plot:
            self.plot_his()
            if self.n_lines == 2:
                self.plot_policy()


class sS(Algo):

    def __init__(self, target_q, target_i, n_lines, utilization, state_space, n_episodes=1000, plot=False, **arg):
        super(sS, self).__init__()
        self.target_q = target_q
        self.target_i = target_i
        self.n_lines = n_lines
        self.utilization = utilization
        self.running = False
        self.n_episodes = n_episodes
        self.plot = plot
        self.state_space = state_space
        self.name = 'sS'

    def possum(self, x):
        return (x * (x > 0)).sum()

    def negsum(self, x):
        return (x * (x < 0)).sum()

    def choose_action(self, observation):
        if self.negsum(observation) < -self.target_q:
            self.running = True
        elif self.possum(observation) > self.target_i:
            self.running = False
        return np.argmin(observation) + 1 if self.running else 0

    def record(self, score):
        error = calc_error(self.score_his)
        self.error_his.append(error)
        self.score_his.append(score)


class QLearning(Algo):
    def __init__(self, action_space, state_space, learning_rate=0.01, gamma=0.9,
                 epsilon=0.9, epsilon_min=0, epsilon_max=1, epsilon_deflator=None,
                 utilization=None, n_episodes=1000, plot=False, **args):
        super(QLearning, self).__init__()
        self.action_space = action_space
        self.state_space = state_space
        self.n_actions = action_space.shape[0]
        self.n_states = state_space.shape[0]
        self.lr = learning_rate
        self.gamma = gamma
        self.q_table = np.zeros((self.n_states, self.n_actions))
        self.errors = []
        self.n_lines = state_space.shape[1]
        self.utilization = utilization
        self.n_episodes = n_episodes
        self.plot = plot
        self.name = 'QLearning'

        if epsilon_deflator:
            self.epsilon_deflator = epsilon_deflator
            self.epsilon_max = epsilon_max
            self.epsilon_min = epsilon_min
            self.epsilon = epsilon_max
        else:
            self.epsilon = epsilon

    def choose_action(self, observation):
        if np.random.uniform() > self.epsilon:
            obs_idx = np.where(np.all(self.state_space == observation, axis=1))[0][0]
            q_values = self.q_table[obs_idx, :]
            max_q = q_values.max()
            action = np.random.choice(np.where(q_values == max_q)[0])
        else:
            action = self.action_space[np.random.choice(self.n_actions)]
        return action

    def learn(self, s, a, r, s_):
        s_idx = np.where(np.all(self.state_space == s, axis=1))[0][0]
        a_idx = np.where(np.all(self.action_space == a, axis=1))[0][0]
        s__idx = np.where(np.all(self.state_space == s_, axis=1))[0][0]
        q_eval = self.q_table[s_idx, a_idx]
        q_target = r + self.gamma * self.q_table[s__idx, :].max()
        dq = q_target - q_eval
        self.errors.append(dq**2)
        self.q_table[s_idx, a_idx] += self.lr * dq

    def record(self, score):
        if hasattr(self, 'epsilon_deflator'):
            self.epsilon = max(self.epsilon * self.epsilon_deflator, self.epsilon_min)
        if not CONSISTENT_ERROR:
            error = np.mean(self.errors) if len(self.errors) else np.nan
        else:
            error = calc_error(self.score_his)
        self.error_his.append(error)
        self.score_his.append(score)
        self.errors = []


class Net(nn.Module):
    def __init__(self, n_i, n_h, n_o):
        super(Net, self).__init__()
        self.i = nn.Linear(n_i, n_h)
        self.i.weight.data.normal_(0, 0.1)
        self.o = nn.Linear(n_h, n_o)
        self.o.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.i(x)
        x = F.relu(x)
        x = F.relu(x)
        x = self.o(x)
        return x


class DQN(Algo):
    def __init__(self, action_space, state_space, hidden_neurons, learning_rate=0.01,
                 gamma=0.9, epsilon=0.9, epsilon_min=0, epsilon_max=1, epsilon_deflator=None,
                 replace_target_iter=300, memory_size=500, batch_size=32, utilization=None,
                 n_episodes=1000, plot=False, **args):
        super(DQN, self).__init__()
        self.action_space = action_space
        self.state_space = state_space
        self.n_actions = action_space.shape[0]
        self.n_lines = state_space.shape[1]
        self.eval_net = Net(self.n_lines, hidden_neurons, self.n_actions)
        self.target_net = Net(self.n_lines, hidden_neurons, self.n_actions)
        self.gamma = gamma
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_size, state_space.shape[1] * 2 + 2))
        self.optimizer = Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        self.errors = []
        self.utilization = utilization
        self.n_episodes = n_episodes
        self.plot = plot
        self.name = 'DQN'

        if epsilon_deflator:
            self.epsilon_deflator = epsilon_deflator
            self.epsilon_max = epsilon_max
            self.epsilon_min = epsilon_min
            self.epsilon = epsilon_max
        else:
            self.epsilon = epsilon

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = Variable(torch.unsqueeze(torch.FloatTensor(observation), 0))

        if np.random.uniform() > self.epsilon:
            q_values = self.eval_net.forward(observation).data.numpy()[0]
            max_q = q_values.max()
            action = np.random.choice(np.where(q_values == max_q)[0])
        else:
            action = self.action_space[np.random.choice(self.n_actions)].squeeze()
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        sample_index = np.random.choice(self.memory_size, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :self.n_lines]))
        b_a = Variable(torch.LongTensor(b_memory[:, self.n_lines:self.n_lines + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, self.n_lines + 1:self.n_lines + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.n_lines:]))
        q_eval = self.eval_net(b_s).gather(1, b_a)
        values_s_ = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * values_s_.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1

    def record(self, score):
        if hasattr(self, 'epsilon_deflator'):
            self.epsilon = max(self.epsilon * self.epsilon_deflator, self.epsilon_min)
        if not CONSISTENT_ERROR:
            error = np.mean(self.errors) if len(self.errors) else np.nan
        else:
            error = calc_error(self.score_his)
        self.error_his.append(error)
        self.score_his.append(score)
        self.errors = []


class PolicyNet(nn.Module):
    def __init__(self, n_i, n_h, n_o):
        super(PolicyNet, self).__init__()
        self.bn0 = nn.BatchNorm1d(n_i)
        self.bn0.weight.data.fill_(1)
        self.bn0.bias.data.fill_(0)
        self.linear1 = nn.Linear(n_i, n_h)
        self.bn1 = nn.BatchNorm1d(n_h)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.fill_(0)
        self.linear2 = nn.Linear(n_h, n_h)
        self.bn2 = nn.BatchNorm1d(n_h)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.fill_(0)
        self.V = nn.Linear(n_h, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)
        self.mu = nn.Linear(n_h, n_o)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)
        self.L = nn.Linear(n_h, n_o**2)
        self.L.weight.data.mul_(0.1)
        self.L.bias.data.mul_(0.1)
        self.tril_mask = Variable(torch.tril(torch.ones(n_o, n_o), diagonal=-1).unsqueeze(0))
        self.diag_mask = Variable(torch.diag(torch.diag(torch.ones(n_o, n_o))).unsqueeze(0))

    def forward(self, inputs):
        x, u = inputs
        x = self.bn0(x)
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        mu = F.elu(self.mu(x))
        V = self.V(x)
        Q = None
        if u is not None:
            num_outputs = mu.size(1)
            L = self.L(x).view(-1, num_outputs, num_outputs)
            L = L * self.tril_mask.expand_as(L) + torch.exp(L) * self.diag_mask.expand_as(L)
            P = torch.bmm(L, L.transpose(2, 1))
            u_mu = (u - mu).unsqueeze(2)
            A = -0.5 * torch.bmm(torch.bmm(u_mu.transpose(2, 1), P), u_mu)[:, :, 0]
            Q = A + V
        return mu, Q, V


class NAF(Algo):
    def __init__(self, action_space, state_space, hidden_neurons, learning_rate=1e-3,
                 gamma=0.9, epsilon=0.9, epsilon_min=0, epsilon_max=1, epsilon_deflator=None,
                 update_per_iter=5, memory_size=500, batch_size=32, tau=0, utilization=None,
                 n_episodes=1000, plot=False, **args):
        super(NAF, self).__init__()
        self.action_space = action_space
        self.state_space = state_space
        self.n_actions = action_space.shape[0]
        self.n_states = state_space.shape[0]
        self.n_lines = state_space.shape[1]
        self.eval_net = PolicyNet(self.n_lines, hidden_neurons, self.n_actions)
        self.target_net = PolicyNet(self.n_lines, hidden_neurons, self.n_actions)
        self.optimizer = Adam(self.eval_net.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = np.zeros((self.memory_size, state_space.shape[1] * 2 + 3))
        self.memory_counter = 0
        self.update_per_iter = update_per_iter
        self.gamma = gamma
        self.tau = tau
        self.utilization = utilization
        self.errors = []
        self.n_episodes = n_episodes
        self.plot = plot
        self.agent = None
        self.name = 'NAF'

        if epsilon_deflator:
            self.epsilon_deflator = epsilon_deflator
            self.epsilon_max = epsilon_max
            self.epsilon_min = epsilon_min
            self.epsilon = epsilon_max
        else:
            self.epsilon = epsilon

        self.hard_update(self.target_net, self.eval_net)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)

    def loss_func(self, input, target):
        return torch.sum((input - target)**2) / input.data.nelement()

    def choose_action(self, observation):
        observation = torch.Tensor(observation)
        self.eval_net.eval()
        mu, _, _ = self.eval_net((Variable(observation.reshape(1, self.n_lines)), None))
        self.eval_net.train()
        action = mu.data.clamp(0, 10).numpy()[0]
        return self.perturbed(action)

    def learn(self):
        for _ in range(self.update_per_iter):
            sample_index = np.random.choice(self.memory_size, self.batch_size)
            b_memory = self.memory[sample_index, :]
            b_s = Variable(torch.FloatTensor(b_memory[:, :self.n_lines]))
            b_a = Variable(torch.FloatTensor(b_memory[:, self.n_lines:self.n_lines + 2]))
            b_r = Variable(torch.FloatTensor(b_memory[:, self.n_lines + 2:self.n_lines + 4]))
            b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.n_lines:]))
            _, _, values_s_ = self.target_net((b_s_, None))
            q_target = b_r + self.gamma * values_s_
            _, q_eval, _ = self.eval_net((b_s, b_a))
            loss = self.loss_func(q_eval, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 1)
            self.optimizer.step()
            self.soft_update(self.target_net, self.eval_net, self.tau)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def record(self, score):
        if hasattr(self, 'epsilon_deflator'):
            self.epsilon = max(self.epsilon * self.epsilon_deflator, self.epsilon_min)
        if not CONSISTENT_ERROR:
            error = np.mean(self.errors) if len(self.errors) else np.nan
        else:
            error = calc_error(self.score_his)
        self.error_his.append(error)
        self.score_his.append(score)
        self.errors = []

    def perturbed(self, a):
        return a * np.random.normal(loc=1, scale=self.epsilon, size=2)


def running_mean(x, window=RUNNING_WINDOW):
    x = np.nan_to_num(x)
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def calc_error(score_his, window=RUNNING_WINDOW):
    L = len(score_his)
    return np.nan if L < 2 else np.nanstd(score_his[L - min(L, window):L])
