import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random
import gym
import math

import argparse
import sys
import matplotlib
matplotlib.use("TkAgg")
from torch.utils.tensorboard import SummaryWriter
from utils import *
from core import *
import yaml

lr_a = 0.0002
lr_c = 0.0004
Capacity = 10000
num_epidose = 10000
Gamma = 0.98
Lambda = 0.99
Beta = 1
K = 3
KL_target = 0.01
eps_clip = 0.1
loss_function = 0 #0 : no clipping or penalty, 1 : KL penalty, 2 : clipping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, input_size,hidden_size, output_size):
        super(Net, self).__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        # self.Linear2 = nn.Linear(hidden_size, hidden_size)
        self.Linear_actor = nn.Linear(hidden_size, output_size)
        self.Linear_critic = nn.Linear(hidden_size, 1)

    def actor_forward(self, s, dim):
        s = torch.tanh(self.Linear1(s))
        prob = F.softmax(self.Linear_actor(s), dim=dim)
        return prob

    def critic_forward(self, s):
        s = torch.tanh(self.Linear1(s))
        # s = F.relu(self.Linear2(s))
        return self.Linear_critic(s)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'rate', 'prob', 'done'))


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def clean(self):
        self.position = 0
        self.memory = []

class PPO(object):
    def __init__(self, input_size, hidden_size, output_size):
        super(PPO, self).__init__()
        self.net = Net(input_size, hidden_size, output_size).to(device)
        self.optim = optim.Adam(self.net.parameters(), lr=lr_a)
        self.buffer = ReplayBuffer(capacity=Capacity)

    def act(self, s, dim):
        prob = self.net.actor_forward(s, dim)
        return prob

    def critic(self, s):
        return self.net.critic_forward(s)

    def put(self, s0, a0, r, s1, rate, prob, done):
        self.buffer.push(s0, a0, r, s1, rate, prob, done)


    def update_parameters(self):
        global Beta, K
        samples = self.buffer.memory
        batch = Transition(*zip(*samples))
        state_batch = torch.Tensor(np.array(batch.state)).to(device)
        action_batch = torch.LongTensor(np.array(batch.action)).view(-1, 1).to(device)
        reward_batch = torch.Tensor(np.array(batch.reward)).view(-1, 1).to(device)
        next_state_batch = torch.Tensor(np.array(batch.next_state)).to(device)
        rate_batch = torch.Tensor(np.array(batch.rate)).view(-1, 1).to(device)
        prob_batch = torch.Tensor(np.array(batch.prob)).view(rate_batch.size(0), -1).to(device)
        done_batch = torch.FloatTensor(np.array(batch.done)).view(-1, 1).to(device)

        for i in range(K):
            td_target = reward_batch + Gamma * self.critic(next_state_batch) * done_batch # TD(0)
            delta = td_target - self.critic(state_batch)
            delta = delta.detach().cpu().numpy()

            advantage_list = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = Lambda * advantage + delta_t
                advantage_list.append(advantage) #TD(lambda)

            advantage_list.reverse()
            advantage = torch.Tensor(np.array(advantage_list)).to(device)
            prob = self.act(state_batch, 1).squeeze(0)
            prob_a = prob.gather(1, action_batch.view(-1, 1))
            ratio = torch.exp(torch.log(prob_a) - torch.log(rate_batch))
            if loss_function == 0:
                loss = - ratio * advantage + F.smooth_l1_loss(self.critic(state_batch), td_target.detach())

            if loss_function == 1:
                surr1 = ratio * advantage
                surr2 = F.kl_div(prob, prob_batch, reduction='batchmean')
                loss = -surr1 + Beta * surr2 + F.smooth_l1_loss(self.critic(state_batch), td_target.detach())

            if loss_function == 2:
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.critic(state_batch), td_target.detach())

            self.optim.zero_grad()
            loss.mean().backward()
            self.optim.step()



        if loss_function == 1:
            if surr2 >= 1.5 *  KL_target:
                Beta *= 2
            if surr2 <= KL_target / 1.5:
                Beta *= 0.5




if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_random_cartpole.yaml', "PPOAgent")
    #env = gym.make('CartPole-v0')
    Agent = PPO(env.observation_space.shape[0], 256, env.action_space.n)
    average_reward = 0
    for i_episode in range(num_epidose):
        s0 = env.reset()
        tot_reward = 0
        while True:
            env.render()
            prob = Agent.act(torch.from_numpy(s0).float().to(device), 0)
            a0 = int(prob.multinomial(1))
            s1, r, done, _ = env.step(a0)
            rate = prob[a0].item()
            Agent.put(s0, a0, r, s1, rate, prob.detach().cpu().numpy(), 1 - done)
            s0 = s1
            tot_reward += r
            if done:
                average_reward = average_reward + 1 / (i_episode + 1) * (
                        tot_reward - average_reward)
                if i_episode % 20 == 0:
                    print('Episode ', i_episode,
                      ' tot_reward: ', tot_reward, ' average_reward: ',
                      average_reward)
                break

        logger.direct_write("reward", tot_reward, i_episode)
        Agent.update_parameters()
        Agent.buffer.clean()