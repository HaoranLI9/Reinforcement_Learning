import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random
import gym
import math
import pickle

import argparse
import sys
import matplotlib
matplotlib.use("TkAgg")
from torch.utils.tensorboard import SummaryWriter
from utils import *
from core import *
import yaml
writer = SummaryWriter("runs/GAIL"+datetime.now().strftime("%Y%m%d-%H%M%S"))
lr_a = 0.0002
lr_c = 0.0004
Capacity = 10000
num_episode = 1000
Gamma = 0.98
Lambda = 0.99
Beta = 1
K = 3
KL_target = 0.01
eps_clip = 0.1
loss_function = 1 #0 : no clipping or penalty, 1 : KL penalty, 2 : clipping

device = torch.device('cpu')

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

class Discriminator(nn.Module):
    def __init__(self, env):
        super(Discriminator, self).__init__()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.net = nn.Sequential(
            nn.Linear(self.action_dim+self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)

    def forward(self, states, actions):
        """
        pass both the (action, states) of critique net and that of expert through
        actions : one_hot 
        """
        input_net = torch.cat([actions, states], dim=-1).to(device)
        output_net = self.net(input_net)

        return output_net

class Expert():
    def __init__(self, nbFeatures, filename):
        self.target = torch.FloatTensor()
        self.nbFeatures = nbFeatures
        self.loadExpertTransitions(filename)


    def loadExpertTransitions(self, file):
        with open(file, 'rb') as handle:
            expert_data = pickle.load(handle).to(self.target)
            expert_states = expert_data[:,:self.nbFeatures]
            expert_actions = expert_data[:,self.nbFeatures:]
            self.states = expert_states.contiguous()
            self.actions = expert_actions.contiguous()

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
    def __init__(self, input_size, hidden_size, output_size, env):
        super(PPO, self).__init__()
        self.net = Net(input_size, hidden_size, output_size).to(device)
        self.discriminator = Discriminator(env)
        self.expert = Expert(env.observation_space.shape[0], 'expert.pkl')
        self.optim = optim.Adam(self.net.parameters(), lr=lr_a)
        self.buffer = ReplayBuffer(capacity=Capacity)

        self.bceloss = nn.BCELoss()

    def act(self, s, dim):
        prob = self.net.actor_forward(s, dim)
        return prob
    
    def discriminator(self, s, a):
        return self.discriminator(a, s)

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
            """
            train the Discriminator
            """

            action_batch_one_hot = torch.nn.functional.one_hot(action_batch.view(-1), num_classes=4)
            disc_actor = self.discriminator(state_batch, action_batch_one_hot.view(action_batch.size(0), -1))
            disc_expert = self.discriminator(self.expert.states, self.expert.actions)

            advantage_list = torch.cumsum(F.sigmoid(disc_actor.detach().flip(dims=[-1])), dim=-1)
            advantage = advantage_list / torch.cumsum(torch.ones(advantage_list.size()).to(device), dim=0)
            advantage = advantage.flip(dims=[-1]).to(device)

            prob = self.act(state_batch, 1).squeeze(0)
            prob_a = prob.gather(1, action_batch.view(-1, 1))
            ratio = torch.exp(torch.log(prob_a) - torch.log(rate_batch))
            if loss_function == 0:
                loss = - ratio * advantage + self.bceloss(F.sigmoid(disc_actor), torch.zeros(disc_actor.size()).to(device))\
                        + self.bceloss(F.sigmoid(disc_expert), torch.ones(disc_expert.size()).to(device))

            if loss_function == 1:
                surr1 = ratio * advantage
                surr2 = F.kl_div(prob, prob_batch, reduction='batchmean')
                loss = -surr1 + Beta * surr2 + self.bceloss(F.sigmoid(disc_actor), torch.zeros(disc_actor.size()))\
                        + self.bceloss(F.sigmoid(disc_expert), torch.ones(disc_expert.size()))

            if loss_function == 2:
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
                loss = -torch.min(surr1, surr2) + self.bceloss(F.sigmoid(disc_actor), torch.zeros(disc_actor.size()))\
                        + self.bceloss(F.sigmoid(disc_expert), torch.ones(disc_expert.size()))

            self.optim.zero_grad()
            loss.mean().backward()
            self.optim.step()



        if loss_function == 1:
            if surr2 >= 1.5 *  KL_target:
                Beta *= 2
            if surr2 <= KL_target / 1.5:
                Beta *= 0.5



if __name__ == '__main__':
    #env = gym.make('LunarLander-v2')
    env = gym.make('LunarLander-v2')
    expert = Expert(env.observation_space.shape[0], 'expert.pkl')
    expert.loadExpertTransitions('expert.pkl')
    print(expert.states.shape)
    print(expert.actions.shape)
    print(env.action_space)
    print(env.observation_space.shape[0])

    #env, config, outdir, logger = init('./configs/config_random_lunar.yaml', "PPOAgent")
    #env = gym.make('CartPole-v0')
    Agent = PPO(env.observation_space.shape[0], 256, env.action_space.n, env)
    #discriminator = Discriminator(env)
    average_reward = 0
    for i_episode in range(num_episode):
        s0 = env.reset()
        tot_reward = 0

        while True:
            #env.render()
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
                if i_episode % 1 == 0:
                    print('Episode ', i_episode,
                      ' tot_reward: ', tot_reward, ' average_reward: ',
                      average_reward)
                break

        #logger.direct_write("reward", tot_reward, i_episode)
        writer.add_scalar("reward", tot_reward, i_episode)
        Agent.update_parameters()
        Agent.buffer.clean()