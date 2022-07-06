import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

import argparse
import sys
import matplotlib
matplotlib.use("TkAgg")
from torch.utils.tensorboard import SummaryWriter
from utils import *
from core import *
import yaml

# Hyper Parameters for Actor
GAMMA = 0.9  # discount factor, 0.9 for Cartpole, 0.9999 for Gridworld
actor_lr = 0.01 # 0.001 for Cartpole
critic_lr = 0.01 # 0.01 for Cartpole
critic_methode = 0 # 0 with R; 1 with TD(0); 2 with TD(lambda)
LAMBDA = 0.99 # TD(lambda)

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False



class PGNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PGNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, action_dim)
        nn.init.normal_(self.fc1.weight, 0, 0.1)
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.normal_(self.fc2.weight, 0, 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out


class Actor(object):
    # dqn Agent
    def __init__(self, env):
        self.env = env
        if env.observation_space.__class__.__name__ == 'Box':
            self.state_dim = env.observation_space.shape[0]
        else:
            self.state_dim = 1
        self.action_dim = env.action_space.n

        # init network parameters
        self.network = PGNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=actor_lr)

        # init some parameters
        self.time_step = 0

    def choose_action(self, observation):
        observation = torch.FloatTensor(observation).view(-1).to(device)
        with torch.no_grad():
            network_output = self.network(observation)
            prob_weights = F.softmax(network_output, dim=-1).cuda().data.cpu().numpy()
        #print(prob_weights)
        #prob_weights = F.softmax(network_output, dim=0).detach().numpy()
        action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights)  # select action w.r.t the actions prob
        return action

    def learn(self, samples, actions, td_error):
        self.time_step += 1
        # Step 1:
        softmax_input = self.network.forward(samples[:-1])
        actions = torch.tensor(actions).view(-1,).long().to(device)
        neg_log_prob = -F.cross_entropy(input=softmax_input, target=actions, reduction='none').to(device)
        if critic_methode == 0:
            neg_log_prob = -neg_log_prob
        #print(neg_log_prob.shape)
        #print(softmax_input[0])
        #softmax_input = torch.log(softmax_input[0])

        # Step 2:
        #s, s_ = torch.FloatTensor(state).to(device), torch.FloatTensor(next_state).to(device)
        #v = ccc.forward(s)     # v(s)
        #v_ = ccc.forward(s_)   # v(s')
        #td_error = reward + GAMMA * v_ - v
        loss_a = torch.sum(neg_log_prob * td_error)

        self.optimizer.zero_grad()
        loss_a.backward()
        self.optimizer.step()

    
    def sample(self, observation):
        obs = observation
        actions = []
        rewards = []
        samples = []
        while True:
            samples.append(obs)
            action = self.choose_action(obs)
            obs, reward, done, _ = self.env.step(action)
            if self.state_dim == 1:
                obs = [self.env.getStateFromObs(obs)]
            actions.append(action)
            if not done and self.state_dim == 1:
                reward = 0
            rewards.append(reward)

            if done:
                samples.append(obs)
                if reward > 0:
                    print(reward)
                if reward > 0 and self.state_dim == 1:
                    rewards[-1] = 10
                break

        return samples, actions, rewards
    

# Hyper Parameters for Critic
EPSILON = 0.99  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch
REPLACE_TARGET_FREQ = 10  # frequency to update target Q network

class CriticNet(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        
        super(CriticNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, 1)
        nn.init.normal_(self.fc1.weight, 0, 0.1)
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.normal_(self.fc2.weight, 0, 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

class Critic(object):
    def __init__(self, env):
        self.action_dim = env.action_space.n

        if env.observation_space.__class__.__name__ == 'Box':
            self.state_dim = env.observation_space.shape[0]
        else:
            self.state_dim = 1

        # init network parameters
        self.network = CriticNet(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=critic_lr)

        # init some parameters
        self.time_step = 0
        self.epsilon = EPSILON

    def learn(self, samples, rewards):
        ls_reward = torch.FloatTensor(rewards).to(device)
        # s, s_ = torch.FloatTensor(state).to(device), torch.FloatTensor(next_state).to(device)
        # forward
        a_hat = self.network(samples[:-1]).view(-1)

        # backward
        if critic_methode == 0:
            huberloss = nn.HuberLoss()
            loss_q = huberloss(ls_reward, a_hat)
        elif critic_methode == 1:
            loss_q = ls_reward + GAMMA * self.network(samples[1:]).view(-1,) - self.network(samples[:-1]).view(-1,)
            loss_q = loss_q.mean()
        else:
            loss_q = self.valeur_but(samples, rewards) - self.network(samples[:-1]).view(-1,)
            loss_q = loss_q.mean()

        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()

        with torch.no_grad():
            """print('ls_reward: ', ls_reward.size())
            print('sample: ',self.network(samples[1:]).size() )
            print('sample: ', self.network(samples[1:]).size())"""
            if critic_methode == 0:
                td_error = ls_reward - self.network(samples[:-1]).view(-1,)
            elif critic_methode == 1:
                td_error = ls_reward + GAMMA * self.network(samples[1:]).view(-1,) - self.network(samples[:-1]).view(-1,)
            else:
                td_error = self.valeur_but(samples, rewards) - self.network(samples[:-1]).view(-1, )

        return td_error

    def valeur_but(self, samples, rewards):
        l = len(rewards)

        if l == 1:
            return torch.FloatTensor([rewards[-1]]).to(device)
        last_sample_hat = self.network(samples[-1].view(-1)).item()

        ls_rewards = [0 for _ in range(l)]
        ls_rewards[-1] = rewards[-1]
        ls_rewards[-2] = rewards[-2] + GAMMA * last_sample_hat

        for j in range(l - 3, -1, -1):
            ls_rewards[j] = rewards[j] + GAMMA * ls_rewards[j + 1]


        return torch.FloatTensor(ls_rewards).to(device)


# Hyper Parameters
ENV_NAME = 'gridworld-v0'
EPISODE = 10000  # Episode limitation
STEP = 3000  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode
NUMBER_SAMPLES = 10

if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_random_lunar.yaml', "A2CAgent")
    #env = gym.make(ENV_NAME)
    actor = Actor(env)
    critic = Critic(env)

    for episode in range(EPISODE):
        # initialize task
        sum_r = 0
        state = env.reset()
        # Sample
        if env.observation_space.__class__.__name__ != 'Box':
            state = [env.getStateFromObs(state)]

        samples, actions, rewards = actor.sample(state)
        ls_samples = torch.tensor(np.array(samples)).float().to(device)

        # Train
        td_error = critic.learn(ls_samples, rewards)
        actor.learn(ls_samples, actions, td_error)

        logger.direct_write("reward", torch.sum(torch.tensor(rewards)), episode)
        logger.direct_write("etapes", len(samples), episode)
