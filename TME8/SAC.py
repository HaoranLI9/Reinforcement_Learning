import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/DDPG"+datetime.now().strftime("%Y%m%d-%H%M%S"))


#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
TAU = 0.01
RENDER = False
ENV_NAME=  'Pendulum-v0'
#ENV_NAME = 'MountainCarContinuous-v0'
#ENV_NAME = 'LunarLanderContinuous-v2'

###############################  SAC  ####################################

class ANet(nn.Module):   # ae(s)=a
    """
    input : state
    output : stochastique action
    """
    def __init__(self,s_dim,a_dim):
        super(ANet,self).__init__()
        self.fc1 = nn.Linear(s_dim,30)
        self.fc1.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(30,a_dim)
        self.out.weight.data.normal_(0,0.1) # initialization
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.tanh(x)
        actions_value = x*2
        return actions_value

class CNetQ(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(CNetQ,self).__init__()
        self.fcs = nn.Linear(s_dim,30)
        self.fcs.weight.data.normal_(0,0.1) # initialization
        self.fca = nn.Linear(a_dim,30)
        self.fca.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(30,1)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self,s,a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x+y)
        q_value = self.out(net)
        return q_value

class CNetV(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(CNetV,self).__init__()
        self.fcs = nn.Linear(s_dim,30)
        self.fcs.weight.data.normal_(0,0.1) # initialization
        self.fca = nn.Linear(a_dim,30)
        self.fca.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(30,1)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self,s,a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x+y)
        v_value = self.out(net)
        return v_value


class SAC(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,

        # Memory
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0

        # Actor net
        self.Actor_eval = ANet(s_dim,a_dim)
        self.Actor_target = ANet(s_dim,a_dim)

        # Critic net
        self.CriticQ1 = CNetQ(s_dim,a_dim)
        self.CriticQ2 = CNetQ(s_dim,a_dim)
        self.CriticV_eval = CNetV(s_dim, a_dim)
        self.CriticV_target = CNetV(s_dim, a_dim)

        # Optim
        self.ctrain = torch.optim.Adam(self.CriticQ1.parameters(),lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(),lr=LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)

        return self.Actor_eval(s)[0].detach() # ae（s）

    def learn(self):

        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        for x in self.CriticQ1.state_dict().keys():
            eval('self.CriticQ1.' + x + '.data.mul_((1-TAU))')
            eval('self.CriticQ1.' + x + '.data.add_(TAU*self.CriticQ1.' + x + '.data)')

        # soft target replacement
        #self.sess.run(self.soft_replace)  # ae、ce update at，ct

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        b_state = torch.FloatTensor(bt[:, :self.s_dim])
        b_action = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        b_reward = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        b_state_ = torch.FloatTensor(bt[:, -self.s_dim:])

        a = self.Actor_eval(b_state)
        q = self.CriticQ1(b_state,a)  # loss=-q=-ce（s,ae（s））update ae   ae（s）=a   ae（s_）=a_
        # Si a est un comportement correct, alors son Q devrait être plus proche de 0
        normal = torch.distributions.Normal(0, 1)
        loga = torch.log(normal.rsample()).sum()
        loss_a = -torch.mean(q - a)
        #print(q)
        #print(loss_a)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_old = self.Actor_target(b_state_)
        q1 = self.CriticQ1(b_state_, a_old)
        q2 = self.CriticQ2(b_state_, a_old)
        y_q = b_reward + GAMMA * torch.minimum(q1, q2)

        delta_q1 = self.CriticQ1(b_state, b_action)
        delta_q2 = self.CriticQ2(b_state, b_action)

        td_error = self.loss_td(delta_q1, y_q) + self.loss_td(delta_q2, y_q)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

###############################  training  ####################################
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

sac = SAC(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        """
        if True:
            env.render()
        """

        # Add exploration noise
        a = sac.choose_action(s)
        s_, r, done, info = env.step(a)
        sac.store_transition(s, a, r / 10, s_)
        
        if sac.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            sac.learn()
        #if i>20:
            #ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, ) 
            #if ep_reward > -300:RENDER = True
            break
        """
        if done:
          print('Episode:', i, ' Reward: %i' % int(ep_reward))
          break
        """
    writer.add_scalar('reward/epoch', ep_reward, i) 
print('Running time: ', time.time() - t1)