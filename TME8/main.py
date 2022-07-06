from SACmodel import *
import gym
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter("runs/SAC"+datetime.now().strftime("%Y%m%d-%H%M%S"))

if __name__ == '__main__':
    #envname = 'Pendulum-v0'
    #envname = 'MountainCarContinuous-v0'
    envname = 'LunarLanderContinuous-v2'

    env = gym.make(envname)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound = [-env.action_space.high[0], env.action_space.high[0]]

    sac = SAC(obs_dim, act_dim, act_bound)

    MAX_EPISODE = 100
    MAX_STEP = 500
    update_every = 100
    batch_size = 50
    rewardList = []
    entropyList = []
    for episode in range(MAX_EPISODE):
        o = env.reset()
        ep_reward = 0
        for j in range(MAX_STEP):
            #env.render()
            a = sac.get_action(o)
            o2, r, d, _ = env.step(a)
            sac.replay_buffer.store(o, a, r, o2, d)

            if episode >= 50 and j % update_every == 0:
                for _ in range(update_every):
                    batch = sac.replay_buffer.sample_batch(batch_size)
                    e = sac.update(data=batch)
                    entropyList.append(e)
                    #writer.add_scalar('entropy', e, len(entropyList)) 
            o = o2
            ep_reward += r
            if d:
                break
        print('Episode:', episode, 'Reward:%i' % int(ep_reward))
        #writer.add_scalar('reward', ep_reward, episode) 