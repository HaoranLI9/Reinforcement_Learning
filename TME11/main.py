import pdb
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import  tqdm
writer = SummaryWriter("runs/DDPG"+datetime.now().strftime("%Y%m%d-%H%M%S"))

if __name__ == '__main__':
    import multiagent.scenarios as scenarios
    scenario = scenarios.load("multiagent-particle-envs/multiagent/scenarios/simple_adversary.py").Scenario()
    world = scenario.make_world()
    from multiagent.environment import MultiAgentEnv
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    n_agents = env.n; dim_act = world.dim_p * 2 + 1
    obs = env.reset(); n_states = len(obs[1])

    n_episode = 100; max_steps = 100
    from maddpg import *
    maddpg = MADDPG(n_agents, n_states, dim_act )

    for i_episode in tqdm(range(n_episode)):
        obs = env.reset()
        
        tmp = np.zeros(10)
        for i in range(8):
            tmp[i] = obs[0][i]
        obs[0] = tmp
        
        obs = np.stack(obs)
        max_steps = 100; total_reward = 0
        adversaries_reward = 0; goodagent_reward = 0
        for t in range(max_steps ):
            actions = maddpg.produce_action(obs)
            obs_, reward, done, _ = env.step(actions.detach() )
            next_obs = None
            tmp = np.zeros(10)
            for i in range(8):
                tmp[i] = obs[0][i]
            obs[0] = tmp
            if t < max_steps - 1:
                next_obs = obs_

            for r in reward:
                total_reward += r

            maddpg.memory.push(obs, actions, next_obs, reward)
            obs = next_obs;
            maddpg.train(i_episode); #env.render()
        writer.add_scalar('reward/epoch', total_reward, i_episode)

        print('Episode: %u' % (i_episode + 1) )
        print('total reward = %f' % (total_reward) )
        maddpg.episode_done += 1