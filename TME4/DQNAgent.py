import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import gym
import gridworld
import torch
import copy
from utils import *
from core import *
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import highway_env
from matplotlib import pyplot as plt
import yaml
from datetime import datetime
from memory import Memory


class DQNAgent(object):
    def __init__(self, env, opt):
        self.opt=opt
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test=False
        self.nbEvents=0
        self.theta = torch.nn.Sequential(
                nn.Linear(env.observation_space.shape[0], 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, env.action_space.n)
        )
        self.thetaHat = torch.nn.Sequential(
                nn.Linear(env.observation_space.shape[0], 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, env.action_space.n)
        )
        self.optim = torch.optim.Adam(self.theta.parameters(), lr=0.0001)
        self.huberLoss = torch.nn.HuberLoss(reduction='none')
        self.D = Memory(1000)



    def  act(self, obs, epsilon):
        if np.random.rand(1) < epsilon:
            action = self.action_space.sample()
        else:
            with torch.no_grad():
                action = np.argmax(self.theta(torch.tensor(obs))).item()

        return action

    # sauvegarde du modèle
    def save(self,outputDir):
        torch.save(self.theta.state_dict(), outputDir)

    # chargement du modèle.
    def load(self,inputDir):
        self.theta = torch.load(inputDir)

    def make_batch(self, nb_samples):
        idx, w, batch = self.D.sample(nb_samples)
        state_batch = torch.tensor([])
        action_batch = torch.tensor([])
        reward_batch = torch.tensor([])
        next_state_batch = torch.tensor([])
        done_batch = torch.tensor([])
        idx_batch = torch.tensor(idx)
        w_batch = torch.tensor(w).view(-1)

        for tr in batch:
            ob, action, reward, new_ob, done = tr
            state_batch = torch.cat([state_batch, torch.tensor(ob)], 0)
            action_batch = torch.cat([action_batch, torch.tensor(action).view(1,1)], 0)
            reward_batch = torch.cat([reward_batch, torch.tensor(reward).view(1,1)], 0)
            next_state_batch = torch.cat([next_state_batch, torch.tensor(new_ob)])
            done_batch = torch.cat([done_batch, torch.tensor([done]).view(1,1)], 0)

        return state_batch, action_batch, next_state_batch, reward_batch.view(-1), done_batch.view(-1), idx_batch, w_batch


    def learn(self, nb_samples, gamma):
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            pass

        state , action, next_state, reward, done, idx, w = self.make_batch(nb_samples)

        q_values = self.theta(state)
        next_q_values = self.thetaHat(next_state)

        q_value = q_values.gather(1, action.long()).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss = self.huberLoss(q_value, expected_q_value) * w
        prios = loss.detach() + 1e-5
        loss = loss.mean()
        self.optim.zero_grad()
        loss.backward(retain_graph = True)
        self.D.update(idx, prios)
        self.optim.step()

        return loss.detach()


    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            tr = (ob, action, reward, new_ob, done)
            self.lastTransition=tr #ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)
            self.D.store(tr)

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0

if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_random_cartpole.yaml', "DQNAgent")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = DQNAgent(env,config)

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 30000

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * np.exp(
        -1. * frame_idx / epsilon_decay)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    eps = 1e-3
    gamma = 0.99
    C = 100
    batch_size = 32
    done = False

    for i in range(episode_count):
        checkConfUpdate(outdir, config)

        rsum = 0
        agent.nbEvents = 0
        ob = env.reset()

        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            #print("Test time! ")
            mean = 0
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            #print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()

        new_ob = agent.featureExtractor.getFeatures(ob)
        while True:
            if verbose:
                env.render()

            ob = new_ob
            action = agent.act(ob, epsilon_by_frame(i+1))
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)

            j+=1

            """# Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")"""

            agent.store(ob, action, new_ob, reward, done,j)
            rsum += reward

            if agent.timeToLearn(done) and agent.D.nentities > batch_size:
                loss = agent.learn(batch_size, gamma)


            if done:
                #print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0

                break

        if i % C == 0:
            agent.thetaHat.load_state_dict(agent.theta.state_dict())  # Target Network

    env.close()
