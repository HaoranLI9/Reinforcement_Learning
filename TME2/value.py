import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy

class ValueAgent(object):

    def __init__(self, env, epsilon, gamma):
        self.env = env
        self.V = np.zeros(env.nS)
        self.epsilon = epsilon
        self.gamma = gamma

    def next_best_action(self, s, V):
        action_values = np.zeros(self.env.nA)
        for a in range(self.env.nA):
            for prob, next_state, reward, done in self.env.P[s][a]:
                action_values[a] += prob * (reward + self.gamma * V[next_state])
        return np.argmax(action_values), np.max(action_values)

    def optimize(self):
        epsilon = self.epsilon
        delta = float("inf")
        round_num = 0

        while delta > epsilon:
            delta = 0
            print("\nValue Iteration: Round " + str(round_num))
            for s in self.env.P.keys():
                best_action, best_action_value = self.next_best_action(s, self.V)
                delta = max(delta, np.abs(best_action_value - self.V[s]))
                self.V[s] = best_action_value
            round_num += 1

        policy = np.ones(self.env.nS)*-1
        print(self.env.nS)
        for s in self.env.P.keys():
            best_action, best_action_value = self.next_best_action(s, self.V)
            policy[s] = best_action

        return policy



if __name__ == '__main__':


    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    #env.render()  # permet de visualiser la grille du jeu
    #env.render(mode="human") #visualisation sur la console
    states, mdp = env.getMDP()  # recupere le mdp et la liste d'etats
    print("Nombre d'etats : ",len(states))
    state, transitions = list(mdp.items())[0]
    print(state)  # un etat du mdp
    print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    # Execution avec un Agent
    agent = ValueAgent(env, epsilon = 1e-4, gamma=0.95)
    policy = agent.optimize()
    print(policy)
"""
    episode_count = 100
    reward = 0
    done = False
    rsum = 0

    for i in range(episode_count):
        obs = env.reset()
        print(obs)
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        while True:
            action = agent.next_best_action(obs)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()
"""