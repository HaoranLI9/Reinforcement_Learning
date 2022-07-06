import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy

class PolicyAgent:
    def __init__(self, env, epsilon, gamma):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

    def get_action(self, x):
        action =np.zeros(self.env.nA)
        action[x] = 1
        return action
    def policy_evaluation(self, policy):
        V = np.zeros(self.env.nS)
        epsilon = self.epsilon
        delta = float("inf")
        
        while delta > epsilon:
            delta = 0
            for s in self.env.P.keys():
                expected_value = 0
                for action, action_prob in enumerate(policy[s]):
                    for prob, next_state, reward, done in self.env.P[s][action]:
                        expected_value += action_prob * prob * (reward +  self.gamma * V[next_state])
                delta = max(delta, np.abs(V[s] - expected_value))
                V[s] = expected_value

        return V
    
    def next_best_action(self, s, V):
        action_values = np.zeros(self.env.nA)
        for a in range(self.env.nA):
            for prob, next_state, reward, done in self.env.P[s][a]:
                action_values[a] += prob * (reward + self.gamma * V[next_state])
        return np.argmax(action_values), np.max(action_values)

    
    def optimize(self):
        policy = np.tile(np.eye(self.env.nA)[1], (self.env.nS, 1))
        
        is_stable = False
        
        round_num = 0
        
        while not is_stable:
            is_stable = True
            
            print("\n policy iterations Round Number:" + str(round_num))
            round_num += 1
            
            #print("Current Policy")
            #print(np.reshape([self.get_action(entry) for entry in [np.argmax(policy[s]) for s in range(self.env.nS)]], (self.env.nS, self.env.nA)))

            V = self.policy_evaluation(policy)

            #for s in range(self.env.nS):
            for s in self.env.P.keys():
                action_by_policy = np.argmax(policy[s])
                best_action, best_action_value = self.next_best_action(s, V)
                policy[s] = np.eye(self.env.nA)[best_action]
                if action_by_policy != best_action:
                    is_stable = False
        policy = [np.argmax(policy[s]) for s in range(self.env.nS)]
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
    agent = PolicyAgent(env, epsilon = 1e-4, gamma=0.95)
    policy = agent.optimize()
    print(policy)

