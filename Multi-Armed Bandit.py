import numpy as np
import matplotlib.pyplot as plt

# Q: dictionary with keys as actions and values as average reward
# e: scalar that is used to determine the degree of randomness

def e_greedy(Q, e):
    p = np.random.random()
    # checking if we will select at random
    if p <= e:
        random_select = np.random.randint(len(Q))
        action = list(Q)[random_select]
    else:
        action_max = max(Q.values())
        action_best = [action for action, reward in Q.items() if reward == action_max]
        action = np.random.choice(action_best)
    return action


# N: dictionary with keys as actions and values as the frequency
# c: scalar that can be used to put varying weights on the frequency

def upper_confidence_bound(Q, N, c):
    t = sum(list(N.values())) + 1
    if 0 in list(N.values()):
        action_best = [k for k, v in N.items() if v == 0]
        action = np.random.choice(action_best)
    else:
        bound = [q + c * np.sqrt(np.log(t) / n) for q, n in zip(list(Q.values()), list(N.values()))]
        bound_max = max(bound)
        action_best = [q for q, b in zip(Q.keys(), bound) if b == bound_max]
        action = np.random.choice(action_best)
    return action

def update_QN(action, reward, Q, N):
    N_new = N.copy()
    N_new[action] = N_new[action] + 1
    Q_new = Q.copy()
    Q_new[action] = Q_new[action] + 1/N_new[action] * (reward - Q_new[action]) 
    return Q_new, N_new

def decide_multiple_steps(Q, N, policy, bandit, max_steps):
    action_reward = []
    for i in range(max_steps):
        action = policy(Q, N)
        reward = bandit(action)
        action_reward.append((action, reward))
        Q, N = update_QN(action, reward, Q, N)
    return {'Q': Q, 'N': N, 'action_reward': action_reward}

def plot_mean_reward(action_reward,label):
    max_steps = len(action_reward)
    reward = [reward for (action,reward) in action_reward]
    mean_reward = [sum(reward[:(i+1)])/(i+1) for i in range(max_steps)]
    plt.plot(range(max_steps), mean_reward, linewidth = 0.9, label = label)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')

def get_samplar():
    mu = np.random.normal(0, 10)
    sd = abs(np.random.normal(5, 2))
    get_sample = lambda: np.random.normal(mu, sd)
    return get_sample

def main():
    np.random.seed(2023)
    K = 10
    max_steps = 1000
    Q = {k: 0 for k in range(K)}
    N = {k: 0 for k in range(K)}
    test_bed = {k: get_samplar() for k in range(K)}
    bandit = lambda action: test_bed[action]()
    
    policies = {}
    policies["e-greedy-0.1"] = lambda Q, N: e_greedy(Q, 0.1)
    policies["e-greedy-0.3"] = lambda Q, N: e_greedy(Q, 0.3)
    policies["e-greedy-0.5"] = lambda Q, N: e_greedy(Q, 0.6)
    policies["UCB-1"] = lambda Q, N: upper_confidence_bound(Q, N, 1)
    policies["UCB-30"] = lambda Q, N: upper_confidence_bound(Q, N, 30)
    policies["UCB-60"] = lambda Q, N: upper_confidence_bound(Q, N, 60)
    
    all_results = {name: decide_multiple_steps(Q, N, policy, bandit, max_steps) for (name, policy) in policies.items()}
    
    for name, result in all_results.items():
         plot_mean_reward(all_results[name]['action_reward'], label = name)
    plt.legend(bbox_to_anchor = (0, 1, 1, 0), loc = 'lower left', ncol = 2, mode = "expand", borderaxespad = 0)
    plt.show()

if __name__=='__main__':
    main()

