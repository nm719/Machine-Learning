import numpy as np
import draw_heat_map as hm
import reward_table as rt
import transition_table as tt

def expect(x_distribution, function):
    expectation = sum([function(x) * px for x, px in x_distribution.items()])
    return expectation

def get_s_prime_distribution_full(s, action, transition_table, reward_table):
    reward = lambda s_prime: reward_table[s][action][s_prime]
    p = lambda s_prime: transition_table[s][action][s_prime]
    s_prime_distribution = {(s_prime, reward(s_prime)): p(s_prime) for s_prime in transition_table[s][action].keys()}
    return s_prime_distribution

def update_Q_full(s, a, Q, get_s_prime_distribution, gamma):
    s_prime_distribution = get_s_prime_distribution(s, a)
    Q_s_a = sum([p_sp * (sp[1] + gamma * max(Q[sp[0]].values())) for sp, p_sp in s_prime_distribution.items()])
    return Q_s_a

def Q_value_iteration(Q, update_Q, state_space, action_space, convergence_tolerance):
    while True:
        par = 0
        Q_new = Q
        for s in state_space:
            for a in action_space:
                if abs(update_Q(s, a, Q) - Q[s][a]) > convergence_tolerance:
                    Q_new[s][a] = update_Q(s, a, Q_new)
                    par = par + 1
        if par == 0:
            break
    return Q_new

def get_policy_full(Q, rounding_tolerance):
    max_val = max(Q.values())
    policy_val = [k for k, v in Q.items() if abs(v - max_val) < rounding_tolerance]
    policy = {p: (1 / len(policy_val)) for p in policy_val}
    return policy

def main():

    min_x, max_x, min_y, max_y = (0, 3, 0, 2)

    action_space = [(0,1), (0,-1), (1,0), (-1,0)]
    state_space = [(i, j) for i in range(max_x + 1) for j in range(max_y + 1) if (i, j) != (1, 1)]
    Q = {s: {a: 0 for a in action_space} for s in state_space}
    
    normal_cost = -0.04
    trap_dict = {(3,1): -1}
    bonus_dict = {(3,0): 1}
    block_list = [(1, 1)]
    
    p = 0.8
    transition_probability = {'forward': p, 'left': (1-p) / 2, 'right': (1-p) / 2, 'back': 0}
    transition_probability = {move: p for move, p in transition_probability.items() if transition_probability[move] != 0}
    
    transition_table = tt.create_transition_table(min_x, min_y, max_x, max_y, trap_dict, bonus_dict, block_list, action_space, transition_probability)
    reward_table = rt.create_reward_table(transition_table, normal_cost, trap_dict, bonus_dict)

    s_prime_distribution = lambda s, action: get_s_prime_distribution_full(s, action, transition_table, reward_table)
    gamma = 0.8
    update_Q = lambda s, a, Q: update_Q_full(s, a, Q, s_prime_distribution, gamma)
    
    convergence_tolerance = 1e-7
    Q_new = Q_value_iteration(Q, update_Q, state_space, action_space, convergence_tolerance)
    
    rounding_tolerance = 1e-7
    get_policy = lambda Q: get_policy_full(Q, rounding_tolerance)
    policy = {s: get_policy(Q_new[s]) for s in state_space}
    
    V = {s: max(Q_new[s].values()) for s in state_space}
    
    V_drawing = V.copy()
    V_drawing[(1, 1)] = 0
    V_drawing = {k: v for k, v in sorted(V_drawing.items(), key = lambda item: item[0])}
    policy_drawing = policy.copy()
    policy_drawing[(1, 1)] = {(1, 0): 1.0}
    policy_drawing = {k: v for k, v in sorted(policy_drawing.items(), key = lambda item: item[0])}

    hm.draw_final_map(V_drawing, policy_drawing, trap_dict, bonus_dict, block_list, normal_cost)
        
if __name__=='__main__': 
    main()


