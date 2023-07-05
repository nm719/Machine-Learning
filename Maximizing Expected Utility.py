def get_expectation(x_distribution, function):
    expectation = 0
    for x, p in x_distribution.items():
        expectation = expectation + p * function(x)
    return expectation

def get_s_prime_probability(s_prime, action, s0_distribution, transition_table):
    function_s_prime = lambda x: transition_table[x][action][s_prime]
    s_prime_probability = get_expectation(s0_distribution, function_s_prime)
    return s_prime_probability

def get_eu(s_prime_distribution_given_action, utility_table):
    function_eu = lambda x: s_prime_distribution_given_action[x]
    eu = get_expectation(utility_table, function_eu)
    return eu

def main():
    # s0_distribution: a dictionary representing the probability distribution of the present state
    s0_distribution = {0:0.125, 1:0.25, 2:0.0625, 3:0.0625, 4:0.25, 5:0.25}
    
    # utility_table: a dictionary representing the utility of each state
    utility_table = {0:2000, 1:-500, 2:100, 3:1000, 4:0, 5:-5000}

    # transition_table: a dictionary showing the possible results of actions
    # {s0:{a:{s':p}}}: p is the probability of the event (Start from s0, take action a, end up at s')
    transition_table = {0:{0:{0:1,1:0,2:0,3:0,4:0,5:0},
                        1:{0:0.05,1:0.9,2:0.05,3:0,4:0,5:0},
                        2:{0:0,1:0.05,2:0.9,3:0.05,4:0,5:0},
                        3:{0:0,1:0,2:0.05,3:0.9,4:0.05,5:0},
                        4:{0:0,1:0,2:0,3:0.05,4:0.9,5:0.05},
                        5:{0:0.05,1:0,2:0,3:0,4:0.05,5:0.9}},
                     1:{0:{0:0,1:1,2:0,3:0,4:0,5:0},
                        1:{0:0,1:0.05,2:0.9,3:0.05,4:0,5:0},
                        2:{0:0,1:0,2:0.05,3:0.9,4:0.05,5:0},
                        3:{0:0,1:0,2:0,3:0.05,4:0.9,5:0.05},
                        4:{0:0.05,1:0,2:0,3:0,4:0.05,5:0.9},
                        5:{0:0.9,1:0.05,2:0,3:0,4:0,5:0.05}},
                     2:{0:{0:0,1:0,2:1,3:0,4:0,5:0},
                        1:{0:0,1:0,2:0.05,3:0.9,4:0.05,5:0},
                        2:{0:0,1:0,2:0,3:0.05,4:0.9,5:0.05},
                        3:{0:0.05,1:0,2:0,3:0,4:0.05,5:0.9},
                        4:{0:0.9,1:0.05,2:0,3:0,4:0,5:0.05},
                        5:{0:0.05,1:0.9,2:0.05,3:0,4:0,5:0}},
                     3:{0:{0:0,1:0,2:0,3:1,4:0,5:0},
                        1:{0:0,1:0,2:0,3:0.05,4:0.9,5:0.05},
                        2:{0:0.05,1:0,2:0,3:0,4:0.05,5:0.9},
                        3:{0:0.9,1:0.05,2:0,3:0,4:0,5:0.05},
                        4:{0:0.05,1:0.9,2:0.05,3:0,4:0,5:0},
                        5:{0:0,1:0.05,2:0.9,3:0.05,4:0,5:0}},
                     4:{0:{0:0,1:0,2:0,3:0,4:1,5:0},
                        1:{0:0.05,1:0,2:0,3:0,4:0.05,5:0.9},
                        2:{0:0.9,1:0.05,2:0,3:0,4:0,5:0.05},
                        3:{0:0.05,1:0.9,2:0.05,3:0,4:0,5:0},
                        4:{0:0,1:0.05,2:0.9,3:0.05,4:0,5:0},
                        5:{0:0,1:0,2:0.05,3:0.9,4:0.05,5:0}},
                     5:{0:{0:0,1:0,2:0,3:0,4:0,5:1},
                        1:{0:0.9,1:0.05,2:0,3:0,4:0,5:0.05},
                        2:{0:0.05,1:0.9,2:0.05,3:0,4:0,5:0},
                        3:{0:0,1:0.05,2:0.9,3:0.05,4:0,5:0},
                        4:{0:0,1:0,2:0.05,3:0.9,4:0.05,5:0},
                        5:{0:0,1:0,2:0,3:0.05,4:0.9,5:0.05}}}

    action_space = [0,1,2,3,4,5]
    state_space = [0,1,2,3,4,5]

    s_prime_distribution = {a: {s: get_s_prime_probability(s, a, s0_distribution, transition_table) for s in state_space} for a in action_space}
    eu = {a: get_eu(s_prime_distribution[a], utility_table) for a in action_space}
    
    print(eu)
# {0: -1056.25, 1: -689.0624999999999, 2: 248.12499999999994, 3: -174.06250000000003, 4: -796.25, 5: -6.875}
# We can see that taking action 2 gives us the best expected utility given the probabilities of two things: the current state and the next state given an action.

if __name__ == '__main__':
    main()
