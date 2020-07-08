import gym
import numpy as np 


env =gym.make("Taxi-v3")
env.reset()
print(env.observation_space)
print(env.action_space)

num_actions = 6
num_states = 500
values = np.zeros(500)
pi = np.zeros(500,dtype=int)

gamma =0.9 #impact of present on future scaled to 0.9

exit_factor =0.01

def best_action_value(state):
    best_action =None
    best_value =float('-inf')
    # check all possible outcomes from state s  if action performed
    for action in range(0,num_actions):
        env.env.s = state
        observation,reward,done,info = env.step(action)
        v = reward +gamma*values[observation]
        if v>best_value:
            best_value=v
            best_action=action
    return best_action,best_value
    

iter_count=0
while True:
    biggest_change=0
    # check all for all possible states and update value vector
    for state in range(0,500):
        old_val =values[state]
        action,new_val = best_action_value(state)
        values[state]=new_val
        pi[state]=action
        biggest_change = max(biggest_change,np.abs(new_val-old_val))
    iter_count+=1
    # change in value updation is very low then break
    if biggest_change<exit_factor:
        break

pi_arr = np.array(pi)
np.save('pi_val.npy',pi_arr)



