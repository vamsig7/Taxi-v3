import gym
import numpy as np
env =gym.make("Taxi-v3")
env.reset()
print(env.observation_space)
print(env.action_space)

num_actions =  env.action_space.n
num_states = env.observation_space.n

q_table = np.zeros([num_states,num_actions])

gamma = 0.9
alpha=0.9
for episode in range(1,701):
    done =False
    reward_total=0
    old_state =env.reset()
    while not done:
        action = np.argmax(q_table[old_state])
        new_state,reward,done,info =env.step(action)
        q_table[old_state,action]+=alpha*(reward+gamma*np.max(q_table[new_state])-q_table[old_state,action])
        reward_total+=reward
        old_state=new_state


np.save('q_table.npy',np.array(q_table))
