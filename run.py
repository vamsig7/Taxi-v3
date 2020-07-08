import gym
import numpy as np

env =gym.make("Taxi-v3")
env.reset()





pi = np.load('pi_val.npy')
q_table = np.load('q_table.npy')
reward_total =0
env.render()
observation =env.reset()
done =False
while not done:
    # action = pi[observation]
    action =np.argmax(q_table[observation])
    observation,reward,done,info = env.step(action)
    reward_total+=reward
    env.render()
print(reward_total)