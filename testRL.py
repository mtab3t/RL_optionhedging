import gym
import gym_foo2
import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

env = gym.make('foo2-v0')
for i_episode in range(1):
    observation = env.reset()
    for j in range (90):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(done)
        if done:
            print("Episode finished after {} timesteps".format(j + 1))
            break
    env.close()

