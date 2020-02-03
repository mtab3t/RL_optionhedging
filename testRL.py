import gym
import gym_foo2
import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

env = gym.make('foo2-v0')
print(env.observation_space.shape)
print(env.action_space.shape)
for i_episode in range(1):
    observation = env.reset()
    done = False
    steps = 0
    while not(done):
        steps +=1
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation)
        print(done)
        if done:
            print("Episode finished after {} timesteps".format(steps))
            break
    env.close()

