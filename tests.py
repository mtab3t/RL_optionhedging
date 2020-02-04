import numpy as np
import gym
import gym_foo2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply, concatenate
from keras.optimizers import Adam
import keras.backend as K
from keras.models import model_from_json

import tensorflow as tf

import random
from collections import deque
import time

#
# vec = []
#
# dict1 = {'iteration': 1, 'price_vector': [100,101,102,103], 'delta_vector': [52,45,58,35], 'bs_delta_vector': [50,51,52,53]}
# vec.append(dict1)
#
# dict2 = {'iteration': 2, 'price_vector': [100,98,95,97], 'delta_vector': [52,45,40,45], 'bs_delta_vector': [50,48,40,47]}
# vec.append(dict2)
#
# dict3 = {'iteration': 3, 'price_vector': [100,105,95,100], 'delta_vector': [52,42,58,39], 'bs_delta_vector': [50,60,40,50]}
# vec.append(dict3)
#
# #print(vec)
#
# np.save("tests.npy", vec)

#vecload = np.load("simulations.npy",  allow_pickle = True).tolist()
#print(vecload)

# load json and create model
json_file = open('actor_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.compile(loss="mse", optimizer='adam')
# load weights into new model
loaded_model.load_weights("actor_model.h5")
print("Loaded model from disk")

print('test model')

obs = np.array([[np.log(110/100),10]])
print(obs)
y =[]
output = loaded_model.predict(obs)
print(output)
