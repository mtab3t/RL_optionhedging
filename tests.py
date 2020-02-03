import numpy as np

vec = []

dict1 = {'iteration': 1, 'price_vector': [100,101,102,103], 'delta_vector': [52,45,58,35], 'bs_delta_vector': [50,51,52,53]}
vec.append(dict1)

dict2 = {'iteration': 2, 'price_vector': [100,98,95,97], 'delta_vector': [52,45,40,45], 'bs_delta_vector': [50,48,40,47]}
vec.append(dict2)

dict3 = {'iteration': 3, 'price_vector': [100,105,95,100], 'delta_vector': [52,42,58,39], 'bs_delta_vector': [50,60,40,50]}
vec.append(dict3)

#print(vec)

np.save("tests.npy", vec)

vecload = np.load("tests.npy",  allow_pickle = True).tolist()
print(vecload[2])