import numpy as np
import os

path = "./dataset"
actions = [label.split('.')[0] for label in os.listdir(path)]
for action in actions:
    data = np.load(path + f'/{action}.npy')
    shape = data.shape[0]
    data = np.concatenate((data, np.zeros((100 - shape, 5, 100))), axis=0)
    print(data.shape)
    np.save(os.path.join(path, f"{action}.npy"), data)
