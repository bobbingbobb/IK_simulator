import numpy as np

from constants import *

def messenger(message):
    for k, v in message.items():
        print(k+':\t'+str(v))

def name_alignment(name):
    name = str(name).split('/')
    name = name[-1].split('.')
    return name[0]

def density(data, data_dim):
    def recur(list, dim):
        result = []
        dim -= 1
        for element in list:
            # print(len(element))
            if not dim == 0:
                result += recur(element, dim)
            else:
                if not len(element) == 0:
                    result += [len(element)]
        return result

    return np.mean(np.array(recur(data, data_dim)))

# def str2trans(key_str):
#     return [float(k) for k in str(key_str)[1:-1].split(' ')]
