import numpy as np
from collections import defaultdict

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

def pos_alignment(position):
    for i,v in enumerate(position):
        position[i] = round(v, 4)

    return position

def show_avg(filename):
    message = np.load(RESULT_FOLDER+filename+'.npy', allow_pickle=True)

    mes = defaultdict(list)
    gdiff = []
    bdiff = []
    for m in message:
        if not m['posture']:
            # print(m['target'])
            continue
        # improvement = m['origin_diff']-m['mean_diff']
        # if m['origin_diff'] >= 0.06:
        # if m['worst_diff'] < 0.005:
        if True:
            for k, v in m.items():
                mes[k].append(v)

        #     gdiff.append(improvement)
        # bdiff.append(improvement)

    # print(gdiff)
    # print(bdiff)
    print('query', len(message))
    print('respond', len(mes['posture']))

    result = {}
    # print(mes)
    for k, v in mes.items():
        if k == 'target':
            continue
        elif k == 'worst_diff':
            result[k] = np.max(v)
        # elif k == 'posture' or k == 'worse_num' or k == 'avg. time' or k == 'total time' or k == 'result_post':
        #     result[k] = np.mean(v, axis=0)
        elif k == 'posture':
            result['pos_min'] = np.min(v)
            result[k] = np.mean(v, axis=0)
        elif k == 'worse_num' or k == 'avg. time' or k == 'total time' or k == 'result_post' or k == 'input_post':
            result[k] = np.mean(v, axis=0)
        else:
            result[k] = np.average(v, axis=0, weights=mes['posture'])
        # print(v)
    result['worst%'] = (1 - result['worst%']) * 100
    result['improve'] = (1 - ((result['origin_diff']-result['mean_diff'])/result['origin_diff'])) * 100

    messenger(result)

def show_sparse(filename):
    message = np.load(RESULT_FOLDER+filename+'.npy', allow_pickle=True)

    mes = defaultdict(list)
    gdiff = []
    bdiff = []
    for m in message:
        if not m['posture']:
            print(m['target'])
            continue
        # improvement = m['origin_diff']-m['mean_diff']
        # if m['origin_diff'] >= 0.05:
        # if m['posture'] < 20:
        if m['target'] == [-0.5906, 0.6227, -0.1446]:
        # if True:
            # print(m['posture'], m['target'])
            messenger(m)


# def str2trans(key_str):
#     return [float(k) for k in str(key_str)[1:-1].split(' ')]
