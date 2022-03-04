import os
import numpy as np
import datetime as d
import random as r
import copy as c

from constants import *
from utilities import *
from data_gen import Robot, DataCollection
from ik_simulator import IKTable, IKSimulator

def fully_covered(iter):
    iktable = IKTable('raw_data_7j_20')
    # iktable = IKTable('dense')
    # iktable = IKTable('full_jointonly')
    for t in iktable.table:
        print(t.bounds)

    avg = []
    n_avg = []
    dev = []
    pos = []
    for i in range(iter):
        counter = 0
        n_counter = 0
        for _ in range(1000):

            x = round(r.uniform(-0.855, 0.855), 4)
            y = round(r.uniform(-0.855, 0.855), 4)
            z = round(r.uniform(-0.36, 1.19), 4)
            # x = round(r.uniform(0.2, 0.25), 4)
            # y = round(r.uniform(0.45, 0.5), 4)
            # z = round(r.uniform(0.3, 0.35), 4)
            # x = round(r.uniform(0.2, 0.21), 4)
            # y = round(r.uniform(0.4, 0.41), 4)
            # z = round(r.uniform(0.3, 0.31), 4)
            result = iktable.dot_query([x, y, z])
            if len(result):#in range
                pos.append(len(result))
                counter += 1
                # print(result)
            else:
                n_counter += 1
                dev.append(np.linalg.norm(np.array(list(iktable.table[0].nearest([x, y, z], objects=True))[0].bounds[::2]) - np.array([x, y, z])))
            # break
        # break
        print(i+1, counter)
        avg.append(counter)
        n_avg.append(n_counter)

    mes = {}
    mes['pos'] = np.mean(pos)
    mes['avg'] = np.mean(avg)
    mes['n_avg'] = np.mean(n_avg)
    mes['len'] = np.mean(dev)
    mes['worst'] = max(dev)
    messenger(mes)

def run_interpolation():
    pass

if __name__ == '__main__':
    print('start')
    start = d.datetime.now()

    fully_covered(100)


    print('duration: ', d.datetime.now()-start)
    print('end')
