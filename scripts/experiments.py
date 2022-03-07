import os
import numpy as np
import datetime as d
import random as r
import copy as c

from constants import *
from utilities import *
from data_gen import Robot, DataCollection
from ik_simulator import IKTable, IKSimulator

from ikpy.chain import Chain
import ikpy.utils.plot as plot_utils

chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], last_link_vector=[0, 0, 0])#, active_links_mask=[False, True, True, True, True, True, True, True, False, False])


def fully_covered(iter):
    # iktable = IKTable('raw_data_7j_20')
    # iktable = IKTable('dense')
    iktable = IKTable('full_jointonly_fixed1')
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

            # x = round(r.uniform(-0.855, 0.855), 4)
            # y = round(r.uniform(-0.855, 0.855), 4)
            # z = round(r.uniform(-0.36, 1.19), 4)
            # x = round(r.uniform(0.2, 0.25), 4)
            # y = round(r.uniform(0.45, 0.5), 4)
            # z = round(r.uniform(0.3, 0.35), 4)
            x = round(r.uniform(0.2, 0.21), 4)
            y = round(r.uniform(0.4, 0.41), 4)
            z = round(r.uniform(0.3, 0.31), 4)

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
    mes['worst'] = max(dev) if dev else 0
    messenger(mes)

def current_ik_speed(iter):
    time = []
    dev = []
    n = 0
    robot = Robot()
    q = np.zeros(7)
    for j in range(6):
        q[j] = r.uniform(robot.joints[j].min, robot.joints[j].max)

    for i in range(iter):
        x = round(r.uniform(-0.855, 0.855), 4)
        y = round(r.uniform(-0.855, 0.855), 4)
        z = round(r.uniform(-0.36, 1.19), 4)
        target = [x, y, z]

        s = d.datetime.now()
        tmp_joint = chain.inverse_kinematics(target, initial_position=[0]*10)[1:8]
        # tmp_joint = chain.inverse_kinematics(target, initial_position=[0, *q, 0, 0])[1:8]
        e = d.datetime.now()

        fk, _ = robot.fk_dh(tmp_joint)
        deviation = np.linalg.norm(np.array(target) - np.array([round(p, 4) for p in fk]))
        print(target, fk)
        print(deviation)
        if not target == [round(p, 4) for p in fk]:
            dev.append(deviation)
            n += 1
        time.append(e-s)

    mes = {}
    mes['deviation'] = np.mean(dev)
    mes['failed'] = n
    mes['avg. time'] = np.mean(np.array(time))

    print(mes)
    print(np.mean(np.array(time)))

if __name__ == '__main__':
    print('start')
    start = d.datetime.now()

    fully_covered(10)
    # current_ik_speed(1000)


    print('duration: ', d.datetime.now()-start)
    print('end')
